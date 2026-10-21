import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import ex6
from ex6 import Context, Message
from _ex6.models import M
from _ex6.tools import ENV_PROMPT, _approve, ask_user_question, git_working_tree, glob, read_audio, read_file

API_ROOT = "https://freesound.org/apiv2"
FIELDS = "id,name,username,license,type,duration,avg_rating,num_ratings,previews"
MAX_RESULTS = 30
MAX_PREVIEW_BYTES = 20 * 1024 * 1024
SORTS = {"score", "duration_asc", "duration_desc", "created_asc", "created_desc",
         "downloads_asc", "downloads_desc", "rating_asc", "rating_desc"}
LICENSES = {
    ("creativecommons.org", "/publicdomain/zero/1.0"):
        ("CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/"),
    ("creativecommons.org", "/licenses/by/4.0"):
        ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),
}
INSTALL_LOCK = threading.Lock()
USER_AGENT = "ex6-sound-agent/1.0"


def _api_key(ctx: Context) -> str:
    load_dotenv(dotenv_path=ctx.resolve(".env"))
    key = os.getenv("FREESOUND_API_KEY", "").strip()
    if not key:
        raise ValueError("FREESOUND_API_KEY is missing from project .env.")
    return key


def _request(ctx: Context, endpoint: str, params: Optional[dict] = None) -> dict:
    url = f"{API_ROOT}/{endpoint.lstrip('/')}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={
        "Authorization": f"Token {_api_key(ctx)}",
        "User-Agent": USER_AGENT,
    })
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = " ".join(exc.read(500).decode("utf-8", "replace").split())
        raise ValueError(f"Freesound API returned HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise ValueError(f"Freesound API request failed: {exc.reason}") from None


def _license(url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(url.strip())
    key = (parsed.netloc.lower().removeprefix("www."), parsed.path.rstrip("/").lower())
    if key not in LICENSES:
        raise ValueError(f"Unsupported Freesound license: {url or '(missing)'}")
    return LICENSES[key]


def _clean(value) -> str:
    return " ".join(str(value or "").replace("|", "/").split())


def _get_sound(ctx: Context, sound_id: int) -> dict:
    if sound_id <= 0:
        raise ValueError("sound_id must be positive.")
    item = _request(ctx, f"sounds/{sound_id}/", {"fields": FIELDS})
    if int(item.get("id", 0)) != sound_id:
        raise ValueError("Freesound returned mismatched sound metadata.")
    license_name, license_url = _license(str(item.get("license", "")))
    preview_url = (item.get("previews") or {}).get("preview-hq-ogg")
    parsed = urllib.parse.urlparse(preview_url or "")
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not (host == "freesound.org" or host.endswith(".freesound.org")):
        raise ValueError(f"Freesound sound {sound_id} has no valid HQ OGG preview.")
    return {
        "sound_id": sound_id, "title": _clean(item.get("name")),
        "creator": _clean(item.get("username")), "license_name": license_name,
        "license_url": license_url, "source_url": f"https://freesound.org/s/{sound_id}/",
        "preview_url": preview_url,
    }


def freesound_search(ctx: Context, query: str, max_results: int = 10, page: int = 1,
                     sort: str = "score", min_duration: Optional[float] = None,
                     max_duration: Optional[float] = None) -> str:
    """Search Freesound for CC0/CC BY 4.0 sounds and return reusable candidate records."""
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty.")
    if not 1 <= max_results <= MAX_RESULTS:
        raise ValueError(f"max_results must be between 1 and {MAX_RESULTS}.")
    if page < 1 or sort not in SORTS:
        raise ValueError("Invalid page or sort.")
    if min_duration is not None and min_duration < 0:
        raise ValueError("min_duration cannot be negative.")
    if max_duration is not None and max_duration < 0:
        raise ValueError("max_duration cannot be negative.")
    if min_duration is not None and max_duration is not None and min_duration > max_duration:
        raise ValueError("min_duration cannot exceed max_duration.")
    filters = ['license:("Creative Commons 0" OR "Attribution")']
    if min_duration is not None or max_duration is not None:
        low = 0 if min_duration is None else min_duration
        high = "*" if max_duration is None else max_duration
        filters.append(f"duration:[{low} TO {high}]")
    data = _request(ctx, "search/text/", {
        "query": query, "filter": " ".join(filters), "sort": sort, "page": page,
        "page_size": min(150, max_results * 3), "fields": FIELDS,
    })
    records = []
    for item in data.get("results", []):
        try:
            license_name, license_url = _license(str(item.get("license", "")))
        except ValueError:
            continue
        preview = (item.get("previews") or {}).get("preview-hq-ogg")
        if not preview:
            continue
        sound_id = int(item["id"])
        records.append(
            f"ID {sound_id} | {_clean(item.get('name'))} | by {_clean(item.get('username'))} | "
            f"{float(item.get('duration') or 0):.2f}s {_clean(item.get('type')).upper()} | "
            f"rating {float(item.get('avg_rating') or 0):.2f}/{int(item.get('num_ratings') or 0)} | "
            f"{license_name} {license_url} | page https://freesound.org/s/{sound_id}/ | preview {preview}"
        )
        if len(records) == max_results:
            break
    return "\n".join(records) or "No CC0 or CC BY 4.0 results with HQ OGG previews."


def _segment(value: str, label: str) -> str:
    value = value.strip()
    if value.lower().endswith(".ogg"):
        value = value[:-4]
    if not value or ".." in value or "/" in value or "\\" in value:
        raise ValueError(f"Invalid {label}; path traversal and separators are not allowed.")
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_-")
    if not value:
        raise ValueError(f"Invalid {label}.")
    return value[:80]


def _relative(ctx: Context, path: Path) -> str:
    return os.path.relpath(path, ctx.resolve(".")).replace("\\", "/")


def _staged(ctx: Context, path: str) -> Path:
    result = Path(ctx.resolve(path)).resolve()
    root = Path(ctx.resolve(".workspace/freesound")).resolve()
    if os.path.commonpath((result, root)) != str(root) or result.suffix.lower() != ".ogg":
        raise ValueError("staged_path must be an OGG file under .workspace/freesound/.")
    return result


def _download(url: str, destination: Path) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    temporary = destination.with_suffix(".ogg.part")
    digest, size = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(temporary, "wb") as output:
            for chunk in iter(lambda: response.read(64 * 1024), b""):
                size += len(chunk)
                if size > MAX_PREVIEW_BYTES:
                    raise ValueError("Preview exceeds 20 MB audio-input limit.")
                digest.update(chunk)
                output.write(chunk)
        os.replace(temporary, destination)
        return digest.hexdigest()
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        reason = getattr(exc, "reason", f"HTTP {getattr(exc, 'code', 'error')}")
        raise ValueError(f"Freesound preview download failed: {reason}") from None
    finally:
        if temporary.exists():
            temporary.unlink()


def freesound_download_preview(ctx: Context, sound_id: int, output_name: str = "",
                               overwrite: bool = False) -> str:
    """Download a verified HQ OGG preview into .workspace/freesound/ for evaluation."""
    metadata = _get_sound(ctx, sound_id)
    name = _segment(output_name or metadata["title"] or f"sound_{sound_id}", "output_name")
    destination = Path(ctx.resolve(f".workspace/freesound/{sound_id}-{name}.ogg"))
    manifest_path = destination.with_suffix(".json")
    if (destination.exists() or manifest_path.exists()) and not overwrite:
        raise ValueError(f"Staged preview already exists: {_relative(ctx, destination)}")
    args = {"sound_id": sound_id, "output_name": output_name, "overwrite": overwrite}
    denial = _approve(ctx, "freesound_download_preview", args,
                      f"Download Freesound #{sound_id} preview to {_relative(ctx, destination)}")
    if denial:
        raise ValueError(f"User denied preview download: {denial}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata.update({"preview_kind": "HQ OGG preview", "sha256": _download(metadata["preview_url"], destination)})
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return (f"Staged: {_relative(ctx, destination)}\nFreesound #{sound_id}: {metadata['title']} by {metadata['creator']}\n"
            f"License: {metadata['license_name']} {metadata['license_url']}\nSource: {metadata['source_url']}\n"
            "Provenance: HQ OGG preview")


def _manifest(audio: Path) -> dict:
    try:
        data = json.loads(audio.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid or missing staged preview manifest: {exc}") from None
    if data.get("preview_kind") != "HQ OGG preview" or data.get("sha256") != hashlib.sha256(audio.read_bytes()).hexdigest():
        raise ValueError("Staged preview does not match its manifest.")
    return data


def _credit(metadata: dict, paths: list[str]) -> str:
    assets = "".join(f"Asset: {path}\n" for path in sorted(set(paths)))
    return (f"[Freesound #{metadata['sound_id']}]\nTitle: {metadata['title']}\nCreator: {metadata['creator']}\n"
            f"Source: {metadata['source_url']}\nLicense: {metadata['license_name']} - {metadata['license_url']}\n"
            f"{assets}Source file: HQ OGG preview\n")


def _upsert_credit(content: str, metadata: dict, asset_path: str) -> str:
    marker = f"[Freesound #{metadata['sound_id']}]"
    start = content.find(marker)
    paths = [asset_path]
    if start >= 0:
        end = content.find("\n[Freesound #", start + len(marker))
        end = len(content) if end < 0 else end + 1
        paths += re.findall(r"(?m)^Asset: (.+)$", content[start:end])
        return content[:start] + _credit(metadata, paths) + content[end:]
    if "FREESOUND ASSETS\n" not in content:
        content = content.rstrip() + "\n\n\nFREESOUND ASSETS\nHQ OGG previews downloaded via Freesound API.\n"
    return content.rstrip() + "\n\n" + _credit(metadata, paths)


def _atomic_write(path: Path, content: str) -> None:
    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as output:
            output.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def freesound_install(ctx: Context, sound_id: int, staged_path: str, category: str,
                      name: str, overwrite: bool = False) -> str:
    """Install a verified staged preview as <sfx_dir>/<category>/<name>.ogg and deduplicate its credits entry."""
    audio = _staged(ctx, staged_path)
    if not audio.is_file():
        raise ValueError(f"Staged preview does not exist: {staged_path}")
    manifest, metadata = _manifest(audio), _get_sound(ctx, sound_id)
    if manifest.get("sound_id") != sound_id or manifest.get("preview_url") != metadata["preview_url"]:
        raise ValueError("sound_id or current Freesound metadata does not match staged preview.")
    sfx_dir = ctx.data.get("sound:sfx_dir", "assets/sfx")
    destination = Path(ctx.resolve(f"{sfx_dir}/{_segment(category, 'category')}/{_segment(name, 'name')}.ogg"))
    asset_path = _relative(ctx, destination)
    credits_path = Path(ctx.resolve(ctx.data.get("sound:credits_file", "CREDITS.txt")))
    with INSTALL_LOCK:
        if destination.exists() and not overwrite:
            raise ValueError(f"Asset already exists: {asset_path}")
        credits = credits_path.read_text(encoding="utf-8") if credits_path.exists() else ""
        updated = _upsert_credit(credits, metadata, asset_path)
        args = {"sound_id": sound_id, "staged_path": staged_path, "category": category,
                "name": name, "overwrite": overwrite}
        denial = _approve(ctx, "freesound_install", args, f"Install {asset_path} and update {_relative(ctx, credits_path)}")
        if denial:
            raise ValueError(f"User denied sound install: {denial}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".ogg.tmp")
        try:
            shutil.copyfile(audio, temporary)
            os.replace(temporary, destination)
            _atomic_write(credits_path, updated)
        finally:
            if temporary.exists():
                temporary.unlink()
    return (f"Installed: {asset_path}\nCredit: Freesound #{sound_id}, {metadata['title']} by {metadata['creator']}, "
            f"{metadata['license_name']} ({metadata['license_url']}), HQ OGG preview")


def freesound_discard_preview(ctx: Context, staged_path: str) -> str:
    """Delete a rejected staged Freesound preview and its manifest."""
    audio = _staged(ctx, staged_path)
    metadata = _manifest(audio)
    denial = _approve(ctx, "freesound_discard_preview", {"staged_path": staged_path},
                      f"Delete rejected preview {staged_path}")
    if denial:
        raise ValueError(f"User denied staged preview deletion: {denial}")
    audio.unlink()
    audio.with_suffix(".json").unlink()
    return f"Deleted rejected Freesound #{metadata['sound_id']} preview: {staged_path}"


SOUND_PROMPT_TEMPLATE = """
You are an agent that sources sound effects from freesound.

You are working on this project:
<project>
{project}
</project>

Installed sounds go to `{sfx_dir}/<category>/<name>.ogg`; credits are kept in `{credits_file}`.

Before you start:
- Search for existing sounds in the project.
- Understand where they are used and why; how frequently they are called, etc.
- Then: listen to these sounds to calibrate yourself.

<agent_strategy>
Firstly:
- clarify desired feel, context, and target category/name.
- Work out what keywords are best to search for; (e.g. smash, crack, sword swing, slash, click, buzz, warning, success, ding, etc etc).
- Then, Search several keywords; (only CC0 1.0 and CC BY 4.0 are allowed.)
- (Sorting by most-downloads is generally quite a good idea.)

Secondly:
- Once you have downloaded a few sounds, you should create a small candidate set.
You MUST call read_audio on every candidate before accepting it: metadata, names, and tags are not audio evaluation.
- Evaluate transient, body, tail, noise, clipping, repetitiveness, mix fit, and feel; compare nearby SFX when useful.
Install the best sounds, and delete rejected staged files.

In total, you should sample AT LEAST 10 sounds.
</agent_strategy>

<sound_judgement>
- Short, punchy sounds are better than long sounds
- Ensure the sound isn't too overbearing and repetitive
- Ensure the sound is clear and smooth. No echoes, no static; quality is important.
- Most importantly: Ensure that the sound matches what is happening.
</sound_judgement>

After you have decided, report what the winning sounds were.
If there were multiple good ones, list all of them, rank all of them, and explain what they "feel like".
"""


def _sound_prompt(ctx: Context) -> str:
    return SOUND_PROMPT_TEMPLATE.format(
        project=ctx.data.get("sound:project", "(not described; inspect the codebase)"),
        sfx_dir=ctx.data.get("sound:sfx_dir", "assets/sfx"),
        credits_file=ctx.data.get("sound:credits_file", "CREDITS.txt"),
    )


SOUND_TOOLS = [
    freesound_search, freesound_download_preview, read_audio, freesound_install,
    freesound_discard_preview, glob, read_file, ask_user_question, git_working_tree,
]
SOUND_PROMPT = Message(role="system", overview="sound-agent", content=_sound_prompt).with_tools(SOUND_TOOLS)


def make_sound_agent(app: ex6.App, project: str, sfx_dir: str = "assets/sfx",
                     credits_file: str = "CREDITS.txt", **context_kwargs) -> Context:
    """Build a Freesound sourcing agent. Needs FREESOUND_API_KEY in project .env and an audio-capable model.
    context_kwargs override Context fields (name, model, reasoning, invoke_llm, messages, ...)."""
    kwargs = dict(name="sound_agent", model=M.GEMINI_LATEST.id, reasoning="high",
                  messages=[SOUND_PROMPT, ENV_PROMPT])
    ctx = Context(app, **{**kwargs, **context_kwargs})
    ctx.data["sound:project"] = project
    ctx.data["sound:sfx_dir"] = sfx_dir
    ctx.data["sound:credits_file"] = credits_file
    return ctx
