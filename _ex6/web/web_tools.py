
import os, sys, re, subprocess, json
import ex6
import time
from _ex6.models import M


_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV = os.path.join(_DIR, ".venv")

def _venv_python():
    if sys.platform == "win32":
        return os.path.join(_VENV, "Scripts", "python.exe")
    return os.path.join(_VENV, "bin", "python")

def _find_compatible_python():
    """Find a Python 3.11-3.13 interpreter. crawl4ai doesn't support 3.14+."""
    for minor in (13, 12, 11):
        if sys.platform == "win32":
            try:
                r = subprocess.run(["py", f"-3.{minor}", "-c", "import sys; print(sys.executable)"],
                                   capture_output=True, text=True)
                if r.returncode == 0:
                    return r.stdout.strip()
            except FileNotFoundError:
                pass
        else:
            for name in (f"python3.{minor}", f"python3"):
                try:
                    r = subprocess.run([name, "-c", f"import sys; v=sys.version_info; assert v.minor=={minor}; print(sys.executable)"],
                                       capture_output=True, text=True)
                    if r.returncode == 0:
                        return r.stdout.strip()
                except FileNotFoundError:
                    pass
    # fallback: current interpreter if it's 3.11-3.13
    if sys.version_info[:2] in ((3,11),(3,12),(3,13)):
        return sys.executable
    raise RuntimeError("crawl4ai requires Python 3.11-3.13. Please install one.")

def _ensure_venv():
    if os.path.isdir(_VENV):
        return
    py = _find_compatible_python()
    subprocess.run([py, "-m", "venv", _VENV], check=True)
    pip = os.path.join(_VENV, "Scripts", "pip.exe") if sys.platform == "win32" else os.path.join(_VENV, "bin", "pip")
    req = os.path.join(_DIR, "requirements.txt")
    subprocess.run([pip, "install", "-r", req], check=True)


def web_search(ctx: ex6.Context, query: str) -> str:
    """Search the web. Returns top results as text."""
    import urllib.request, urllib.parse, html as html_mod
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        page = r.read().decode()
    # split into per-result blocks to avoid cross-contamination
    blocks = re.split(r'class="links_main links_deep result__body"', page)
    out = []
    for block in blocks[1:9]:
        m_title = re.search(r'class="result__a"[^>]*>(.*?)</a>', block)
        m_snippet = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block)
        m_href = re.search(r'class="result__a"[^>]*href="(.*?)"', block)
        if not (m_title and m_snippet and m_href):
            continue
        title = re.sub(r'<[^>]+>', '', m_title.group(1)).strip()
        snippet = re.sub(r'<[^>]+>', '', m_snippet.group(1)).strip()
        snippet = html_mod.unescape(snippet)
        # extract real URL from DDG redirect
        raw_href = html_mod.unescape(m_href.group(1))
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_href).query)
        href = urllib.parse.unquote(parsed.get("uddg", [raw_href])[0])
        out.append(f"{title}\n  {href}\n  {snippet}")
    if not out:
        raise ValueError("No results found! Do NOT make up results; inform your user that there was likely a search error.")
    return "\n\n".join(out)




def web_scrape(ctx: ex6.Context, url: str, max_chars: int = 50_000) -> str:
    """
    Fetch and return the readable text content of a webpage.
    - Use this after web_search() to read the full content of a result
    - max_chars: truncate output to this many characters (default 50k)
    - Will fail gracefully on paywalled, bot-protected, or JS-only pages
    - Avoid scraping the same URL repeatedly in one session
    """
    _ensure_venv()
    worker = os.path.join(_DIR, "_worker.py")
    result = subprocess.run(
        [_venv_python(), worker, url, str(max_chars)],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        raise RuntimeError(f"web_scrape failed: {err}")
    data = json.loads(result.stdout)
    if not data["ok"]:
        raise RuntimeError(data["error"])
    return data["markdown"]





WEBSEARCH_MODEL = M.GEMINI31_FLASH_LITE.id

WEBSEARCH_SYSTEM_PROMPT = ex6.Message(role="system", content="""\
You are a focused web research agent. Answer the given question using web_search and web_scrape tools.
Strategy:
- Search for the question. Read 1-2 top results if needed for detail.
- Answer concisely — facts only, no padding. Conciseness is much more important than grammatical correctness.
- If you can answer from search snippets alone, do so without scraping.
- Plain text only, no markdown.
""")


WEBSEARCH_TOOLS_MSG = ex6.Message(role="system", overview="tools",
    content="Use web_search to find pages, and web_scrape to read a page in full.",
    tools={"web_search": web_search, "web_scrape": web_scrape})

def websearch_agent(ctx: ex6.Context, question: str) -> str:
    """Spawn a websearch subagent to research a question. Returns a concise answer.
    Use this when you need up-to-date information from the web."""
    sub = ex6.Context("websearch", model=WEBSEARCH_MODEL, messages=[WEBSEARCH_SYSTEM_PROMPT, WEBSEARCH_TOOLS_MSG])
    sub.parent = ctx.name
    sub.invoke(question)
    while sub.llm_is_running:
        time.sleep(0.05)
    result = sub.messages[-1].content if sub.messages else "No answer."
    del ex6.state.contexts[sub.name]
    return result

