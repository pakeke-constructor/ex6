# First-class image tool results

## Motivation

Let coding agents inspect local screenshots without treating image bytes/base64 as text, leaking them into the TUI, or retaining large byte arrays in long-running contexts. Keep ex6 thin: core represents generic tool results and disk-backed attachments; tools create them; providers translate them into native multimodal content parts.

## Relevant files

- `ex6.py` — generic result/attachment types, tool-return normalization, safe display/persistence
- `_ex6/tools.py` — image loading/normalization and binary guidance
- `_ex6/provider.py` — OpenRouter Chat Completions serialization
- `_ex6/provider_openai.py` — OpenAI Responses serialization
- `_ex6/agents.py` — tool registration
- `_ex6/_test.py` or focused new tests

## Design

Add generic core types:

```python
@dataclass(frozen=True)
class Attachment:
    path: str
    mime_type: str

@dataclass(frozen=True)
class ImageAttachment(Attachment):
    width: int
    height: int
    detail: Literal["auto", "low", "high"] = "auto"

@dataclass
class ToolResult:
    text: str
    attachments: tuple[Attachment, ...] = ()
```

`Attachment.path` points to an ex6-managed normalized temporary file. Context history never owns image bytes or base64. A tool result may contain zero, one, or many attachments, matching provider content-part APIs and leaving room for future file/audio support without adding provider concepts to core.

Existing tools do not change their public return values. `call_tools()` normalizes at its boundary:

- `ToolResult` remains structured.
- `str` becomes `ToolResult(text=value)`.
- Other values become `ToolResult(text=str(value or ""))`.
- Exceptions become textual error results.
- Text limits apply only to `ToolResult.text`.

Only tool-role messages contain `ToolResult`; system/user/assistant content remains unchanged.

## Implementation

### 1. Core result plumbing

In `ex6.py`:

- Add `Attachment`, `ImageAttachment`, and `ToolResult`.
- Broaden tool-message content typing and callable return typing accordingly.
- Normalize every completed tool return in `call_tools()` while preserving call order and parallel execution.
- Add a small helper that returns visible text from `str | ToolResult`; use it in tool rows, context display, error detection, and repetition checks.
- Ensure repr/display/debug paths never include attachment payloads or generated data URLs.
- Keep repetition fingerprints based on tool name/arguments and compare only result text.
- Persist tool results as text only. Restored context history intentionally has no attachment because managed files are session-local.
- Runtime clones/forks may share immutable attachment references.

### 2. Disk-backed attachment lifecycle

- Create normalized image files under one process-scoped temporary directory using `tempfile`.
- Use deterministic/content-addressed filenames so repeated reads deduplicate files.
- Store only paths and metadata in messages.
- Providers read and base64-encode files immediately before a request; bytes/base64 remain local temporaries and are released after invocation.
- Delete process temporary directory on clean shutdown; OS temporary-directory cleanup remains fallback after crashes.
- Do not implement context-level reference counting initially. Process-scoped files are bounded by explicit source/transmitted size limits and disappear when ex6 exits.

### 3. `read_image` tool

Add to `_ex6/tools.py`:

```python
def read_image(
    ctx: ex6.Context,
    path: str,
    max_dimension: int = 3072,
    detail: str = "auto",
) -> ex6.ToolResult:
```

Use Pillow to:

- Require an existing regular file.
- Enforce named maximum source-byte and decoded-pixel limits.
- Validate `max_dimension` against a bounded range and `detail` against `auto | low | high`.
- Decode genuine PNG, JPEG, WebP, GIF, and optionally BMP; reject corrupt/unsupported files clearly.
- Apply EXIF orientation.
- Use first frame of animated images.
- Preserve aspect ratio, never upscale, and resize only above `max_dimension`.
- Normalize modes safely to RGB/RGBA.
- Preserve screenshot clarity: deterministic PNG for alpha, resized images, GIF/WebP/BMP, or screenshot-like inputs; retain deterministic JPEG where suitable.
- Strip unnecessary metadata.
- Write normalized bytes to managed temporary storage, then discard in-memory bytes.

Return concise text plus one `ImageAttachment`. Include source/transmitted dimensions and sizes in text, for example:

```text
Loaded image .agent/shots/ticket.png
1440x900 · image/png · 184 KB
```

or:

```text
Loaded image .agent/shots/full-page.png
1440x8120 → 545x3072 · image/png · 1.8 MB → 412 KB
```

Attachment contains transmitted dimensions, MIME type, path, and requested detail.

### 4. Text-file behavior and read policy

Update `read_file()`:

- Document that it reads text and images require `read_image`.
- Open explicitly as UTF-8 with strict errors.
- Detect supported image signatures/extensions before decoding and return `'<path>' is an image. Use read_image instead.`
- Convert remaining Unicode decode failures/NUL-heavy input into `'<path>' is a binary file and cannot be read as text.`

Keep glob/search gitignore behavior unchanged. For `read_image`, permit explicit reads of ignored generated files, including `.agent/shots/**`, but run a separate conservative sensitive-path check first. Protect `.env`, `.env.*`, SSH private keys, common credential files, and obvious secret/token filenames. Do not change write protections.

### 5. Provider serialization

OpenRouter Chat Completions (`_ex6/provider.py`):

- Plain `ToolResult` serializes exactly like the current textual tool output.
- Results with image attachments serialize tool `content` as an ordered array: one text part followed by one `image_url` part per attachment.
- Each image URL is a MIME-correct base64 data URL used only inside `image_url.url`.
- Build fresh content dictionaries. `_apply_cache_control()` must copy list/block structures before annotating a valid final block, never mutate stored results or attachments.
- Missing files or unsupported attachment types raise a concise actionable provider error rather than silently dropping content.

OpenAI Responses (`_ex6/provider_openai.py`):

- Plain results remain string `function_call_output.output` values.
- Attached results use `function_call_output.output` as a list containing `input_text` followed by `input_image` parts, preserving `call_id`, attachment order, and each image's `detail`.
- This matches current SDK schema: function output accepts `str | list[input_text | input_image | input_file]`; no adjacent synthetic user message is needed.

Do not log serialized request bodies or data URLs. Let provider/API errors report model-specific vision incompatibility rather than maintaining a hardcoded model-name list; wrap obvious unsupported-image errors with guidance to switch to a vision-capable model.

### 6. Registration and dependency

- Add `read_image` beside `read_file` in `_ex6/agents.py` main coding tools.
- Do not add it to read-only subagents unless their configured model is intentionally vision-capable.
- Keep its docstring concise: screenshots/local images, PNG/JPEG/WebP/GIF, while `read_file` is text-only.
- Add Pillow using the project's dependency-management convention. If none exists, document/install it in the minimal existing setup location rather than introducing a packaging system solely for this feature.

## Focused tests

Core/pipeline:

- String/scalar/error returns normalize to `ToolResult`.
- Rich results survive parallel tool execution in call order.
- Text limit applies to text, not attachment file size.
- TUI display, repetition guard, context dump, and persistence expose text only.
- No image bytes/base64 remain in message content.

Image tool:

- PNG, JPEG, RGBA PNG, first-frame GIF, and WebP when supported.
- EXIF orientation, proportional resize, and no upscale.
- Corrupt/unsupported/excessive images rejected.
- `read_file` gives image/binary guidance.
- Ignored `.agent/shots` image can be read; sensitive ignored files remain blocked.
- Repeated identical normalized images reuse a managed file.

Providers:

- OpenRouter emits text plus one or multiple ordered `image_url` parts with correct MIME and round-trippable base64.
- Responses emits text plus one or multiple ordered `input_image` items with call ID and detail preserved.
- Text-only tool messages remain unchanged.
- Cache-control does not mutate/corrupt multimodal blocks.
- Provider logs and visible output contain no data URLs.

## Validation

- Run focused tests, full available suite, and type/lint checks.
- Confirm `read_image(".agent/shots/ticket.png")` lets a vision-capable model describe visible UI.
- Confirm `read_file` directs image reads to `read_image` and still reads source text normally.
- Confirm `.env` remains inaccessible.
- Inspect git diff and debug/TUI output for leaked bytes or base64.
