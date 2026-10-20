# paste-images: pasted images never reach the model

## Motivation
User: "the image-pasting has been like this for a few weeks". Pasting an image into the work-mode input box
shows `[pasted-image 0x3cc8d]`, but the model only ever receives that text token. The image is silently dropped.
ex6's promise is "no hidden context, full control"; silently losing user input breaks that.

## Root cause
Input side works:
- `InputBox._insert_attachment` (ex6.py) stores the png in `%TEMP%/ex6-attachments-*/<sha256>.png`, inserts token.
- On Enter, `InputValue(text, {token: ImageAttachment})` -> `Context.invoke(..., attachments=)` ->
  `Message(role="user", content=text, attachments={token: att})`.

Provider side drops it. All three providers only encode attachments on `ToolResult` (tool messages),
never `Message.attachments` on user messages:
- `_ex6/provider_anthropic.py`: `send_user(text)` sends a plain string; `_bootstrap_prompt` flattens
  history into one string. Both ignore `m.attachments`.
- `_ex6/provider.py` (openrouter): `msg_to_dict` uses `m.get_msg(ctx)` only.
- `_ex6/provider_openai.py`: `_to_responses_input` user branch is `[{"type":"input_text","text": c}]` only.

## Fix
Per provider, when a user message has attachments, send a content array instead of a string:
text block with the full prompt, then for each `(token, att)`: text block `token` + image block.
The token label lets the model map "the image I pasted" to the right image. No interleaving/splitting text.

- provider_anthropic.py:
  - `send_user(message)` takes the Message; builds content
    `[{"type":"text",...}, {"type":"image","source":{"type":"base64","media_type":mime,"data":b64}}]`.
    (Anthropic API block shape, not the MCP `{"type":"image","data","mimeType"}` shape used in `_tool_result_payload`.)
  - `_bootstrap_prompt`: return content blocks, not str. Keep transcript text as-is, append image blocks
    (token label + image) for every user message with attachments, so images survive a process rebuild.
  - Verify stream-json `user` events accept a content array (quick manual test with `claude -p`).
- provider.py: in `msg_to_dict`, user msg with attachments -> `[{"type":"text"}, {"type":"image_url", ...}]`
  (reuse the data-url encoding from `_tool_content`).
- provider_openai.py: user branch appends `{"type":"input_image","image_url": data_url, "detail": att.detail}`
  (reuse encoding from `_responses_tool_output`).

Missing file (temp dir wiped on exit; contexts restored later) -> raise ValueError like the tool-result paths do.

## Verify
- Paste image in each provider's context, ask "what's in the image?"; answer must describe it.
- anthropic: also force a process rebuild (stop a turn mid-stream, then re-ask about the image).
- `/` commands with pasted image still fine; selection panel unaffected.

## Out of scope
- Persisting attachments across restarts (temp dir is deleted at exit).
- Audio paste.

## Extra User Instructions:
"Please make it as simple as possible, in as least amount of code as possible"
