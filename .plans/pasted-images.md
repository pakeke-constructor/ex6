# Pasted images

## Motivation and scope

Allow images from the system clipboard to be pasted into ex6 without turning the input box into a rich-text editor. Keep all context explicit: the input shows exactly where each image sits, while image bytes live outside the text buffer and are attached structurally when submitted.

Provider integration is explicitly out of scope for now.

## Representation

- InputBox remains text plus a cursor.
- A pasted image is inserted as a short unique token such as `[pasted-image 0x99843]` at the cursor.
- InputBox owns a token -> `ImageAttachment` map for the current draft.
- Normal input callback remains text-only; `Context.submit_input` reads current draft attachments from its input box. Tokens remain in submitted text, so conversation rendering and providers without image support still expose image position explicitly.
- Backspace/delete/cut need no attachment-specific logic: attachments whose tokens no longer occur are discarded when submitting or replacing text.
- Repeated pastes get unique tokens, even when image bytes are identical.

## Storage lifecycle

- Copy clipboard image immediately into ex6's existing process-scoped attachment temp directory via `store_attachment`; do not retain a mutable clipboard/Pillow object.
- Encode clipboard bitmaps and image files as PNG for one predictable stored format.
- `ImageAttachment` stores temp path, MIME type, dimensions, and detail.
- Extend `Message` with attachments so submitted user images belong to conversation data rather than InputBox state.
- Temp files intentionally live for process lifetime, matching existing tool attachments. Clear/fork only drop references; no risky per-message deletion or shared-file refcounting.
- Context dump stays text-only for now; it already cannot persist tool attachment files. Provider wiring and durable image serialization remain separate tasks.

## Input behavior

- Keep paste handling inside InputBox. Bracketed terminal paste continues to insert text; Alt+V queries the clipboard for an image because terminals cannot send image bytes through bracketed paste.
- Query clipboard through Pillow `ImageGrab.grabclipboard()` (Pillow is already a documented dependency).
- Accept a clipboard bitmap or the first clipboard file which Pillow can identify as an image.
- If clipboard has no image or clipboard image access is unsupported, leave draft unchanged.

## Relevant files

- `ex6.py`: Message attachments, Context invoke/submit path, clipboard capture, InputBox token storage/render/edit lifecycle.
- `_ex6/provider.py`, `_ex6/provider_openai.py`, `_ex6/provider_anthropic.py`: intentionally unchanged until provider task.

## Implementation

1. Add `attachments` to `Message` and allow `Context.invoke(text, attachments=())`.
2. Let `InputBox` own draft attachments and all paste handling. On Alt+V, store a clipboard image and insert a collision-free token at cursor.
3. Let Context read referenced attachments on text submission, then clear draft attachment map. Keep generic/selection input callback behavior unchanged.
4. Ensure `set_text` clears stale draft attachments.
5. Make debug/context/token helpers continue treating message content as text; optionally annotate attachment count in debug dump, without embedding image bytes.
6. Add focused tests for token insertion, ordering/de-duplication, deletion before submit, draft reset, clipboard bitmap/file capture, and message ownership. Mock clipboard reads; do not require real desktop clipboard.
7. Run tests/compile check and inspect diff/status.
