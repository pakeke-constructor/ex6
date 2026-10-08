# Clean pasted-image input

## Motivation

Make image paste explicit and predictable without interfering with terminal-native text paste. Terminals already own text clipboard shortcuts and deliver text through `BRACKETED_PASTE`; ex6 should not overload `Ctrl+V` with platform-dependent behavior.

Keep InputBox responsible for editing/paste behavior, while avoiding Context reaching into InputBox internals to assemble a submitted message.

## Desired behavior

- Normal text paste remains untouched:
  - terminal handles `Ctrl+V`, `Cmd+V`, or `Ctrl+Shift+V`
  - ex6 receives `BRACKETED_PASTE` or ordinary characters
  - `InputPass.consume_text()` inserts text as today
- `Alt+V` is the explicit image-paste shortcut:
  - query system clipboard for image
  - store normalized PNG in existing process-scoped attachment directory
  - insert `[pasted-image 0xabc12]` token at cursor
  - no image / unsupported clipboard access is a no-op
- Remove `KEY_CTRL_V` image handling and alias. Do not make assumptions about whether a terminal intercepts or forwards Ctrl+V.
- Keep all paste handling inside InputBox, not work-mode renderer.
- No provider integration in this task.

## Submission representation

Add a small immutable input value, e.g.:

```python
@dataclass(frozen=True)
class InputValue:
    text: str
    attachments: dict[str, Attachment] = field(default_factory=dict)
```

- InputBox calls `on_submit(InputValue(...))`.
- InputBox includes only attachments whose tokens still occur in text.
- Context receives complete submitted input and does not inspect InputBox state.
- `Context.invoke` accepts the input value, or `submit_input` unwraps it into `Message(content=value.text, attachments=value.attachments)`. Prefer the smaller change after checking callers.
- Selection-mode input callback reads `value.text`; it naturally receives no attachments because image paste is disabled there or ignored.
- Message continues to own attachment references after submission.

## Relevant file

- `ex6.py`
  - `InputPass.KEY_ALIASES`
  - `InputBox`
  - `make_input`
  - `Context.get_input_box`, submission/invoke path
  - `TUI.sel_on_submit`
  - clipboard image helper and `Message.attachments`

Provider files remain unchanged.

## Implementation

1. Add `InputValue` near attachment/input data types.
2. Change InputBox submission to construct one `InputValue` from current text and referenced draft attachments, then clear draft state.
3. Update Context submission callback to consume `InputValue` directly. Remove Context lookup of `get_input_box().get_attachments()`.
4. Update generic/selection input callback to consume `InputValue.text` without adding image behavior to selection mode.
5. Remove `KEY_CTRL_V` alias and only trigger clipboard-image lookup on `KEY_ALT_V` inside InputBox.
6. Decide image-paste capability explicitly per InputBox, using the smallest API:
   - likely `InputBox(on_submit, allow_images=False)`
   - context input enables it; selection input keeps default disabled
   - avoid renderer-specific checks or callbacks
7. Retain current content-addressed temporary PNG storage and token collision handling.
8. Update `.plans/pasted-images.md` if needed so it no longer describes Ctrl+V image behavior or Context reading InputBox internals.

## Verification

Use a small temporary smoke test, then delete it:

- `BRACKETED_PASTE` inserts text and never queries image clipboard.
- literal Ctrl+V is not consumed as image paste.
- Alt+V inserts image token only in image-enabled context input.
- selection input does not query clipboard on Alt+V.
- deleting token excludes attachment from submitted `InputValue`.
- repeated identical image pastes get distinct tokens.
- submitted user Message retains attachments after InputBox clears.
- commands still dispatch from context and selection input.
- compile, existing tests, `git diff --check`, and final working-tree review.
