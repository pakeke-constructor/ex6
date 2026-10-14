# Panel-first scroll input

## Motivation

UP/DOWN conversation scrolling currently runs inside `render_work_mode`, before foreground UI panels are drawn. This consumes navigation keys before ask-question and other panels can handle them. Preserve `InputPass`'s consume-first behavior by giving the frontmost panel first refusal, then using any remaining scroll keys for conversation history.

## Relevant files

- `ex6.py`: work-view scrolling and render/input order
- `_ex6/tools.py`: ask-question panel behavior to verify; no change expected
- `_ex6/commands.py`, `_ex6/themes.py`: existing scrollable panels to verify; no change expected

## Plan

1. Remove conversation scroll-key consumption from `render_work_mode`; keep scroll offset calculation, clamping, and rendering there.
2. In `_tui_loop`, after the active global/context panel has drawn and consumed its input, handle remaining work-mode conversation scroll keys before flushing the buffer.
   - Defer UP/DOWN and PGUP/PGDOWN together so all conversation navigation follows the same priority rule.
   - Use the work region height for page-scroll distance.
   - Unconsumed panel keys intentionally fall through to conversation scrolling.
3. Accept the one-frame visual update from deferred handling; the next render loop applies the new offset without an extra redraw or input-routing abstraction.
4. Verify manually:
   - With no panel open, UP/DOWN and PGUP/PGDOWN still scroll conversation history.
   - Ask-question UP/DOWN changes selected option and does not move conversation history.
   - Existing scrollable text/theme panels receive UP/DOWN before conversation history.
   - Closing a panel restores normal conversation scrolling.
5. Run syntax/tests available in the project and inspect the final diff.
