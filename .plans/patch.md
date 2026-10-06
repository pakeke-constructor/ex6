# Specialized patch_file

Motivation: make it easier for codex-style agents to edit files without duplicating paths, multi-file envelopes, or exposing extra file operations. Keep ex6's tool/context surface explicit and minimal.

Research:
- Official Codex grammar: https://github.com/openai/codex/blob/main/codex-rs/apply-patch/src/parser.rs
- Matching: https://github.com/openai/codex/blob/main/codex-rs/apply-patch/src/seek_sequence.rs
- Full format wraps operations in `*** Begin Patch` / `*** End Patch`; updates use `*** Update File: path`, optional `@@` / `@@ anchor`, space-prefixed context, `-` removals, `+` additions, and optional `*** End of File`.
- `@@` is not a unified-diff line-number header. First hunk may omit it; further hunks use separators. Anchors search forward.

Implementation:
1. Add `patch_file(ctx, file, patch)` in `_ex6/tools.py`, accepting only update-body syntax (no envelopes, paths, create/delete/move operations). Retain `write_file`; do not add delete-file.
2. Support ordered hunks, optional/consecutive anchors, exact then trailing-whitespace then stripped matching. Preserve original context lines. Intentionally omit Codex's Unicode-punctuation fuzzing. EOF hunks must match the tail; addition-only hunks append, like Codex.
3. Preserve existing newline style and final-newline state. Validate complete patch before approval/write; reuse read-before-edit, per-file lock, diff preview, denial, and read tracking.
4. Replace `edit_file` with `patch_file` in `_ex6/agents.py` MAIN_TOOLS. Leave existing edit helpers available for other plugins; retain `write_file` unchanged.
5. Add focused standard-library tests outside `_ex6` (not an auto-loaded plugin). Cover syntax, matching, anchors, EOF, additions/deletions, newline preservation, malformed patches, failed later hunks, read guard, approval/denial, and tool registration.
6. Run tests, compile changed files, inspect git diff. Leave existing user notes untouched.

Files: `_ex6/tools.py`, `_ex6/agents.py`, `tests/test_patch_file.py`.
