---
name: ssot
description: remove fragile state, single-source-of-truth
---
Evaluate the statefulness of the system/code you just worked on.
(Bad state is one of the most common causes of bugs, and we want to avoid it.)
Some guidelines, in order:
- If it's possible to remove the state entirely via smarter code: REMOVE IT.
- Otherwise, if the state can't be removed, try make it a single-source-of-truth (SSOT).
- Otherwise, if state must be duplicated, then make sure the state is either short-lived or recomputed frequently.
- Lastly, if the bad state can't be short-lived or recomputed, think about a way to invalidate it, or make the consumers aware of the duplication.
Don't forget the bigger picture.
If the system/code is clean and minimal; that's fine, no changes needed.

(Rule of thumb: If a goblin comes in and changes a bunch of memory/state randomly;
the program should just change behaviour; nothing should ever become broken.)
