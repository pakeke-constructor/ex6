
Implement a barebones epic management system in Python. Here are the full specifications:

## Overview
A single-file CLI tool that manages "epics" — structured XML files that track tasks,
learnings, and archived content. Agents append to these files; humans approve or deny.

---

## File Schema
Every epic is stored as an XML file with this exact structure:

<epic>

<objective confidence="1.0" approved-by="human" approved-at="ISO8601">
# Title
Human-authored description.

The user wants to achieve X, ideally under Y constraint.
This allows system Foo too finally do "XYZ".
</objective>


<human_notes>
Do it this way.
dont use xyz, use foobar instead.
<!-- AGENTS SHOULD NEVER EVER EDIT THIS. THIS IS FOR HIGH-IMPORTANTCE, HUMAN NOTES. -->
</human_notes>


<tasks>
  <task status="pending" confidence="0.8" time="ISO8601">
    Description.
  </task>
  <task status="pending" confidence="0.6" time="ISO8601">
    Description. Add commandbuffer. Blah blah description
  </task>
</tasks>

<learnings>
  [ISO8601] Openrouter doesn't pass anthropic request params. use `extraParams` param instead.
  [ISO8601] Prompt caching for gemini needs warmup. keep in mind for testing.
  [ISO8601] Prompt caching for gemini needs warmup. keep in mind for testing.
</learnings>

<archive>
  <!-- Denied/superseded tasks land here. -->
</archive>

</epic>

---

## Confidence Rules

- objective block is always 1.0 confidence.
- When an agent creates a block, it inherits confidence minus 0.3.
- Human-created tasks are always 1.0
- Denied tasks are set to 0.0
- Tasks with confidence < 0.6 must not be executed.

---

## Human notes
This is where the human can put critical information.
the agent should pay deep attention to this section, as it contains instructions directly from the human.

Workflow is that the human should directly edit this file when they want,
essentially directly interfacing with LLM.


---

## Task Status Rules

- Valid: pending, in_progress, completed, denied
- Only one task in_progress at a time
- Mark in_progress BEFORE starting work
- Mark completed IMMEDIATELY after finishing
- On denial: set status=denied, confidence=0.0, add <denial-reason> child,
  echo reason to <learnings>, move task to <archive>

---

## Plan Denial

When the entire plan is denied:
1. Move ALL tasks to <archive> with a <denial-reason> block
2. Clear <tasks>
3. Append a learning entry
4. Halt

---

## Append-Only Rules

- Never modify <main>
- Never delete from <learnings> or <archive>
- Moving to <archive> = copy there + remove from <tasks>
- Every append includes ISO8601 timestamp + agent/human attribution

---

## tools to define in `tools.py`:

epic tasks add <epic-id> <description> [--human]
  Adds a task. --human flag sets confidence to 1.0, otherwise inherits from epic - 0.2.
  Prints the assigned task ID.

epic tasks list <epic-id>
  Lists all non-archived tasks with id, status, confidence, and description.

epic tasks complete <task-id>
  Sets a task to completed.

epic learn <epic-id> <message> [--human]
  Appends a learning entry with timestamp and attribution.

epic show <epic-id>
  Pretty-prints the current state: main, active tasks, recent learnings (last 5).

---

## Implementation Requirements

- Pure Python stdlib only — no dependencies
- Single file: epic.py
- XML parsing via xml.etree.ElementTree
- Task IDs are auto-incremented: EPICNAME-001, EPICNAME-002, etc.
- ISO8601 timestamps via datetime.utcnow().isoformat()
- All file writes are atomic: write to a .tmp file, then os.replace() into place
- Epic files are stored in ./epics/<epic-id>.xml
- On any confidence violation, print a clear warning and raise ValueError

---

## What NOT to build

- No subtask nesting
- No compaction
- No multi-agent coordination
- No network calls
- No database

