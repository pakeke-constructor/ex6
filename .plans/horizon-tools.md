# Easier custom tools for horizon agents

## Motivation

Project plugins should be able to add a project-specific tool to horizon agents without rebuilding `default_agents`, mutating each returned `Context`, or appending artificial system messages. Current API makes a simple extension feel indirect and ugly.

Current friction in `_ex6/my_agents.py`:

```python
agents = h.default_agents(tui.app)
# No direct way to add my_tool here.
h.run_horizon(tui.app, tasks_path, agents)
```

The available workaround is to append a tool-bearing `Message` to one or more contexts after construction. That exposes message internals, adds prompt clutter, and requires repetitive agent selection.

## Desired outcome

Add a small first-class extension point in core ex6 for supplying extra tools when creating default horizon agents. It should:

- Keep existing calls unchanged.
- Allow adding a tool to one agent or a deliberate set of agents.
- Preserve each agent's existing tools and tool ordering.
- Validate tools through normal `Message.with_tools` behavior.
- Avoid post-construction context/message mutation in project plugins.

Target usage should be roughly as simple as:

```python
agents = h.default_agents(tui.app, main_tools=[my_tool])
h.run_horizon(tui.app, tasks_path, agents)
```

If tools commonly need to be shared, consider parallel optional arguments such as `review_tools` and `doc_tools`, or one clearly typed mapping keyed by agent role. Prefer smallest API that covers real usage; don't add a general agent-builder abstraction.

## Relevant files in main ex6

- `_ex6/agents_long_horizon.py`: `default_agents`; constructs main, review, and doc contexts and owns their tool lists.
- `ex6.py`: `Message.with_tools`, tool validation, and `Context.get_tools` semantics.
- `_ex6/_plugin_examples/_example_horizon.py`: update/add concise example if it demonstrates default-agent customization.

## Proposed execution (awaiting API agreement)

1. Agree on keyword-only `main_tools`, `review_tools`, and `doc_tools`, each defaulting to `None`. Share tools by passing the same list to selected roles; no mapping or builder abstraction.
2. Extend each existing system message with a second `.with_tools(...)` call. This preserves defaults and normal same-name replacement/order semantics without changing `ex6.py`.
3. Add a concise default-agent customization example, retaining the existing full-customization example.
4. Smoke-test unchanged defaults, role isolation, ordering, same-name replacement, and normal tool validation; inspect diff.

Pause for agreement before implementation.

## Project reproduction

- `_ex6/my_agents.py`: `my_tool` and `/horizon` command expose the need for this extension point.
