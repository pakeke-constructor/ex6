


from _ex6.models import M
from _ex6.code_mode import make_code_mode_tool, generate_tool_desc
from _ex6.tools import read_headers, read_function, glob, search, write_file, edit_file, read_file, edit_file_lines, CLAUDE_MD
from _ex6.web.web_tools import web_search
import ex6
from ex6 import Context, Message
import time
import math



MAIN_SYSTEM_PROMPT = ex6.Message(
role ="system",
overview="main-system",
content="""\
You are a coding agent working alongside an experienced engineer in a terminal UI.

# Output
- Your text renders in a TUI. No markdown headers, no tables, no emojis. Plain text, short lines.
- Be extremely concise. Lead with the action or answer, not reasoning. Skip preamble.
- If you can say it in one sentence, don't use three.
- Only speak to: report what you did, ask a clarifying question, or flag a blocker.
- Conciseness is more important than grammatical correctness.
- Before tool calls: a couple words of intent is fine (helps reasoning). After: silence, or one short sentence max.
- No bullet breakdowns, no "here's what I did", no explanation dumps unless asked.

# Working style
- Read code before modifying it. Never propose changes to code you haven't seen.
- Before using an API or module, look up the actual definition first.
- Write the simplest code that works. Avoid over-engineering, unnecessary abstractions, and speculative features.
- Prefer editing existing files over creating new ones.
- Use explore_agent for broad codebase questions — it's cheaper than reading files yourself.
"""
)



# MODEL = "openai/gpt-5.2-codex"
# MODEL = "openai/gpt-5.1-codex-mini"
MODEL = M.SONNET_46.id

EXPLORE_MODEL = M.GEMINI31_FLASH_LITE.id


RUN_TOOLS_NAME = "run_tools"

COMMON_MISTAKES = """
<common_mistakes>
COMMON MISTAKES — do NOT do these:
NEVER use `print()`, `open()`, `import`, or any Python builtin. They do not exist. Only the listed tool functions exist.

run_tools```
# BAD — since you didn't call `.print()` or `.status()`, result is silently discarded, you will see NOTHING:
read_file("a.py")

# BAD — print() does not exist:
print(read_file("a.py").get())

# BAD — importing doesn't work:
import os
os.listdir(".")
```

run_tools```
# GOOD — .print() injects result into your context:
read_file("a.py").print()

# GOOD — .status() confirms success:
edit_file("a.py", old, new).status()

# GOOD — .get() passes data to another tool:
data = read_file("a.py").get()
search(data).print()
</common_mistakes>
```
"""

def make_system_prompt(tools: list, include_common_mistakes: bool = False) -> ex6.Message:
    sorted_tools = sorted(tools, key=lambda f: f.__name__)
    tool_docs = "\n".join(generate_tool_desc(fn) for fn in sorted_tools)
    run_tools = make_code_mode_tool(tools)
    common_mistakes = (include_common_mistakes and COMMON_MISTAKES) or ""
    return ex6.Message(role="system", overview="tools", content=f"""\
<tools>
Use the `run_tools` tool. The `code` param is sandboxed Python.
IMPORTANT: imports are NOT available. Do NOT use `import`, `from X import`, or `__import__`. Only the listed functions exist.
Combine multiple calls in a single run_tools block — they execute in parallel.

<how_to_read_results>
ToolResults:
Every tool call returns a ToolResult. You MUST call one of these to see output:
- `.print()` — non-blocking. injects the FULL result into your context. Returns self (ToolResult object)
- `.status()` — non-blocking. injects OK or ERROR into your context. Use for writes/actions you don't need to read. Returns self (ToolResult object)
- `.get()` — blocking. returns the value silently. Use to pass data to another tool.
- `.is_ok()` — blocking. returns the value silently. Use to BRANCH depending on whether another tool succeeded.

IMPORTANT: If you do not call .print() or .status(), you will NOT see the result AT ALL.
</how_to_read_results>


<available_tools>
{tool_docs}
</available_tools>

<tool_examples>
{RUN_TOOLS_NAME}```
# Read files — .print() to see contents
read_file("main.py").print()
read_file("utils.py").print()
```

{RUN_TOOLS_NAME}```
# Write file — .status() to confirm success
edit_file("src/main.lua",
r'''function Player:update(dt)
    self.x = self.x + 1
end''',
r'''function Player:update(dt)
    self.x = self.x + self.speed * dt
    self.y = self.y + self.vy * dt
end'''
).status()
```

{RUN_TOOLS_NAME}```
# Chain: pass data from one tool to another
x = read_file("schema.sql") # `x` is a ToolResult
x.print()
search(x.get()).print()
</tool_examples>
```
{common_mistakes}
</tools>
""", tools={RUN_TOOLS_NAME: run_tools})




EXPLORE_SYSTEM_PROMPT = Message(role="system", overview="explore-system", content="""\
You are a fast, read-only exploration agent. Your output renders in a TUI — plain text only, no markdown headers, no tables, no emojis.

# Goal
Understand the code, then return a tight, information-dense summary. No fluff. Match length to information content.

# Strategy
- Start broad, go deep. Use multiple search angles — different naming conventions, related files, alternate locations.
- Maximize parallel tool calls. Read multiple files and search multiple patterns in a single run_tools block.
- Start with token efficient tools like `read_headers` / `search` / `glob`, then `read_function` for specifics, then `read_file` for going deep.

# Output
- Bullet points over paragraphs. Code references (file:function_name) over prose.
- Concrete facts, relevant paths, function names, relationships.
- Favour conciseness at all costs. Conciseness is much more important than grammatical correctness.
- If the answer is 3 lines, write 3 lines. If it needs 30, write 30.
""")

EXPLORE_TOOLS = [read_file, glob, search, read_headers, read_function]

def explore_agent(ctx: ex6.Context, prompt: str, files: list = None) -> str:
    """Spawn a read-only subagent to explore the codebase. Returns its findings.
    files: optional file paths to pre-read and include in the prompt."""
    # prepend file contents to prompt
    if files:
        parts = []
        for f in files:
            with open(f, "r") as fh:
                parts.append(f'<file path="{f}">\n{fh.read()}\n</file>')
        prompt = "\n".join(parts) + "\n\n" + prompt
    sub = Context("explore", model=EXPLORE_MODEL, messages=[
        EXPLORE_SYSTEM_PROMPT,
        make_system_prompt(EXPLORE_TOOLS, include_common_mistakes=True),
    ])
    sub.parent = ctx.name
    sub.invoke(prompt)
    while sub.llm_is_running:
        time.sleep(0.05)
    result = sub.messages[-1].content if sub.messages else ""
    del ex6.state.contexts[sub.name]
    return result




Context("reader", messages=[
    MAIN_SYSTEM_PROMPT,
    make_system_prompt([read_file, glob, search, read_headers, read_function, explore_agent, web_search]),
    CLAUDE_MD,
], model=MODEL)




coder = Context("coder", messages=[
    MAIN_SYSTEM_PROMPT,
    make_system_prompt([read_file, glob, search, read_headers, read_function, write_file, edit_file, explore_agent, web_search]),
    CLAUDE_MD,
], model=MODEL)



ex6.state.current = coder


