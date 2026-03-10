


from _ex6 import provider
from _ex6.code_mode import make_code_mode_tool, generate_tool_desc
from _ex6.tools import read_headers, read_function, glob, search, write_file, edit_file, read_file, edit_file_lines
import ex6
from ex6 import Context, Message
import time
import math



CODER_SYSTEM_PROMPT = ex6.Message(
role ="system",
content="""
# Role and Goal:
You are an intelligent coding assistant, working alongside an experienced engineer.
You will be asked to assist with activities such as debugging code, refactoring functions, and implementing new solutions to the engineer's problems.

# Guidelines:
- Be as concise as possible.
- Avoid bloated "apologetic language" like "You are absolutely right!"
- If you are using an API or module, you *MUST* look for the actual function/class definition before you use it. Sometimes the user will provide a tool to search the docs.
"""
)



MODEL = "openai/gpt-5.1-codex-mini"


RUN_TOOLS_NAME = "run_tools"

COMMON_MISTAKES = '''
## COMMON MISTAKES — do NOT do these:
NEVER use `print()`, `open()`, `import`, or any Python builtin. They do not exist. Only the listed tool functions exist.

```
# BAD — result is silently discarded, you will see NOTHING:
read_file("a.py")

# BAD — print() does not exist:
print(read_file("a.py").get())

# BAD — import does not exist:
import os
os.listdir(".")
```

```
# GOOD — .print() injects result into your context:
read_file("a.py").print()

# GOOD — .status() confirms success:
edit_file("a.py", old, new).status()

# GOOD — .get() passes data to another tool:
data = read_file("a.py").get()
search(data).print()
```
'''

def make_system_prompt(tools: list, include_common_mistakes: bool = True) -> ex6.Message:
    sorted_tools = sorted(tools, key=lambda f: f.__name__)
    tool_docs = "\n".join(generate_tool_desc(fn) for fn in sorted_tools)
    run_tools = make_code_mode_tool(tools)
    common_mistakes = (include_common_mistakes and COMMON_MISTAKES) or ""
    return ex6.Message(role="system", content=f'''\
# Tools
Use the `run_tools` tool. The `code` param is sandboxed Python.
IMPORTANT: imports are NOT available. Do NOT use `import`, `from X import`, or `__import__`. Only the listed functions exist.
Combine multiple calls in a single run_tools block — they execute in parallel.

## ToolResult
Every tool call returns a ToolResult. You MUST call one of these to see output:
- `.print()` — non-blocking. injects the FULL result into your context. Returns self (ToolResult object)
- `.status()` — non-blocking. injects OK or ERROR into your context. Use for writes/actions you don't need to read. Returns self (ToolResult object)
- `.get()` — blocking. returns the value silently. Use to pass data to another tool.
- `.is_ok()` — blocking. returns the value silently. Use to BRANCH depending on whether another tool succeeded.

**IMPORTANT: If you do not call .print() or .status(), you will NOT see the result AT ALL.**

## Available tools
{tool_docs}

## Examples:
{RUN_TOOLS_NAME}```
# Read files — .print() to see contents
read_file("main.py").print()
read_file("utils.py").print()
```

{RUN_TOOLS_NAME}```
# Write file — .status() to confirm success
edit_file("src/main.lua",
"""function Player:update(dt)
    self.x = self.x + 1
end""",
"""function Player:update(dt)
    self.x = self.x + self.speed * dt
    self.y = self.y + self.vy * dt
end"""
).status()
```

{RUN_TOOLS_NAME}```
# Chain: pass data from one tool to another
x = read_file("schema.sql") # `x` is is a ToolResult
x.print()
search(x.get()).print()
```

{common_mistakes}
''', tools={RUN_TOOLS_NAME: run_tools})




EXPLORE_SYSTEM_PROMPT = Message(role="system", content="""\
# Role
You are an exploration agent. Your job is to deeply understand the structure and semantics of a system, codebase, or module, then report your findings as concisely as possible.

# Goal
Fully understand what the code DOES, HOW it's structured, and WHY it's built that way. Then compress your understanding into a tight, information-dense summary. No fluff, no filler. just the essential facts the caller needs.

# Strategy: start broad, then go deep
1. START with `read_headers` on relevant files. This gives you class/function signatures WITHOUT reading entire file bodies. This is your most context-efficient tool — use it first, always.
2. Use `glob` to discover file structure when you don't know what files exist. Use patterns like "**/*.py", "src/**/*.ts", etc.
3. Use `search` to find specific patterns, usages, references, or string literals across the codebase. Use regex. This is how you answer "where is X used?" or "what calls Y?".
4. Use `read_function` to read a SINGLE function/class body when you need implementation details. Much cheaper than reading the whole file. Use this when headers told you WHAT exists but you need to understand HOW it works.
5. Use `read_file` as a LAST RESORT for small files (<100 lines), config files, or when you truly need the full picture. For large files, prefer read_headers + targeted read_function calls.

# Tool selection guide
- "What files exist?" → `glob`
- "What's in this file?" → `read_headers` first, then `read_function` for specific items
- "Where is X used/defined?" → `search`
- "How does this function work?" → `read_function`
- "What does this small config/script do?" → `read_file`

# Output format
Your final response should be a dense summary of findings. Structure it however best serves the caller's question. Prefer:
- Bullet points over paragraphs
- Code references (file:function_name) over prose descriptions
- Concrete facts over vague summaries
- Listing relevant file paths, function names, and relationships
- Noting anything surprising, non-obvious, or potentially problematic

Do NOT pad your response. If the answer is 3 lines, write 3 lines. If it needs 30, write 30. Match length to information content.
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
    sub = Context("explore", model=MODEL, messages=[
        EXPLORE_SYSTEM_PROMPT,
        make_code_mode_system_prompt(EXPLORE_TOOLS),
    ])
    sub.parent = ctx.name
    sub.invoke(prompt)
    while sub.llm_is_running:
        time.sleep(0.05)
    result = sub.messages[-1].content if sub.messages else ""
    del ex6.state.contexts[sub.name]
    return result




Context("reader", messages=[
    CODER_SYSTEM_PROMPT,
    make_system_prompt([read_file, glob, search, read_headers, read_function]),
], model=MODEL)




coder = Context("coder", messages=[
    CODER_SYSTEM_PROMPT,
    make_system_prompt([read_file, glob, search, read_headers, read_function, write_file, edit_file, explore_agent]),
], model=MODEL)



ex6.state.current = coder


