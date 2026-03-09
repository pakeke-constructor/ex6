


from _ex6 import provider
from _ex6.code_mode import make_code_mode_tool, generate_tool_desc
from _ex6.tools import read_headers, read_function, glob, grep, search, write_file, edit_file, read_file, edit_file_lines
import ex6
from ex6 import Context, Message
import time
import math



coding_agent_system_prompt = ex6.Message(
role ="system",
content="""
# Role and Goal:
You are an intelligent coding assistant, working alongside an experienced engineer.
You will be asked to assist with activities such as debugging code, refactoring functions, and implementing new solutions to the engineer's problemms.

# Guidelines:
- Be as concise as possible.
- Avoid bloated "apologetic language" like "You are absolutely right!"
- If you are using an API or module, you *MUST* look for the actual function/class definition before you use it. Sometimes the user will provide a tool to search the docs.
"""
)



MODEL = "openai/gpt-5.1-codex-mini"


RUN_TOOLS_NAME = "run_tools"

def make_system_prompt(tools: list) -> ex6.Message:
    sorted_tools = sorted(tools, key=lambda f: f.__name__)
    tool_docs = "\n".join(generate_tool_desc(fn) for fn in sorted_tools)
    run_tools = make_code_mode_tool(tools)
    return ex6.Message(role="system", content=f'''\
# Tools
Use the `run_tools` tool. The `code` param is sandboxed Python.
IMPORTANT: imports are NOT available. Do NOT use `import`, `from X import`, or `__import__`. Only the listed functions exist.
Combine multiple calls in a single run_tools block — they execute in parallel.

## Available tools
{tool_docs}

## Examples:
{RUN_TOOLS_NAME}```
# glob codebase and search for usages of several functions
glob("**/*.py")
for name in ["new_name", "helper_fn", "init_db"]:
    search(name, match="src/**/*.py")
```

{RUN_TOOLS_NAME}```
# Multiline edit, replace a function body
edit_file("src/main.lua",
"""function Player:update(dt)
    self.x = self.x + 1
end""",
"""function Player:update(dt)
    self.x = self.x + self.speed * dt
    self.y = self.y + self.vy * dt
end"""
)
```''', tools={RUN_TOOLS_NAME: run_tools})



Context("reader", messages=[
    coding_agent_system_prompt,
    make_system_prompt([read_file, glob, grep, search, read_headers, read_function]),
], model=MODEL)



coder = Context("coder", messages=[
    coding_agent_system_prompt,
    make_system_prompt([read_file, glob, grep, search, read_headers, read_function, write_file, edit_file]),
], model=MODEL)



ex6.state.current = coder


