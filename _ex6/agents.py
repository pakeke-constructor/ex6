


from _ex6 import provider
from _ex6.code_mode import make_code_mode_system_prompt
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


Context("reader", messages=[
    coding_agent_system_prompt,
    make_code_mode_system_prompt([read_file, glob, grep, search, read_headers, read_function]),
], model=MODEL)



coder = Context("coder", messages=[
    coding_agent_system_prompt,
    make_code_mode_system_prompt([read_file, glob, grep, search, read_headers, read_function, write_file, edit_file]),
], model=MODEL)



ex6.state.current = coder




