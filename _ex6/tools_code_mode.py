

import ex6


@ex6.override
def call_tools(ctx:ex6.Context, llmres:ex6.LLMResult) -> bool:
    '''
    allow code-mode tool calling.

    Instead of outputting typical json format for tool-calls,
    Agent should output:

    ```tools
    read_file("file.py")
    write_file("file.py", txt)
    ```

    '''
    return True



