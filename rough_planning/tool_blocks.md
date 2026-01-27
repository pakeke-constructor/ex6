


## tool-blocks:

Inspired by cloudflare's `code-mode`.
https://blog.cloudflare.com/code-mode/


The better way to do function calling for LLMs.

Instead of using `tools` properly in the API,
have the LLM output a "tool" block.

eg:

```tools

read_file("src/g.py")

write_file("test.txt", "TEST STRING")

glob("src/**")

```
^^^ in reality, this is just sandboxed python.
(Use RestrictedPython library for "good enough" sandboxing)


## Rendering:
Okay, and how are tools rendered?  
A: Use `@output_renderer` to clear the ```tools ``` block,
and track calls as they go:  
```
read_file("src/g.py") -> RUNNING
write_file("test.txt", "TEST STRING") -> COMPLETED
glob("src/**") -> RUNNING
```


## And how is it fed back into LLM?
Each tool-call should append it's own little header.
eg:
```
<tool_result read_file("src/g.py)>

...
... file contents
... blah blah
...
</tool_result>


<tool_result glob(src/**)>
g.py
entities.py
...
</tool_result>

```


