

## COT COMMENTS:

I've noticed LLMs write comments explaining the task before writing code.
EG:
```py
def big_complex_func():
    ...
    # lerp between world-space and scene-space
    ... (code)
```
This makes sense, as it likely improves performance.


What if LLMs explicitly wrote CoT comments, and in a post-process step, a script cleared the dirty CoT comments?

Eg:
```py
def big_complex_func():
  ...
  # [thinking] here, I need to lerp from world-space to screen-space.
  # [thinking] a more efficient way would be to use foobar, but (blah blah)
  # [thinking] Wait! I can do XYZ because blah blah blah
  XYZ()
```

and then in post-processing, strip the `# [THINKING]` lines from the model's context, and from the file; so you end up with output:
```py
def big_complex_func():
  ...
  XYZ()
```




## Non-code cot-comments:
LLMs should also be able to write CoT comments when they aren't coding too.
e.g. when they are having conversation, and it gets complicated.



