
import ex6


s = '''

SPINNER
SPINNER
SPINNER

# hello.
*I am italic.*
### i am a 3rd heading!
and im a `func()` call.
- a
- bbb
- cccd


```python
def func(x: int):
    for i in range(10):
        print(i)
        break
    return 0.0
```

'''

@ex6.override
def invoke_llm(ctx):
    """Override this to use real LLM."""
    yield ex6.ResponseChunk("text", s)


