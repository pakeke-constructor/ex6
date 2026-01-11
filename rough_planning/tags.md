

# Tags:

LLMs *really really* like it when you tag things.
like
```md
<tone>
Speak like a pirate when coding
<tone>
```


Example with tool-usage:

```md

## Discover tool:
Discover-tool will spawn a cheap subagent that reads files and returns information.  
Use it freely for SIMPLE tasks where there isn't much deep reasoning.

<good-example>
Hmm... I should check if entities are buffered when destroyed.
TOOL-CALL: Discover("YES or NO: Are entities buffered when destroyed?", ["ecs/entities.py", "ecs/ecs.py"])
[subagent returns YES.]
[CoT is pruned, your context window will be something like:]

Q: Are entities buffered when destroyed?
A: YES
</good-example>


<bad-example>
I need to find this complex bug. Lets spin up a subagent to do so:
-> Discover("Find the cause of this difficult memory leak. It's either in foobar or baz.", ["src/foobar.py", "src/baz.py"])
[subagent returns some hallucinated data.]
</bad-example>


<bad-example>
I need to find this complex bug. Lets spin up a subagent to do so:
-> Discover("Find the cause of this difficult memory leak. It's either in foobar or baz.", ["src/foobar.py", "src/baz.py"])
[subagent returns some hallucinated data.]
</bad-example>



```


