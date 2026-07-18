

# tasks / goals:

Overarching goal:  
Be a harness where every token is 100% explicit.
Every tool is explicit. EVERYTHING, every piece of control flow -> explicit.



# ===============================
# TASKS:
# ===============================



- Make gitignore handling per `ctx.cwd`, not import-time process cwd.


THEN: Oli, you could use ex6 to optimize and organize your life a bit more.
Discord bot for ex6? checklists / goal tracking? running stuff in background, etc

<cwd-agents>
Overarching goal: make an agent that has reference to ex6 codebase;
FROM ANY CODEBASE.

make it so you call setup_ex6_agent(), to setup this agent in ANY repository. It should be well-aware of _ex6 plugins and how they work.

Make it so the agent can swap between working-directory with safe_cwd() function.
<cwd-agents>


<compression>
ex6 compression idea:
"condense()" should be a function with 0 args. 

when 'condense' is called, instead of instantly condensing, it should inject a bunch of guidance and information into the ctx window as a tool-result: (just return string from the tool-call)

 Information to be injected:
- list all checkpoints, AND cumulative token-counts. eg
- checkpoint 1: "objective blah" (10k toks)
- checkpoint 2: "blah blh" (25k toks)
- checkpoint 3: "hjhdfjdfh" (32k toks)

- Add a new method to code-mode tools, called condense_to_checkpoint(...)
- list the method signature for condense_to_checkpoint(...)
- Add guidance for how to use condense_to_checkpoint(...), practices, etc
- If less than 15k tokens used, tell the LLM "You probably don't need to condense, there's hardly any tokens used"
- Tell the LLM how to choose the checkpoint to condense to, by calling appropriate tool.
</compression>


- sandboxing for ex6 agents (docker container)


- ex6 agent skills



- Tell LLMs to write comments in code as a form of "CoT" thinking


- system reminder infrastructure.
-> Do exactly what was asked. Nothing more, nothing less. Never create files unless necessary. Never add docs/READMEs unprompted.
https://claude.ai/share/a720b25a-9705-461a-9ebf-25aa0adbca12


- Create a new agent `debugger`, that uses codex 5.3 Apparently codex is excellent at debugging.


- Create a new agent `tester`, (codex 5.3) Codex excels at testing and solving issues.
(Generates test-cases, finds edge-cases, runs in a loop; then feeds output to `main` agent)



- IDEA INFRASTRUCTURE:
- Agents automatically author and maintain their own skill/context files, seeded by a human-defined list of core concepts, with level-of-detail variants for context-efficient runtime injection.
- The idea is that over time, for every project, you'll build up a SUPER ROBUST ecosystem of contexts and skills.

^^^ EXAMPLES: 
- ev/q buses would become a core `idea`.
- knowing how best to write ui code/layout would become an `idea`.
- writing animations simplfy/robustly (ie with state-robust incremental timers) is an `idea`
ideas would be iterated on / tuned when the user does `/tune` command.
(That way, it doesnt just end up like slop.)




<better_interop>
SPIKE: 

What if agents could "interact" with ex6 much better?
- Have tools to set/get users clipboard?
- Send prompts to other agents?
- Store data in ex6? like a buffer? 
- Look at / change settings?
</better_interop>




- Add this to prompt:
"When I report a bug, don't start by trying to fix it. Instead, start by writing a test that reproduces the bug. Then, have subagents try to fix the bug and prove it with a passing test."



- NEW TOOL: read_warnings("my_file.py")  reads warnings/errors from file (pylance, LuaLS)


- plugin: similar to `SKILLS.md`. Allow agents to dynamically pull in skills. 


- system-reminders infrastructure:
- sys-reminder: notify LLMs if a file has been modified


- choice/options plugin, like claude-code.


- In-editor LLM invocation (like _99 from primeagen):
The real value: invoke LLMs directly from your editor without leaving your flow.
use `watchfiles` to monitor project files. User ends a line with `;;;` to trigger.
```lua
function my_func()
refactor this to use async;;;
-- ^^^ the system will detect this text, delete it, and fire a callback.
-- An agent will boot up instantly and start working.
end
```
Could even do other cooler stuff, like different char-combinations:
```py
def my_func():
    see if we can remove this. ;;;e
    # a `e` at the end could mean like: `explore`? so 
    # maybe different characters mean different things:

    # s = simplify code
    # p = create a plan, don't edit anything

    # not sure. maybe best to keep it simple. See what works first; dont guess features
```


- tools can block for results too..?
```tools
res = read_file("foo.py")
make_subagent("find all entities in this file: " + res.get())
# ^^^ the `res.get()` thing should block until got results.
```
NOTE: IT DOESNT NEED TO BE `res.get()`.  
We should ideally use a builtin python abstraction.




## PROBLEM-SOLVING-AGENTS:
One thing I really want to experiment with is having agents that are specialized in finding solutions of a certain type.
Because IME, the thing that slows me down a lot with agents is just that they don't find the best solution, even though it seems obvious to us humans. It means I just have to check everything, which sucks.

So I wonder if spinning up different subagents that are specialized in certain "solution classes" could work-
eg:
Agent-1: look for a solution by changing the structure of the objects that are being called.
Agent-2: look for a solution by relaxing the problem requirements
Agent-3: look for a solution by encoding some of the surrounding data as first class functions or objects
Agent-4: look for a solution by replacing/removing objects .... etc (opposite of the above)
... etc

Because one thing I have noticed is that the frontier intelligence models are generally really good at knowing if a solution is elegant, but they can't necessarily come up with the solution on it's own.
But its like, as soon as they see the solution, they instantly recognize it's utility

