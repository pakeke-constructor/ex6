

# tasks / goals:

Overarching goal:  
Be a harness where every token is 100% explicit.
Every tool is explicit. EVERYTHING, every piece of control flow -> explicit.



# ===============================
# TASKS:
# ===============================



- Make gitignore handling dynamic per `ctx.cwd`, not import-time process cwd.




I want to make it so it's easier for agents to create their own little local branch, and can make changes easily.



Discord bot for ex6?
Being able to integrate clanker into codetheory Ltd server better?





<cwd-agents>
Overarching goal: make an agent that has reference to ex6 codebase;
FROM ANY CODEBASE.
make it so you call setup_ex6_agent(), to setup this agent in ANY repository. It should be well-aware of _ex6 plugins and how they work.
Make it so the agent can swap between working-directory with safe_cwd() function.
<cwd-agents>




- IDEA INFRASTRUCTURE:
- Agents automatically author and maintain their own skill/context files, seeded by a human-defined list of core concepts, with level-of-detail variants for context-efficient runtime injection.
- The idea is that over time, for every project, you'll build up a SUPER ROBUST ecosystem of contexts and skills.

^^^ EXAMPLES: 
- ev/q buses would become a core `idea`.
- knowing how best to write ui code/layout would become an `idea`.
- writing animations simplfy/robustly (ie with state-robust incremental timers) is an `idea`
ideas would be iterated on / tuned when the user does `/tune` command.
(That way, it doesnt just end up like slop.)

ANOTHER GOOD IDEA:
when writing a skill, agents are forced to choose a "template structure" for said skill.
This way, they don't just ramble about slop. They end up with a smart, well-structured, and well-scoped skill file.




<better_interop>
SPIKE: 
What if agents could "interact" with ex6 much better?
- Have tools to set/get users clipboard?
- Send prompts to other agents?
- Store data in ex6? like a buffer? 
- Look at / change settings?

EVEN BROADER:
What kinds of UX things would make ex6 easier to work with?
Maybe prompts as a first-class primitive?
</better_interop>




- Add this to system-prompt:
"When there is a difficult bug, don't start by trying to fix it. Instead, start by writing a test that reproduces the bug. Then, have subagents try to fix the bug and prove it with a passing test."





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

