

# "How do we handle approval of changes? How to see diffs?"

Let git handle diffs. Its built for that. 
Generally, agents should write directly to files.

Then, we approve the changes ourselves after.


========================================


That way, subagents can look at working tree to critique code:
```py
import git
# GitPython plugin

import git
repo = git.Repo('.')
for d in repo.head.commit.diff(None, create_patch=True):
    print(f"--- {d.a_path}\n+++ {d.b_path}\n{d.diff.decode()}")
# ^^^ subagent can analyze the working tree changes!
```


and workflow can be tightly integrated:  
imagine: press `s` key, spins up agent to negotiate/simplify code

or imagine 2: An agent that is specialized in a particular area is spun up to check new code for bugs.


