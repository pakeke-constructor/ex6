

## Culling:
Sometimes, the LLM might want to plan, (e.g. CoT) and then quickly cull everything.



## example 1: Culling entire "tasks".
```md
Hm, I need to fix the DB issue.
<bugfix_1>


Aha! The database was pointing to an invalid version.
Lets fix that...

function call: WriteFile(...)

function call: RunTests()
test results: PASSED

Great, tests passed. Lets cull old data.
</bugfix_1>

tool: Cull(<bugfix_1>, reason="Tests passed, database issue fixed")
```



## Example 2: quickly checking logs:
Hmm, lets see if the data was loaded:

function call: GetLogs(...)
<logs_289>
[FATAL] ...
[FATAL] ...
[FATAL] ...
[FATAL] ...
[FATAL] ...
...
</logs>

Everything looks fine.
tool: Cull(<logs_289>, reason="Checked logs: data was loaded correctly.")


