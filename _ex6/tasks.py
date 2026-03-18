
'''
tasks.py: tools for LLMs to manage tasks.
TODO.


each task is a .md file inside of `.tasks/*`.
Task spec is a given by `TASKS/task_spec.md`.

Agents interact with these tasks via tools.

tool functions we want:


task_focus(id) # focuses on this task. stores in ctx.data["tasks:id"] = id
task_create(description) # returns id (base32 id:  `dc5`)

task_read(id = None) # if None, reads focused task
task_write_plan(full_plan, id=None) # if id=None, writes focused task

task_add_log(short_str, type="BLOCKER" or "PROGRESS" or "LEARNING" or "HUMAN")
# logs to focused task. 
# should be used to record progress, learnings, or blockers on tasks.
# (Auxiliary agent should use this automatically maybe...?)
# NOTE: ALL HUMAN INPUT SHOULD BE LOGGED AS A TASK

task_query_logs(query, id=None)
# if id=None, does focused task
# spins up (cheap) subagent to query the logs. eg:
# task_query_logs("have there ")

'''

