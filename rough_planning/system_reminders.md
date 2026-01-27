

## System reminders:

claude-code often has "system reminders" which output little notifications/blobs of text to the agent to help it stay on track.


See:
https://jannesklaas.github.io/ai/2025/07/20/claude-code-agent-design.html


### What sys reminders would be good here?
Maybe reminders like: 
```
[pruned context: <message>]

<system_reminder>
You are past 150k tokens. Be wary about starting new tasks.
It is much better to refuse to do something because the context window is too long, than it is to do something incorrectly.
</system_reminder>
```
^^ just experiment a bit.


