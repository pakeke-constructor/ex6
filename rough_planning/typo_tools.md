

## Mistake tool:
If you believe the user has made a mistake, (i.e. sent text early, or unintelligable text) you should call the Mistake() tool.
<example>
User: "Ok. your next task is to mak"
Assistant: tool Mistake()
</example>


## Typo tool:
If you believe the user has made excessive typos or errors that would impede performance, you should offer to correct/clarify it with the Typo() tool.
<example>
User: "why is teh bugffer not cleard proeprly?"
Assistant: tool Typo("Why is the buffer not cleared properly?")
</example>


