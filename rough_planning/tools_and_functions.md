

# Tools:
Tools are standardized across LLM providers.

- Claude calls them "tools"
- OpenAI calls them "functions"
- Gemini calls them "function declarations"

Example:
```json
{
  "name": "edit_file",
  "description": "Edit a file using search/replace",
  "input_schema": {
    "type": "object",
    "properties": {
      "path": {"type": "string"},
      "search": {"type": "string", "description": "Exact text to find"},
      "replace": {"type": "string", "description": "Text to replace it with"}
    }
  }
}
```

No point in reinventing the wheel with janky pattern matching and calls.   
Just use `tools`, it's more efficient.







## According to claude:
● Yes, tools are defined in my system prompt as function schemas. I see something like:

<function>
{
"name": "Task",
"description": "Launch a new agent to handle complex, multi-step tasks...",
"parameters": {
    "prompt": "The task for the agent to perform",
    "subagent_type": "Explore | Plan | Bash | ...",
    "run_in_background": boolean,
    ...
}
}
</function>

Each tool has a JSON schema defining its parameters. The system prompt includes lengthy descriptions for each - when to use them, when NOT to use them, examples, etc.

For Task specifically, my instructions list available subagent types:
- Explore: Fast codebase exploration (Glob, Grep, Read)
- Plan: Design implementation approaches
- Bash: Command execution
- general-purpose: Full tool access for complex tasks
- etc.

When I invoke it, I just emit a tool call:
<invoke name="Task">
<parameter name="subagent_type">Plan</parameter>
<parameter name="prompt">Design the implementation...</parameter>
</invoke>

Then I receive back a <function_results> block with whatever the subagent returned. I don't see how it got there - just the final output. 

The guidelines are basically: use Explore for searching/understanding, Plan for designing, don't use Task when a direct Read/Grep would be faster for a specific lookup.

