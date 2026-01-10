


# Claude code system prompt:

## QUESTION:  
What is the structure of claude-code's system prompt?


## ANSWER:
```
● The docs say it's "not published" - but I am Claude Code right now. I can tell you the structure I'm seeing:

Claude Code System Prompt Structure

1. Tool Definitions (JSON schema for each tool)

2. Core Identity
    - "You are Claude Code, Anthropic's official CLI..."
    - Purpose statement

3. Safety/Security Rules
    - IMPORTANT blocks for security constraints

4. Tone and Style
    - Emoji usage, markdown, conciseness
    - Output formatting rules

5. Professional Objectivity
    - Technical accuracy focus
    - Avoid excessive praise/validation

6. Task Management
    - TodoWrite tool usage guidance
    - Examples of good task tracking

7. Asking Questions
    - When/how to use AskUserQuestion

8. Doing Tasks (Core Workflow)
    - Read before modifying
    - Security awareness (OWASP)
    - Avoid over-engineering
    - Git commit/PR workflows

9. Tool Usage Policy
    - When to use which tools
    - Parallel vs sequential calls
    - Specialized tools over bash

10. Environment Info
    - Working directory, platform, date
    - Model info

11. [User context injected here - CLAUDE.md, etc.]
```















## KEY OBSERVATIONS:
- Tools defined first - Model knows capabilities upfront
- Heavy on examples - Lots of <example> blocks
- Explicit "IMPORTANT" markers - For critical rules
- Workflow-specific sections - Git, PRs, task management get dedicated blocks



# QUESTION: 
Show me how `<example>` blocks are used.
And expand the tool-usage policy section for me. 


● Here's what I'm seeing:
```
  Example Block Format

  They're fairly verbose and structured. Here's the pattern:

  <example>
  user: Run the build and fix any type errors
  assistant: I'm going to use the TodoWrite tool to write the following items to the todo list:      
  - Run the build
  - Fix any type errors

  I'm now going to run the build using Bash.

  Looks like I found 10 type errors. I'm going to use the TodoWrite tool to write 10 items to the todo list.

  marking the first todo as in_progress

  Let me start working on the first item...

  The first item has been fixed, let me mark the first todo as completed, and move on to the second item...
  ..
  ..
  </example>

  Also negative examples showing what not to do:

  <good-example>
  pytest /foo/bar/tests
  </good-example>
  <bad-example>
  cd /foo/bar && pytest tests
  </bad-example>

  Tool Usage Policy (Expanded)

  Here's the actual content:

  # Tool usage policy
  - When doing file search, prefer to use the Task tool to reduce context usage.
  - Proactively use the Task tool with specialized agents when task matches agent description.       

  - When WebFetch returns a redirect to different host, immediately make new request with redirect URL.

  - Call multiple tools in single response if no dependencies between them.
    Maximize parallel tool calls. But if calls depend on previous results,
    call sequentially. Never use placeholders or guess missing parameters.

  - If user says run tools "in parallel", MUST send single message with
    multiple tool use content blocks.

  - Use specialized tools instead of bash when possible:
    - Read for reading files (not cat/head/tail)
    - Edit for editing (not sed/awk)
    - Write for creating files (not cat heredoc or echo redirection)

    Reserve bash for actual system commands requiring shell execution.
    NEVER use bash echo to communicate with user.

  - VERY IMPORTANT: When exploring codebase for context (not needle queries
    for specific file/class/function), use Task tool with subagent_type=Explore
    instead of search commands directly.

  <example>
  user: Where are errors from the client handled?
  assistant: [Uses Task tool with subagent_type=Explore instead of Glob/Grep directly]
  </example>
```


