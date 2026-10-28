# clicky: agents that do the UI busywork

## Motivation
User's words: "the nitty-gritty engineering stuff is what im good at, and the clicking through UI, signing up, navigating complex dashboards: That's what i want to automate." "im sure we can find a way for this"

Target prompt: "make a supabase DB for this project. go". The agent should sign up or log in, click through the dashboard, create the thing, and drop the keys into the project's `.env`. The user only steps in when a human is truly required.

Constraint (user): "Not looking to reinvent the wheel here". Reuse existing tools. No custom browser engine. No MCP client.

## Tool choice: browser-use CLI (default)
User wanted mouse control and human-like interaction ("browser-use or smthn?").
- It drives the user's real, logged-in Chrome over CDP. Logins and sessions just work.
- The agent writes Python against the browser (`click_at_xy`, `type_text`, `js`, page_info, accessibility tree). When a dashboard is weird, the agent writes the code it needs instead of being limited to fixed commands.
- Cloud stealth browsers (`browser-use auth login`, about $0.02/browser-hour) include CAPTCHA solving and residential proxies. Local Chrome is free. Use local for logged-in work, cloud for signups that hit bot walls.
- It's a CLI + skill file, so it fits ex6 (no schemas upfront). `browser-use skill` prints the skill doc and goes into `_ex6/_skills/browser.md`.
- Install: `uv` with Python 3.12, `uvx browser-use` / `browser-use`. Not the legacy package.
- Windows catch: their examples pipe bash heredocs (`browser-use <<'PY' ... PY`). ex6's shell is PowerShell, so check whether a here-string or a temp `.py` file works.

Fallbacks:
- playwright-cli (`attach --cdp=chrome` / `--extension`, `mousemove/mousedown/mouseup/drag`). Cheaper on tokens, but no CAPTCHA/stealth help.
- browser-harness (same team): minimal CDP-to-real-Chrome harness. The agent writes its own helpers.
- The service's own CLI/API, when one exists (supabase CLI, gh, vercel). The browser agent handles only what the CLI can't: signup, getting the first login token, billing, dashboard-only settings.

## Known blockers and how we get around them
1. Not logged in -> drive the user's real Chrome (browser-use). Prefer "Sign in with Google/GitHub" for every signup. That mostly skips CAPTCHAs, email verification and passwords.
2. CAPTCHA / bot walls -> retry in a browser-use cloud stealth browser.
3. 2FA / payment / anything that truly needs a human -> the browser stays visible. The agent calls the existing ask tool ("do the 2FA in the browser, then hit enter") and resumes.
4. Email verification codes/links -> the agent opens Gmail in the same logged-in Chrome. Only if that's painful: a read-only `read_inbox` tool (Gmail API/IMAP).
5. Secrets from dashboards (API keys, DB URLs) -> the agent writes them straight into the project `.env`. It doesn't echo them back into chat.
6. Prompt injection (web pages/emails can carry instructions). This matters more now because the agent controls the user's real logged-in browser.
   -> A separate browser agent Context with narrow tools: shell allowlisted to `browser-use ...`, ask, write `.env`. No general shell, no repo writes.
   -> `approve_tool` gates anything irreversible (payments, deletions, "confirm" on billing pages).
   -> The main coding agent delegates to it as a subagent ("go make a supabase project, return the env vars").

## Steps
1. Spike, no code: install browser-use, connect it to Chrome, check whether heredoc vs PowerShell works. Have an agent with the shell tool do the supabase prompt by hand. Write down every place it gets stuck.
2. Skill file `_ex6/_skills/browser.md`, taken from `browser-use skill` plus our rules: "prefer OAuth signup", "ask the human for 2FA/payment", "write secrets to .env", "cloud browser on bot walls".
3. Browser agent + subagent tool in `_ex6/agents.py` (like explore_agent): narrow tools.
4. Approval rules in an `approve_tool` handler: shell restricted to `browser-use ...`. Payment/delete actions need confirmation.
5. Only if the spike shows a need: `read_inbox` tool, or a thin `browser` tool wrapper if shell quoting on Windows is too painful.
6. Later, maybe: desktop control (pyautogui + screenshot as ImageAttachment) for native apps with no web UI. Out of scope for now.

## Relevant files
- `_ex6/agents.py`: agent definitions, tool groups, subagent pattern
- `_ex6/tools.py`: powershell tool, explore_agent (subagent template), ask tool
- `_ex6/skills.py` + `_ex6/_skills/`: progressive-disclosure skill docs
- `ex6.py`: `approve_tool` handler chain, `ImageAttachment`
