
from typing import Optional
import subprocess
import threading
import ex6


@ex6.command
def clr(name: Optional[str]):
    ctx = ex6.state.contexts.get(name) if name else ex6.state.current
    if not ctx: return
    ctx.clear()


@ex6.command
def delete(name: Optional[str]):
    ctx = ex6.state.contexts.get(name) if name else ex6.state.current
    if not ctx: return
    del ex6.state.contexts[ctx.name]
    if ex6.state.current is ctx:
        ex6.state.current = None


@ex6.command
def fork(name: Optional[str]):
    ctx = ex6.state.current
    if not ctx: return
    ctx.fork(name)


@ex6.command
def crash():
    raise RuntimeError("Crash!")


def _llm_one_shot(model: str, system: str, user: str) -> str:
    """Synchronously run one LLM call. Returns assistant text."""
    ctx = ex6.Context(name="__tmp_cm__", model=model)
    ctx.messages.append(ex6.Message(role="system", content=system))
    ctx.messages.append(ex6.Message(role="user", content=user))
    result_text = []
    for item in ex6.invoke_llm(ctx):
        if isinstance(item, ex6.ResponseChunk) and item.type == "text":
            result_text.append(item.content)
    del ex6.state.contexts["__tmp_cm__"]
    return "".join(result_text).strip()



CM_SYSTEM_PROMPT = """
You write git commit messages.
Use conventional commits: feat(...), fix(...), chore(...), refactor(...), etc.
One line only. No quotes. No explanation.
Be extremely concise, grammatical correctness is not important.
"""


@ex6.command
def cm(msg: Optional[str]):
    """Generate a commit message from git diff and commit."""
    def run():
        ex6.enter_scroll_mode()

        # mark untracked files with intent-to-add so they show in diff
        subprocess.run(["git", "add", "."], capture_output=True)

        diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True).stdout
        if not diff:
            print("No changes to commit.")
            return

        from _ex6.models import M
        model = M.GEMINI31_FLASH_LITE.id

        hint = f"User hint: {msg}" if msg else ""
        system = CM_SYSTEM_PROMPT
        user = f"Write a commit message for this diff:{hint}\n\n{diff[:8000]}"

        print("Generating commit message...")
        commit_msg = _llm_one_shot(model, system, user)
        print(f"Commit: {commit_msg}")

        subprocess.run(["git", "add", "."], capture_output=True)
        result = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
        if result.returncode == 0:
            print("Committed.")
        else:
            print(f"git commit failed:\n{result.stderr}")

    threading.Thread(target=run, daemon=True).start()



@ex6.command
def help():
    ex6.enter_scroll_mode()
    print("Commands:")
    for name, (fn, spec) in sorted(ex6._commands.items()):
        args = " ".join(f"<{a}>" for a, _ in spec)
        doc = (fn.__doc__ or "").strip()
        line = f"  /{name} {args}".rstrip()
        print(f"{line}  {doc}" if doc else line)



@ex6.command
def show_ctx():
    ex6.enter_scroll_mode()
    print("="*30)
    print("CONTEXT WINDOW")
    print("="*30)
    print("\n\n")
    term = ex6.state.term
    colors = {"system": term.blue, "user": term.green, "assistant": term.red, "tool": term.yellow}
    ctx = ex6.state.current
    for msg in ctx.messages:
        content = msg.get_msg(ctx) if callable(msg.content) else msg.content
        color = colors.get(msg.role, lambda x: x)
        print(f"{color(f'[{msg.role}]')}\n{color(content)}\n")


