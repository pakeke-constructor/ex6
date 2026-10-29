"""Throwaway prototype: native-scrollback TUI (Claude Code / Codex style) + modal selection.
Run in Windows Terminal: python scroll_proto.py

- Selection mode: alt-screen (full-screen, redrawn every frame). Agents keep running in background.
- Work mode: normal screen. Finished messages printed ONCE -> real scrollback (wheel, select, copy all native).
- Live area (streaming text + running tool spinner + input) sits right under history.
- Esc work->selection->quit. Returning to the same agent needs no reprint (terminal keeps the main screen).
- Switching agent / resize / /edit -> clear screen+scrollback and reprint everything.
"""

import os
os.environ.setdefault('ESCDELAY', '25') # reduce escape key delay (ms)
# (if using SSH, you might want to set this higher. Ask some LLM to explain why.)

import random, sys, textwrap, time
from blessed import Terminal

term = Terminal()
out = sys.stdout
INPUT_H = 4  # divider + input + divider + footer
LEFT_W = 34
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

USER_BG = "\x1b[48;5;237m"
DIM, GREEN, CYAN, YELLOW, BOLD, INVERT, RESET = "\x1b[2m", "\x1b[32m", "\x1b[36m", "\x1b[33m", "\x1b[1m", "\x1b[7m", "\x1b[0m"
WORDS = ("the model ponders scrollback terminals alt screens cursor rows reflow "
         "flicker synchronized output wheel selection copy paste viewport").split()
TOOLS = ["read_file(path='ex6.py')", "search(pattern='def flush')", "bash(command='git status')",
         "read_headers(file='_ex6/tools.py')", "edit_file(file='ex6.py')"]

state = dict(w=0, h=0, hist_rows=0, ctx=None)  # ctx: agent currently printed on the main screen


def spin():
    return SPIN[int(time.time() * 12) % len(SPIN)]


def move(y, x=0):
    return f"\x1b[{y + 1};{x + 1}H"


def fake_word():
    r = random.random()
    if r < 0.04: return "\n\n"
    if r < 0.07: return "\n    def foo(x): return x * 2\n"
    return random.choice(WORDS) + " "


def fake_turn():
    steps = [("text", random.randint(15, 60))]
    for _ in range(random.randint(1, 3)):
        steps.append(("tool", random.choice(TOOLS), random.uniform(0.8, 2.5)))
    steps.append(("text", random.randint(40, 300)))
    return steps


class Agent:
    def __init__(self, name, intro):
        self.name = name
        self.messages = [("assistant", intro)]
        self.pending = None  # streaming text
        self.left = 0        # words left in current stream
        self.tool = None     # (label, end_time) of running tool
        self.steps = []      # queued work: ("text", n_words) | ("tool", label, secs)
        self.printed = 0     # messages committed to main screen
        self.pc = 0          # lines of `pending` already committed to main screen
        self.inp = ""

    def busy(self):
        return bool(self.steps) or self.pending is not None or self.tool is not None

    def status(self):
        return self.tool[0] if self.tool else "streaming" if self.pending is not None else "thinking" if self.steps else "idle"

    def tick(self):
        if self.pending is not None:
            self.pending += fake_word()
            self.left -= 1
            if self.left <= 0:
                self.messages.append(("assistant", self.pending))
                self.pending = None
        elif self.tool:
            if time.time() >= self.tool[1]:
                self.messages.append(("tool", f"{self.tool[0]}  -> {random.randint(10, 3000)} lines"))
                self.tool = None
        elif self.steps:
            step = self.steps.pop(0)
            if step[0] == "text": self.pending, self.left = "", step[1]
            else: self.tool = (step[1], time.time() + step[2])


def render(role, text, w):
    lines = []
    for para in text.split("\n"):
        for ln in textwrap.wrap(para, w - 2, drop_whitespace=False) or [""]:
            if role == "user": lines.append(USER_BG + (" " + ln).ljust(w - 1) + RESET)
            elif role == "tool": lines.append(GREEN + " ✓ " + ln + RESET)
            elif ln.startswith("    "): lines.append(CYAN + ln + RESET)
            else: lines.append(ln)
    return lines + [""]


# ---------- work mode (normal screen, native scrollback) ----------

def commit(lines):
    """Append lines to history. Past the bottom row, LF scrolls the whole screen up (top row -> native scrollback)."""
    s = []
    for ln in lines:
        if state["hist_rows"] < state["h"]:
            s.append(move(state["hist_rows"]) + "\x1b[2K" + ln)
            state["hist_rows"] += 1
        else:
            s.append(move(state["h"] - 1) + "\n\x1b[2K" + ln)
    return "".join(s)


def reprint(a):
    state["w"], state["h"] = term.width, term.height
    state["hist_rows"], state["ctx"] = 0, a
    a.printed, a.pc = 0, 0
    return "\x1b[H\x1b[2J\x1b[3J" + sync(a)


def sync(a):
    """Commit messages finished since last frame. Only the first can be partially committed (former stream)."""
    s = ""
    for role, text in a.messages[a.printed:]:
        s += commit(render(role, text, state["w"])[a.pc:])
        a.pc = 0
    a.printed = len(a.messages)
    return s


def pending_lines(a):
    if a.pending is None: return []
    return render("assistant", a.pending + "█", state["w"])[:-1][a.pc:]


def flush_overflow(a):
    """If streaming text won't fit on screen, commit its oldest lines into history."""
    lines = pending_lines(a)
    extra = len(lines) - (state["h"] - INPUT_H - 1)
    if extra <= 0: return ""
    a.pc += extra
    return commit(lines[:extra])


def draw_live(a):
    w, h = state["w"], state["h"]
    rows = pending_lines(a)
    if a.tool:
        rows.append(f"{YELLOW} {spin()} {a.tool[0][:w - 4]}{RESET}")
    rows.append(DIM + "─" * (w - 1) + RESET)
    rows.append("> " + a.inp[-(w - 4):] + "▏")
    rows.append(DIM + "─" * (w - 1) + RESET)
    footer = f"{a.name} | {a.status()} | /long /tool /edit | ctrl+r reprint | esc agents"
    rows.append((f"{spin()} " if a.busy() else "  ") + DIM + footer[:w - 3] + RESET)
    s = ""
    overflow = state["hist_rows"] + len(rows) - h
    if overflow > 0:  # push history up into scrollback to make room
        s += move(h - 1) + "\n" * overflow
        state["hist_rows"] -= overflow
    top = state["hist_rows"]
    s += move(top) + "\x1b[J"
    s += "".join(move(top + i) + r for i, r in enumerate(rows))
    return s


def work_key(a, key):
    if key == "\x12":  # ctrl+r
        return reprint(a)
    if key.code == term.KEY_BACKSPACE:
        a.inp = a.inp[:-1]
    elif key.code == term.KEY_ENTER:
        text, a.inp = a.inp, ""
        if text == "/long":
            a.messages.append(("assistant", "\n".join(f"line {i}: " + " ".join(random.choices(WORDS, k=8)) for i in range(200))))
        elif text == "/tool":
            a.steps.append(("tool", random.choice(TOOLS), 2.5))
        elif text == "/edit":
            a.messages[0] = (a.messages[0][0], "[EDITED] " + a.messages[0][1])
            return reprint(a)
        elif text:
            a.messages.append(("user", text))
            a.steps += fake_turn()
    elif not key.is_sequence and key.isprintable():
        a.inp += key
    return ""


# ---------- selection mode (alt screen, full redraw) ----------

def draw_selection(agents, sel):
    w, h = term.width, term.height
    s = "\x1b[H\x1b[2J" + move(0, 1) + BOLD + "AGENTS" + RESET
    for i, a in enumerate(agents):
        name = f" {a.name:<12}{len(a.messages):>3} msgs"
        mark = YELLOW + spin() + RESET if a.busy() else DIM + "·" + RESET
        s += move(2 + i) + (INVERT + name + RESET if i == sel else name) + f" {mark} " + DIM + a.status()[:LEFT_W - len(name) - 4] + RESET
    s += move(h - 1, 1) + DIM + "up/down select | enter open | esc quit" + RESET

    # right panel: tail of selected agent's conversation
    a, px = agents[sel], LEFT_W + 2
    pw = max(10, w - px - 1)
    lines = []
    for role, text in a.messages:
        lines += render(role, text, pw)
    if a.pending is not None: lines += render("assistant", a.pending + "█", pw)
    if a.tool: lines.append(f"{YELLOW} {spin()} {a.tool[0]}{RESET}")
    lines = lines[-(h - 3):]
    for y in range(h - 1):
        s += move(y, LEFT_W) + DIM + "│" + RESET
    for i, ln in enumerate(lines):
        s += move(1 + i, px) + ln
    return s


def main():
    agents = [Agent("main", "Native scrollback prototype. Type something and hit enter. "
                            "Try mouse wheel, selecting text, resizing the window."),
              Agent("explore", "Explore agent. Already busy in the background."),
              Agent("reviewer", "Reviewer agent. Idle.")]
    agents[1].messages.append(("user", "map out the render loop"))
    agents[1].steps = fake_turn() * 3
    mode, sel = "selection", 0
    with term.cbreak(), term.hidden_cursor():
        out.write("\x1b[?1049h")
        try:
            while True:
                for a in agents: a.tick()
                key = term.inkey(timeout=0.03, esc_delay=0)
                frame = ""
                if mode == "selection":
                    if key.code == term.KEY_ESCAPE: break
                    if key.code == term.KEY_UP: sel = (sel - 1) % len(agents)
                    if key.code == term.KEY_DOWN: sel = (sel + 1) % len(agents)
                    if key.code == term.KEY_ENTER:
                        mode, a = "work", agents[sel]
                        frame += "\x1b[?1049l"
                        if state["ctx"] is not a or (term.width, term.height) != (state["w"], state["h"]):
                            frame += reprint(a)
                    else:
                        frame += draw_selection(agents, sel)
                else:
                    a = agents[sel]
                    if key.code == term.KEY_ESCAPE:
                        mode = "selection"
                        frame += "\x1b[?1049h" + draw_selection(agents, sel)
                    else:
                        if (term.width, term.height) != (state["w"], state["h"]):
                            frame += reprint(a)
                        if key: frame += work_key(a, key)
                        frame += sync(a) + flush_overflow(a) + draw_live(a)
                out.write("\x1b[?2026h" + frame + "\x1b[?2026l")
                out.flush()
        finally:
            if mode == "selection":
                out.write("\x1b[?1049l")
            if state["ctx"]:
                out.write(move(state["hist_rows"]) + "\x1b[J")
            out.flush()


if __name__ == "__main__":
    main()
