"""Frame profiler for the TUI main thread.
/prof           toggle: per-frame section timings + live HUD (top-right). Stopping writes a report.
/prof cpu [n]   cProfile the main thread for n seconds (default 5), write top functions.
Reports go to <get_folder("profiler")>/. _tui_loop wrapper is always installed (passthrough when idle);
other patches only while /prof is on.
"""
import sys
import time
import threading
import cProfile
import pstats
import io
from collections import deque
from datetime import datetime
from typing import Optional
import ex6

S = {"on": False, "tui": None, "frames": deque(maxlen=200_000), "cur": None, "orig": {},
     "started": 0.0, "proc0": 0.0, "hud": "", "hud_t": 0.0, "cpu_prof": None, "cpu_until": 0.0}
MAIN = threading.main_thread().ident


def _add(name, ms):
    cur = S["cur"]
    if cur is not None and threading.get_ident() == MAIN:
        cur[name] = cur.get(name, 0.0) + ms


def _timed(name, fn):
    def wrapper(*a, **kw):
        t = time.perf_counter()
        try: return fn(*a, **kw)
        finally: _add(name, (time.perf_counter() - t) * 1000)
    return wrapper


class _Out:
    def __init__(self, out): self.out = out
    def write(self, s):
        t = time.perf_counter()
        self.out.write(s)
        _add("write", (time.perf_counter() - t) * 1000)
        _add("bytes", len(s))
    def flush(self):
        t = time.perf_counter()
        self.out.flush()
        _add("write", (time.perf_counter() - t) * 1000)


def _loop(tui):
    if S["cpu_prof"] and time.perf_counter() > S["cpu_until"]:
        _finish_cpu()
    if not S["on"]:
        return _ORIG_LOOP(tui)
    S["cur"] = cur = {}
    t0, c0 = time.perf_counter(), time.thread_time()
    try:
        return _ORIG_LOOP(tui)
    finally:
        cur["frame"] = (time.perf_counter() - t0) * 1000
        cur["cpu"] = (time.thread_time() - c0) * 1000
        if "key_t" in cur:
            cur["key_latency"] = (cur.pop("flush_end", time.perf_counter()) - cur.pop("key_t")) * 1000
        cur.pop("flush_end", None)
        cur["t"] = t0
        S["frames"].append(cur)
        S["cur"] = None


def _inkey(orig):
    def inkey(*a, **kw):
        t = time.perf_counter()
        key = orig(*a, **kw)
        now = time.perf_counter()
        _add("wait", (now - t) * 1000)
        cur = S["cur"]
        if key and cur is not None:
            cur.setdefault("key_t", now)
            _add("keys", 1)
        return key
    return inkey


def _flush(orig):
    def flush(self, term, output=None):
        _draw_hud(self)
        t = time.perf_counter()
        orig(self, term, _Out(output or sys.__stdout__))
        now = time.perf_counter()
        _add("flush", (now - t) * 1000)
        if S["cur"] is not None: S["cur"]["flush_end"] = now
    return flush


def _draw_hud(buf):
    now = time.perf_counter()
    if now - S["hud_t"] > 0.5:
        S["hud_t"] = now
        recent = [f for f in S["frames"] if now - f["t"] < 1.0]
        if recent:
            work = sorted(f["frame"] - f.get("wait", 0) for f in recent)
            lat = [f["key_latency"] for f in recent if "key_latency" in f]
            cpu = sum(f["cpu"] for f in recent) / 10  # ms per 1s -> %
            S["hud"] = f" {len(recent)}fps work p50 {_pct(work, 50):.1f} max {work[-1]:.1f}ms cpu {cpu:.0f}%" + \
                       (f" key {max(lat):.1f}ms " if lat else " ")
    if S["hud"]:
        buf.puts(buf.w - len(S["hud"]), 0, S["hud"], txt_color=S["tui"].app.theme.accent)


def _pct(xs, p):
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))] if xs else 0.0


def _start(tui):
    app, term = tui.app, tui.term
    S["orig"] = {"flush": ex6.ScreenBuffer.flush, "clear": ex6.ScreenBuffer.clear,
                 "print_wrapped": ex6.ScreenBuffer.print_wrapped}
    ex6.ScreenBuffer.flush = _flush(S["orig"]["flush"])
    ex6.ScreenBuffer.clear = _timed("buf.clear", S["orig"]["clear"])
    ex6.ScreenBuffer.print_wrapped = _timed("print_wrapped", S["orig"]["print_wrapped"])
    orig_call, orig_hooks = app.call, app.run_context_hooks
    def call(name, *a, **kw):
        t = time.perf_counter()
        try: return orig_call(name, *a, **kw)
        finally: _add("call:" + name, (time.perf_counter() - t) * 1000)
    def run_context_hooks(kind, *a):
        t = time.perf_counter()
        try: return orig_hooks(kind, *a)
        finally: _add("hook:" + kind, (time.perf_counter() - t) * 1000)
    app.call, app.run_context_hooks = call, run_context_hooks
    term.inkey = _inkey(term.inkey)
    S.update(on=True, tui=tui, started=time.perf_counter(), proc0=time.process_time(), hud="")
    S["frames"].clear()


def _stop(tui):
    S["on"] = False
    ex6.ScreenBuffer.flush = S["orig"]["flush"]
    ex6.ScreenBuffer.clear = S["orig"]["clear"]
    ex6.ScreenBuffer.print_wrapped = S["orig"]["print_wrapped"]
    del tui.app.call, tui.app.run_context_hooks, tui.term.inkey
    return _report(tui)


def _report(tui):
    frames = list(S["frames"])
    wall = time.perf_counter() - S["started"]
    proc_cpu = time.process_time() - S["proc0"]
    ctx = tui.current
    lines = [f"ex6 frame profile  {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"wall {wall:.1f}s  frames {len(frames)}  fps {len(frames) / wall:.0f}  term {tui.term.width}x{tui.term.height}  mode {tui.mode}",
             f"ctx {ctx.name if ctx else None}  messages {len(ctx.get_messages()) if ctx else 0}",
             f"main-thread cpu {sum(f['cpu'] for f in frames) / 10 / wall:.0f}%   process cpu (all threads) {proc_cpu * 100 / wall:.0f}%",
             f"frames writing 0 bytes: {sum(1 for f in frames if not f.get('bytes')) * 100 / max(1, len(frames)):.0f}%",
             ""]
    for f in frames:
        f["work"] = f["frame"] - f.get("wait", 0)
        f["stall"] = max(0.0, f["work"] - f["cpu"])  # off-cpu time while working: GIL waits / blocking IO
    keys = sorted({k for f in frames for k in f} - {"t", "keys"})
    order = ["frame", "wait", "work", "cpu", "stall", "key_latency", "flush", "write", "bytes", "buf.clear", "print_wrapped"]
    keys = [k for k in order if k in keys] + [k for k in keys if k not in order]
    lines.append(f"{'section':<40}{'n':>7}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}{'%wall':>7}")
    for k in keys:
        xs = sorted(f[k] for f in frames if k in f)
        if not xs: continue
        share = "" if k in ("bytes", "frame") else f"{sum(xs) / 10 / wall:6.1f}"
        lines.append(f"{k:<40}{len(xs):>7}{sum(xs) / len(xs):>9.2f}{_pct(xs, 50):>9.2f}{_pct(xs, 95):>9.2f}"
                     f"{_pct(xs, 99):>9.2f}{xs[-1]:>9.2f}{share:>7}")
    lines.append("(ms per frame, except bytes. 'work' = frame - wait. flush includes write.)")
    path = ex6.get_folder("profiler") / f"frames-{datetime.now():%Y%m%d-%H%M%S}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path, lines


def _finish_cpu():
    pr = S["cpu_prof"]
    pr.disable()
    S["cpu_prof"] = None
    out = io.StringIO()
    st = pstats.Stats(pr, stream=out)
    st.sort_stats("cumulative").print_stats(50)
    st.sort_stats("tottime").print_stats(50)
    path = ex6.get_folder("profiler") / f"cpu-{datetime.now():%Y%m%d-%H%M%S}.txt"
    path.write_text(out.getvalue(), encoding="utf-8")
    S["tui"].app.debug_print(f"[prof] cpu profile written: {path}")


@ex6.command
def prof(tui, mode: Optional[str], secs: Optional[int]):
    'Profile frames. /prof toggles timing+HUD (report on stop). /prof cpu [secs] runs cProfile.'
    if mode == "cpu":
        if S["cpu_prof"]: return
        S["tui"] = tui
        S["cpu_until"] = time.perf_counter() + (secs or 5)
        S["cpu_prof"] = cProfile.Profile()
        S["cpu_prof"].enable()
        tui.app.debug_print(f"[prof] cProfile running for {secs or 5}s")
        return
    if S["on"]:
        path, lines = _stop(tui)
        for line in lines: tui.app.debug_print(line)
        tui.app.debug_print(f"[prof] report written: {path}")
    else:
        _start(tui)


_ORIG_LOOP = ex6._tui_loop
ex6._tui_loop = _loop
