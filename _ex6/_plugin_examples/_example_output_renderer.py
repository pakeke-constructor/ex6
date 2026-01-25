

import time, math
import ex6


def render_spinner(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
    txt = "spinner! " + ("\\|/—"[math.floor(time.time()*5) % 4])
    buf.puts(x, y, txt, txt_color='red')
    lines_used = 1
    return lines_used


@ex6.output_renderer
def example_renderer(output: list[ex6.OutputLine], ctx: ex6.Context) -> None:
    # Replace lines containing "SPINNER" with a red spinner
    for i, line in enumerate(output):
        if isinstance(line, str) and "SPINNER" in line:
            # if line contains `SPINNER`, replace line with a spinner!
            output[i] = render_spinner

    # we can do other stuff too:

    # Delete empty lines
    # output[:] = [l for l in output if l != '']

    # Insert custom header
    # output.insert(0, "=== OUTPUT ===")

