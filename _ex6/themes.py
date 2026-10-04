import ex6
import json, os, copy
import math
from typing import Optional


def oklch(lightness: float, chroma: float, hue: float) -> tuple[int, int, int]:
    angle = math.radians(hue)
    a = chroma * math.cos(angle)
    b = chroma * math.sin(angle)
    l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    linear = (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )
    rgb = []
    for channel in linear:
        channel = max(0.0, min(1.0, channel))
        if channel <= 0.0031308:
            channel *= 12.92
        else:
            channel = 1.055 * channel ** (1 / 2.4) - 0.055
        rgb.append(round(channel * 255))
    return rgb[0], rgb[1], rgb[2]


THEMES = {
    "default": ex6.Theme(),

    "green": ex6.Theme(
        name="green",
        text=oklch(0.92, 0.008, 85),
        muted=oklch(0.60, 0.01, 175),
        cot=oklch(0.66, 0.10, 190),
        accent=oklch(0.72, 0.21, 145),
        accent_alt=oklch(0.74, 0.12, 195),
        success=oklch(0.75, 0.21, 140),
        warning=oklch(0.80, 0.16, 105),
        error=oklch(0.65, 0.20, 29),
        running=oklch(0.73, 0.12, 185),
        invoking=oklch(0.78, 0.15, 100),
        selection=oklch(0.85, 0.17, 105),
        error_bg=oklch(0.22, 0.045, 25),
        diff_add_bg=oklch(0.22, 0.04, 150),
        diff_del_bg=oklch(0.20, 0.04, 25),
        md_bullet=oklch(0.76, 0.15, 100),
        md_code=oklch(0.76, 0.12, 190),
        md_link=oklch(0.72, 0.12, 200),
        md_italic=oklch(0.77, 0.15, 110),
        md_bold=oklch(0.98, 0.005, 85),
    ),

    "blue": ex6.Theme(
        name="blue",
        text=oklch(0.92, 0.008, 250),
        muted=oklch(0.60, 0.01, 285),
        cot=oklch(0.66, 0.12, 320),
        accent=oklch(0.65, 0.21, 260),
        accent_alt=oklch(0.70, 0.22, 328),
        success=oklch(0.74, 0.16, 165),
        warning=oklch(0.80, 0.16, 105),
        error=oklch(0.65, 0.20, 29),
        running=oklch(0.74, 0.12, 195),
        invoking=oklch(0.72, 0.21, 335),
        selection=oklch(0.82, 0.13, 195),
        error_bg=oklch(0.22, 0.045, 15),
        diff_add_bg=oklch(0.21, 0.04, 165),
        diff_del_bg=oklch(0.20, 0.04, 15),
        md_bullet=oklch(0.71, 0.21, 330),
        md_code=oklch(0.76, 0.12, 190),
        md_link=oklch(0.70, 0.18, 250),
        md_italic=oklch(0.73, 0.20, 320),
        md_bold=oklch(0.98, 0.005, 250),
    ),

    "red": ex6.Theme(
        name="red",
        text=oklch(0.92, 0.008, 65),
        muted=oklch(0.60, 0.01, 335),
        cot=oklch(0.66, 0.12, 330),
        accent=oklch(0.66, 0.23, 29),
        accent_alt=oklch(0.79, 0.16, 105),
        success=oklch(0.74, 0.18, 150),
        warning=oklch(0.80, 0.16, 105),
        error=oklch(0.68, 0.23, 20),
        running=oklch(0.71, 0.21, 335),
        invoking=oklch(0.78, 0.15, 100),
        selection=oklch(0.85, 0.17, 105),
        error_bg=oklch(0.24, 0.05, 25),
        diff_add_bg=oklch(0.21, 0.04, 150),
        diff_del_bg=oklch(0.22, 0.045, 20),
        md_bullet=oklch(0.76, 0.15, 100),
        md_code=oklch(0.78, 0.16, 105),
        md_link=oklch(0.70, 0.21, 335),
        md_italic=oklch(0.73, 0.20, 325),
        md_bold=oklch(0.98, 0.005, 65),
    ),
}

def _load_saved_theme(app):
    try:
        data = json.loads((ex6.get_folder() / "theme.json").read_text())
        name = data.get("name", "")
        if name in THEMES:
            app.theme = copy.copy(THEMES[name])
    except: pass

def setup(app):
    _load_saved_theme(app)

@ex6.command
def theme(tui, name: Optional[str]):
    """Switch theme. No arg lists available themes."""
    if not name:
        lines = ["Themes:"] + [f"  {n}" for n in THEMES.keys()]
        scroll = [0]
        def draw(buf, inpt, r):
            x, y, w, h = r
            th = tui.app.theme
            buf.fill(r, ' ')
            buf.rect_line(r, txt_color=th.accent)
            if inpt.consume('KEY_UP') and scroll[0] > 0: scroll[0] -= 1
            if inpt.consume('KEY_DOWN'): scroll[0] += 1
            visible = h - 2
            max_scroll = max(0, len(lines) - visible)
            if scroll[0] > max_scroll: scroll[0] = max_scroll
            for i, line in enumerate(lines[scroll[0]:scroll[0] + visible]):
                buf.puts(x + 2, y + 1 + i, line[:w - 4], txt_color=th.text)
        tui.ui_panel_stack.append(draw)
        return
    if name not in THEMES:
        tui.app.debug_print(f"Unknown theme: {name}")
        return
    tui.app.theme = copy.copy(THEMES[name])
    path = ex6.get_folder() / "theme.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"name": name}))
    os.replace(tmp, path)


