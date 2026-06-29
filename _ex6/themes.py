import ex6
import json, os
from typing import Optional

# colors:
# https://blessed.readthedocs.io/en/latest/colors.html

THEMES = {
    "default": ex6.Theme(),

    "green": ex6.Theme(
        name="green",
        text = "white",
        muted="bright_black",
        cot = "red",
        accent="seagreen",
        accent_alt="darkturquoise",
        success="mediumseagreen",
        warning="darkkhaki",
        error="indianred",
        running="teal",
        invoking="cadetblue",
        selection="mediumaquamarine",
        error_bg=(15, 45, 20),
        diff_add_bg=(10, 50, 30),
        diff_del_bg=(50, 18, 25),
        md_bullet="darkturquoise",
        md_code="darkseagreen",
        md_link="seagreen",
        md_italic="cadetblue",
        md_bold="bright_white",
    ),

    "blue": ex6.Theme(
        name="blue",
        text = "white",
        muted="bright_black",
        cot = "blue",
        accent="bright_blue",
        accent_alt="mediumpurple",
        success="cornflowerblue",
        warning="plum",
        error="hotpink",
        running="deepskyblue",
        invoking="mediumslateblue",
        selection="bright_cyan",
        error_bg=(40, 10, 60),
        diff_add_bg=(10, 20, 60),
        diff_del_bg=(50, 12, 40),
        md_bullet="deepskyblue",
        md_code="steelblue",
        md_link="cornflowerblue",
        md_italic="orchid",
        md_bold="bright_white",
    ),

    "red": ex6.Theme(
        name="red",
        text = "white",
        muted="bright_black",
        cot = "orange",
        accent="red",
        accent_alt="bright_red",
        success="blue",
        warning="yellow",
        error="bright_red",
        running="bright_red",
        invoking="orange",
        selection="bright_red",
        error_bg=(100, 10, 10),
        diff_add_bg=(18, 60, 18),
        diff_del_bg=(80, 10, 10),
        md_bullet="bright_red",
        md_code="bright_white",
        md_link="red",
        md_italic="bright_red",
        md_bold="bright_white",
    ),
}

def _load_saved_theme():
    try:
        data = json.loads((ex6.get_folder() / "theme.json").read_text())
        name = data.get("name", "")
        if name in THEMES:
            ex6.set_theme(THEMES[name])
    except: pass

_load_saved_theme()

@ex6.command
def theme(name: Optional[str]):
    """Switch theme. No arg lists available themes."""
    if not name:
        lines = ["Themes:"] + [f"  {n}" for n in THEMES.keys()]
        scroll = [0]
        def draw(buf, inpt, r):
            x, y, w, h = r
            th = ex6.get_theme()
            buf.fill(r, ' ')
            buf.rect_line(r, txt_color=th.accent)
            if inpt.consume('KEY_UP') and scroll[0] > 0: scroll[0] -= 1
            if inpt.consume('KEY_DOWN'): scroll[0] += 1
            visible = h - 2
            max_scroll = max(0, len(lines) - visible)
            if scroll[0] > max_scroll: scroll[0] = max_scroll
            for i, line in enumerate(lines[scroll[0]:scroll[0] + visible]):
                buf.puts(x + 2, y + 1 + i, line[:w - 4], txt_color=th.text)
        ex6.push_ui_panel(draw)
        return
    if name not in THEMES:
        ex6.debug_print(f"Unknown theme: {name}")
        return
    ex6.set_theme(THEMES[name])
    path = ex6.get_folder() / "theme.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"name": name}))
    os.replace(tmp, path)


