import ex6
from typing import Optional

THEMES = {
    "default": ex6.Theme(),

    "green": ex6.Theme(
        name="green",
        text="bright_white",
        muted="bright_black",
        accent="green",
        accent_alt="cyan",
        success="bright_green",
        warning="yellow",
        error="bright_red",
        running="cyan",
        invoking="bright_yellow",
        selection="green",
        error_bg=(10, 60, 10),
        diff_add_bg=(18, 60, 18),
        diff_del_bg=(60, 18, 18),
        md_bullet="cyan",
        md_code="white",
        md_link="green",
        md_italic="yellow",
        md_bold="bright_white",
    ),

    "red": ex6.Theme(
        name="red",
        text="bright_white",
        muted="bright_black",
        accent="red",
        accent_alt="bright_red",
        success="bright_green",
        warning="yellow",
        error="bright_red",
        running="bright_red",
        invoking="bright_yellow",
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


@ex6.command
def theme(name: Optional[str]):
    """Switch theme. No arg lists available themes."""
    if not name:
        ex6.debug_print("Themes:", ", ".join(THEMES.keys()))
        return
    if name not in THEMES:
        ex6.debug_print(f"Unknown theme: {name}")
        return
    ex6.state.theme = THEMES[name]
