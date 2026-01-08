
from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window, VSplit
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.widgets import TextArea


tabs = [" CHAT ", " FILES ", " LOGS "]
active_tab_index = 0


def get_tab_tokens():
    """Generates the text for the tab bar based on the active index."""
    result = []
    for i, name in enumerate(tabs):
        if i == active_tab_index:
            result.append(("class:tab.active", name))
        else:
            result.append(("class:tab", name))
    return result

# Top bar for tabs
header_window = Window(content=FormattedTextControl(get_tab_tokens), height=1)

# Main content area (where your output or chat would go)
body_control = FormattedTextControl("Press Left/Right to switch tabs. Press Ctrl-C to exit.")
body_window = Window(content=body_control, style="class:body")

# Input area (the command line)
input_field = TextArea(height=1, prompt=">>> ", multiline=False)



kb = KeyBindings()

@kb.add("left")
def _(event):
    global active_tab_index
    active_tab_index = (active_tab_index - 1) % len(tabs)
    body_control.text = f"You are now viewing: {tabs[active_tab_index]}"

@kb.add("right")
def _(event):
    global active_tab_index
    active_tab_index = (active_tab_index + 1) % len(tabs)
    body_control.text = f"You are now viewing: {tabs[active_tab_index]}"

@kb.add("c-c")
def _(event):
    event.app.exit()


# stack the Header, Body, and Input vertically
root_container = HSplit([
    header_window,
    Window(height=1, char="-", style="class:line"), # A separator line
    body_window,
    input_field,
])

layout = Layout(root_container, focused_element=input_field)

app = Application(layout=layout, key_bindings=kb, full_screen=True)
app.run()