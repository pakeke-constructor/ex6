
import os
import sys
import readchar
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

console = Console()

# --- 1. State ---
tabs = ["CHAT", "FILES", "LOGS"]
active_tab_index = 0
input_buffer = ""
message_history = []

def render_ui():
    """The Render Function: Creates the UI based on current state."""
    # Build Tab Header
    header = Text()
    for i, name in enumerate(tabs):
        style = "bold white on blue" if i == active_tab_index else "dim"
        header.append(f"  {name}  ", style=style)
        header.append(" ")

    # Build Content Body
    body_text = f"Active Tab: {tabs[active_tab_index]}\n"
    body_text += "\n".join(message_history[-10:]) # Show last 10 lines
    
    # Assemble Layout
    layout = Layout()
    layout.split_column(
        Layout(Panel(header), size=3),
        Layout(Panel(body_text, title="Output")),
        Layout(Panel(f"> {input_buffer}█", title="Input (Left/Right to switch tabs, Enter to send)")),
    )
    return layout

# --- 2. The Loop ---
def run_app():
    global active_tab_index, input_buffer
    
    # We use Rich's Live to handle the 'clear and re-print' efficiently
    with Live(render_ui(), auto_refresh=False, screen=True) as live:
        while True:
            live.update(render_ui(), refresh=True)
            
            # Read a single key/sequence
            key = readchar.readkey()
            
            # Handle Navigation (Escape sequences)
            if key == readchar.key.LEFT:
                active_tab_index = (active_tab_index - 1) % len(tabs)
            elif key == readchar.key.RIGHT:
                active_tab_index = (active_tab_index + 1) % len(tabs)
            
            # Handle Text Input
            elif key == readchar.key.ENTER:
                if input_buffer.strip().lower() == "exit":
                    break
                message_history.append(f"[{tabs[active_tab_index]}] {input_buffer}")
                input_buffer = ""
            elif key == readchar.key.BACKSPACE:
                input_buffer = input_buffer[:-1]
            elif key == readchar.key.CTRL_C:
                break
            elif len(key) == 1: # Standard character typing
                input_buffer += key

if __name__ == "__main__":
    run_app()