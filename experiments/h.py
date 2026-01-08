
import threading
import time
from dataclasses import dataclass
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
import readchar

@dataclass
class AppState:
    active_tab: int = 0
    input_buffer: str = ""
    llm_output: str = "Streaming starts here..."
    running: bool = True

state = AppState()
tabs = ["CHAT", "FILES", "LOGS"]

def input_thread():
    """Independent thread to handle blocking keypresses."""
    while state.running:
        key = readchar.readkey()
        
        if key == readchar.key.CTRL_C:
            state.running = False
        elif key == readchar.key.LEFT:
            state.active_tab = (state.active_tab - 1) % len(tabs)
        elif key == readchar.key.RIGHT:
            state.active_tab = (state.active_tab + 1) % len(tabs)
        elif key == readchar.key.BACKSPACE:
            state.input_buffer = state.input_buffer[:-1]
        elif key == readchar.key.ENTER:
            # Process command...
            state.llm_output += f"\nUser: {state.input_buffer}"
            state.input_buffer = ""
        elif len(key) == 1:
            state.input_buffer += key

def make_layout():
    """The 'Render' function."""
    layout = Layout()
    layout.split_column(
        Layout(Panel(f"Tabs: {tabs[state.active_tab]}", style="blue"), size=3),
        Layout(Panel(state.llm_output, title="Output")),
        Layout(Panel(f"> {state.input_buffer}█", title="Input")),
    )
    return layout

# --- Main Game Loop ---
threading.Thread(target=input_thread, daemon=True).start()

with Live(make_layout(), screen=True, auto_refresh=False) as live:
    while state.running:
        # Simulate LLM Streaming (the 'Update' phase)
        if time.time() % 1 > 0.95: # Just a dummy ticker
             state.llm_output += " ."

        # Draw the frame
        live.update(make_layout(), refresh=True)
        time.sleep(0.05) # 20 FPS is plenty for a CLI


