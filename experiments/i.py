
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
    _lock: threading.Lock = None # type: ignore
    
    def __post_init__(self):
        self._lock = threading.Lock()
    
    def __enter__(self):
        self._lock.acquire()
        return self
    
    def __exit__(self, *args):
        self._lock.release()

state = AppState()
tabs = ["CHAT", "FILES", "LOGS"]


def input_thread():
    """Independent thread to handle blocking keypresses."""
    with state as s:
        running = s.running
    
    while running:
        key = readchar.readkey()
        
        with state as s:
            if key == readchar.key.CTRL_C:
                s.running = False
            elif key == readchar.key.LEFT:
                s.active_tab = (s.active_tab - 1) % len(tabs)
            elif key == readchar.key.RIGHT:
                s.active_tab = (s.active_tab + 1) % len(tabs)
            elif key == readchar.key.BACKSPACE:
                s.input_buffer = s.input_buffer[:-1]
            elif key == readchar.key.ENTER:
                s.llm_output += f"\nUser: {s.input_buffer}"
                s.input_buffer = ""
            elif len(key) == 1:
                s.input_buffer += key
            else:
                print(key)
            
            running = s.running


def make_layout():
    """The 'Render' function."""
    with state as s:
        active_tab = tabs[s.active_tab]
        llm_output = s.llm_output
        input_buffer = s.input_buffer
    
    layout = Layout()
    layout.split_column(
        Layout(Panel(f"Tabs: {active_tab}", style="blue"), size=3),
        Layout(Panel(llm_output, title="Output")),
        Layout(Panel(f"> {input_buffer}█", title="Input")),
    )
    return layout


# --- Main Game Loop ---
threading.Thread(target=input_thread, daemon=True).start()

with Live(make_layout(), screen=True, auto_refresh=False) as live:
    with state as s:
        running = s.running
    
    while running:
        # Simulate LLM Streaming
        if time.time() % 1 > 0.95:
            with state as s:
                s.llm_output += " ."

        # Draw the frame
        live.update(make_layout(), refresh=True)
        
        with state as s:
            running = s.running
        
        time.sleep(0.05)