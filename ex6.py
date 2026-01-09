
import threading
import time
from dataclasses import dataclass
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
import readchar


@dataclass
class AppState:
    '''
    can put ANY kind of objects in here.
    lists, dicts, anything.
    
    We will need custom objects in here in future.
    '''
    active_tab: int = 0
    input_buffer: str = "hello i am [orange]colored text!!![/orange]! yay."
    llm_output: str = "Streaming starts here..."
    running: bool = True


class ThreadSafeState:
    def __init__(self, state):
        self._state = state
        self._lock = threading.Lock()
    def __enter__(self):
        self._lock.acquire()
        return self._state
    def __exit__(self, *args):
        self._lock.release()



state = ThreadSafeState(AppState())

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
            
            running = s.running


def render():
    with state as s:
        active_tab = tabs[s.active_tab]
        llm_output = s.llm_output
        input_buffer = s.input_buffer
    
    layout = Layout()
    layout.split_column(
        Layout(Panel(f"Tabs: {active_tab}", style="blue"), size=3),
        Layout(Panel(llm_output, title="Output", style="red")),
        Layout(Panel(f"> {input_buffer}█", title="[blue bold]Input[/blue bold]")),
    )
    return layout



threading.Thread(target=input_thread, daemon=True).start()



# main "render" loop:
with Live(render(), screen=True, auto_refresh=False) as live:
    with state as s:
        running = s.running
    
    while running:
        # Simulate LLM Streaming
        if time.time() % 1 > 0.98:
            with state as s:
                s.llm_output += " ."

        # Draw the frame
        live.update(render(), refresh=True)
        
        with state as s:
            running = s.running
        
        time.sleep(0.01)


