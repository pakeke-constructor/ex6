
"""
Simple Textual Feature Showcase

Install: pip install textual

Run: python app.py
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import (
    Header, Footer, Static, Input, Label,
    RadioButton, RadioSet, TabbedContent, TabPane
)


class SimpleApp(App):
    """A minimal Textual application showcasing core features"""
    
    TITLE = "Simple Textual Showcase"
    
    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()
        
        with TabbedContent():
            # Tab 1: Input Form
            with TabPane("Form Input"):
                yield Label("Enter your name:")
                yield Input(placeholder="Type here and press Enter...", id="name-input")
                yield Static("", id="greeting")
                
            # Tab 2: Radio Selection
            with TabPane("Radio Select"):
                yield Label("Choose your favorite language:")
                with RadioSet(id="language-radio"):
                    yield RadioButton("Python", value=True)
                    yield RadioButton("JavaScript")
                    yield RadioButton("Rust")
                    yield RadioButton("Go")
                yield Static("", id="selection-result")
            
            # Tab 3: Code Display
            with TabPane("Code Box"):
                yield Label("Example Python function:")
                code = """def greet(name): return f"Hello, {name}!"""
                yield Static(code, id="code-display")
                
            # Tab 4: Info Box
            with TabPane("Info"):
                info = """Press Q to quit"""
                yield Static(info)
        
        yield Footer()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission"""
        if event.input.id == "name-input":
            name = event.value.strip()
            greeting_widget = self.query_one("#greeting", Static)
            if name:
                greeting_widget.update(f"Hello, {name}! 👋")
            else:
                greeting_widget.update("Please enter a name.")
    
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle radio button changes"""
        if event.radio_set.id == "language-radio":
            result_widget = self.query_one("#selection-result", Static)
            selected = event.pressed.label
            result_widget.update(f"You selected: {selected} ✓")


if __name__ == "__main__":
    app = SimpleApp()
    app.run()