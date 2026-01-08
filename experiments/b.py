
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane, Label

class TabApp(App):
    """App with horizontal tabs."""
    
    BINDINGS = [("q", "quit", "Quit")]
    
    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("tab1"):
                yield Label("TAB1:\n\nContent for tab 1 goes here...")
            with TabPane("tab2"):
                yield Label("TAB2:\n\nContent for tab 2 goes here...")
            with TabPane("tab3"):
                yield Label("TAB3:\n\nContent for tab 3 goes here...")
        yield Footer()

if __name__ == "__main__":
    TabApp().run()

