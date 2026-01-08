
import questionary
from rich.console import Console
from rich.panel import Panel

console = Console()

tabs = ["tab1", "tab2", "tab3"]
current_tab = 0

def display_tab(tab_index):
    console.clear()
    
    # Display tab headers
    header = ""
    for i, tab in enumerate(tabs):
        if i == tab_index:
            header += f"[bold cyan][{tab}][/bold cyan]  "
        else:
            header += f"{tab}  "
    
    console.print(header)
    console.print("=" * 40)
    
    # Display tab content
    content = f"\n{tabs[tab_index].upper()}:\n\nContent for {tabs[tab_index]} goes here...\n"
    console.print(content)

while True:
    display_tab(current_tab)
    
    action = questionary.select(
        "",
        choices=[
            "← Previous tab",
            "→ Next tab",
            "Select specific tab",
            "Exit"
        ],
        use_shortcuts=True
    ).ask()
    
    if action == "← Previous tab":
        current_tab = (current_tab - 1) % len(tabs)
    elif action == "→ Next tab":
        current_tab = (current_tab + 1) % len(tabs)
    elif action == "Select specific tab":
        tab = questionary.select("Choose tab:", choices=tabs).ask()
        current_tab = tabs.index(tab)
    elif action == "Exit":
        break


