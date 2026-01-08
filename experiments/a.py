from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.syntax import Syntax
import time

console = Console()

# 1. Basic styled output
console.print("[bold blue]User:[/bold blue] Hello Claude!")

# 2. Render markdown (like Claude's responses)
markdown = Markdown("## Response\n\nHere's some **bold** text and `code`.")
console.print(markdown)

# 3. Syntax highlighted code
code = '''def hello():
    print("Hello world")'''
syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
console.print(Panel(syntax, title="Python Code"))

# 4. Streaming text (like LLM responses)
console.print("\n[bold green]Assistant:[/bold green]", end=" ")
with Live("", console=console, refresh_per_second=20) as live:
    response = "This is a simulated streaming response from Claude..."
    accumulated = ""
    for char in response:
        accumulated += char
        live.update(accumulated)
        time.sleep(0.05)  # Simulate streaming

console.print("\n")

# 5. Panels for structure
console.print(Panel(
    "This could be a system message or file content",
    title="📄 File: main.py",
    border_style="cyan"
))



import questionary

# With default selection
answer = questionary.select(
    "Choose an option:",
    choices=["option 1", "option 2", "option 3"],
    default="option 2",
    use_jk_keys=True
).ask()

# Or use Choice objects for more control
from questionary import Choice

answer = questionary.select(
    "What would you like to do?",
    choices=[
        Choice("Create new file", value="create"),
        Choice("Edit existing", value="edit"),
        Choice("Delete", value="delete", disabled="Not implemented yet")
    ],
    use_jk_keys=True
).ask()


