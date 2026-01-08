"""
Textual Feature Showcase - Interactive Task Manager Dashboard

Install: pip install textual

Run: python app.py
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Button, Static, Input, Label, 
    DataTable, ProgressBar, Checkbox, RadioButton, RadioSet,
    Select, TabbedContent, TabPane, Tree, Log, Rule, Markdown
)
from textual.binding import Binding
from textual.reactive import reactive
from datetime import datetime


class TaskStats(Static):
    """Reactive statistics display"""
    total = reactive(0)
    completed = reactive(0)
    
    def watch_total(self, total: int) -> None:
        self.update(self.render_stats())
    
    def watch_completed(self, completed: int) -> None:
        self.update(self.render_stats())
    
    def render_stats(self) -> str:
        pct = (self.completed / self.total * 100) if self.total > 0 else 0
        return f"📊 Tasks: {self.total} | ✅ Done: {self.completed} | 📈 {pct:.1f}%"


class TaskManagerApp(App):
    """A feature-rich Textual application showcasing various components"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #sidebar {
        width: 30;
        background: $panel;
        border-right: solid $primary;
    }
    
    #main-content {
        width: 1fr;
    }
    
    .box {
        height: auto;
        border: solid $primary;
        margin: 1;
        padding: 1;
    }
    
    TaskStats {
        background: $accent;
        color: $text;
        padding: 1;
        text-align: center;
        text-style: bold;
    }
    
    Button {
        margin: 0 1;
    }
    
    ProgressBar {
        margin: 1 0;
    }
    
    Input {
        margin: 1 0;
    }
    
    DataTable {
        height: 15;
    }
    
    Tree {
        height: 12;
    }
    
    Log {
        height: 10;
        border: solid $primary;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "toggle_dark", "Toggle Dark Mode", show=True),
        Binding("a", "add_task", "Add Task", show=True),
    ]
    
    TITLE = "Textual Feature Showcase - Task Manager"
    
    def __init__(self):
        super().__init__()
        self.task_counter = 0
        self.tasks = []
    
    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header(show_clock=True)
        
        with Horizontal():
            # Sidebar with Tree navigation
            with Vertical(id="sidebar"):
                yield Static("🌳 Navigation", classes="box")
                tree = Tree("📁 Projects")
                tree.root.expand()
                work = tree.root.add("💼 Work", expand=True)
                work.add_leaf("📝 Documentation")
                work.add_leaf("🐛 Bug Fixes")
                personal = tree.root.add("🏠 Personal", expand=True)
                personal.add_leaf("🛒 Shopping")
                personal.add_leaf("📚 Learning")
                yield tree
                
                yield Static("⚙️ Settings", classes="box")
                with RadioSet():
                    yield RadioButton("High Priority")
                    yield RadioButton("Medium Priority", value=True)
                    yield RadioButton("Low Priority")
            
            # Main content area
            with ScrollableContainer(id="main-content"):
                yield TaskStats()
                
                # Tabbed interface
                with TabbedContent():
                    # Task Management Tab
                    with TabPane("Tasks", id="tasks-tab"):
                        yield Static("➕ Add New Task", classes="box")
                        yield Input(placeholder="Enter task name...", id="task-input")
                        
                        with Horizontal():
                            yield Button("Add Task", variant="primary", id="add-btn")
                            yield Button("Clear Completed", variant="warning", id="clear-btn")
                            yield Button("Delete Selected", variant="error", id="delete-btn")
                        
                        yield Rule()
                        
                        table = DataTable(cursor_type="row")
                        table.add_columns("✓", "ID", "Task", "Status", "Priority", "Created")
                        yield table
                        
                        yield Label("Task Progress")
                        yield ProgressBar(total=100, show_eta=False, id="progress")
                    
                    # Activity Log Tab
                    with TabPane("Activity Log", id="log-tab"):
                        yield Static("📜 Recent Activity", classes="box")
                        log = Log(highlight=True)
                        log.write_line("Application started")
                        log.write_line("Ready to manage tasks...")
                        yield log
                    
                    # Help Tab
                    with TabPane("Help", id="help-tab"):
                        markdown_content = """
# Task Manager Help

## Features Demonstrated

- **Header & Footer**: Navigation and key bindings
- **Reactive State**: Live updating statistics
- **DataTable**: Sortable task list with selection
- **Tree Widget**: Hierarchical project navigation
- **Tabs**: Organized content sections
- **Forms**: Input fields and buttons
- **Radio Buttons**: Priority selection
- **Progress Bar**: Visual task completion
- **Log Widget**: Activity tracking
- **CSS Styling**: Custom theming
- **Key Bindings**: Keyboard shortcuts

## Keyboard Shortcuts

- `a` - Add new task
- `d` - Toggle dark/light mode
- `q` - Quit application
- `↑/↓` - Navigate table rows
- `Enter` - Select/toggle row

## Try It Out!

Add some tasks and watch the statistics update in real-time!
                        """
                        yield Markdown(markdown_content)
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Set up the app after mounting"""
        self.table = self.query_one(DataTable)
        self.progress = self.query_one("#progress", ProgressBar)
        self.log_widget = self.query_one(Log)
        self.stats = self.query_one(TaskStats)
        self.update_stats()
        self.log_widget.write_line(f"[bold green]App mounted at {datetime.now().strftime('%H:%M:%S')}[/]")
    
    def action_toggle_dark(self) -> None:
        """Toggle dark mode"""
        self.dark = not self.dark
        mode = "dark" if self.dark else "light"
        self.log_widget.write_line(f"Switched to {mode} mode")
    
    def action_add_task(self) -> None:
        """Focus the task input field"""
        self.query_one("#task-input", Input).focus()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks"""
        if event.button.id == "add-btn":
            self.add_task()
        elif event.button.id == "clear-btn":
            self.clear_completed()
        elif event.button.id == "delete-btn":
            self.delete_selected()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key in input field"""
        if event.input.id == "task-input":
            self.add_task()
    
    def add_task(self) -> None:
        """Add a new task to the table"""
        task_input = self.query_one("#task-input", Input)
        task_name = task_input.value.strip()
        
        if not task_name:
            self.log_widget.write_line("[yellow]⚠️ Task name cannot be empty[/]")
            return
        
        self.task_counter += 1
        task_id = f"T{self.task_counter:03d}"
        created = datetime.now().strftime("%H:%M:%S")
        
        # Get selected priority
        radio_set = self.query_one(RadioSet)
        priority = radio_set.pressed_button.label.plain if radio_set.pressed_button else "Medium"
        
        self.table.add_row("⬜", task_id, task_name, "Pending", priority, created)
        self.tasks.append({"id": task_id, "completed": False})
        
        task_input.value = ""
        self.log_widget.write_line(f"[green]✅ Added task: {task_name}[/]")
        self.update_stats()
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Toggle task completion when row is selected"""
        row_key = event.row_key
        current_check = self.table.get_cell(row_key, "✓")
        
        if current_check == "⬜":
            self.table.update_cell(row_key, "✓", "✅")
            self.table.update_cell(row_key, "Status", "Done")
            # Find and update task
            for task in self.tasks:
                if self.table.get_cell(row_key, "ID") == task["id"]:
                    task["completed"] = True
                    break
            self.log_widget.write_line(f"[green]✅ Completed task {self.table.get_cell(row_key, 'ID')}[/]")
        else:
            self.table.update_cell(row_key, "✓", "⬜")
            self.table.update_cell(row_key, "Status", "Pending")
            for task in self.tasks:
                if self.table.get_cell(row_key, "ID") == task["id"]:
                    task["completed"] = False
                    break
            self.log_widget.write_line(f"[yellow]↩️ Reopened task {self.table.get_cell(row_key, 'ID')}[/]")
        
        self.update_stats()
    
    def clear_completed(self) -> None:
        """Remove all completed tasks"""
        rows_to_remove = []
        for row_key in self.table.rows:
            if self.table.get_cell(row_key, "✓") == "✅":
                rows_to_remove.append(row_key)
        
        for row_key in rows_to_remove:
            task_id = self.table.get_cell(row_key, "ID")
            self.table.remove_row(row_key)
            self.tasks = [t for t in self.tasks if t["id"] != task_id]
        
        if rows_to_remove:
            self.log_widget.write_line(f"[blue]🗑️ Cleared {len(rows_to_remove)} completed tasks[/]")
        else:
            self.log_widget.write_line("[yellow]ℹ️ No completed tasks to clear[/]")
        
        self.update_stats()
    
    def delete_selected(self) -> None:
        """Delete the currently selected task"""
        if self.table.cursor_row is not None:
            row_key = self.table.coordinate_to_cell_key(self.table.cursor_coordinate).row_key
            task_id = self.table.get_cell(row_key, "ID")
            self.table.remove_row(row_key)
            self.tasks = [t for t in self.tasks if t["id"] != task_id]
            self.log_widget.write_line(f"[red]🗑️ Deleted task {task_id}[/]")
            self.update_stats()
        else:
            self.log_widget.write_line("[yellow]⚠️ No task selected[/]")
    
    def update_stats(self) -> None:
        """Update statistics and progress bar"""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t["completed"])
        
        self.stats.total = total
        self.stats.completed = completed
        
        if total > 0:
            self.progress.update(progress=(completed / total) * 100)
        else:
            self.progress.update(progress=0)


if __name__ == "__main__":
    app = TaskManagerApp()
    app.run()