from ex6 import hook, command, console


@hook.on_startup
def init():
    console.add("Chat plugin loaded.\n")


@command("clear")
def clear_cmd():
    console.clear()
    return True


@command("echo", args=[("text", str)])
def echo_cmd(text):
    console.add(f"Echo: {text}\n")
    return True


@hook.on_submit
def handle_input(text, state):
    console.add(f"User: {text}\n")
    state.current_window.append({"role": "user", "content": text})
    console.add("AI: ")
    state.llm.send(text)
    return True


@hook.on_tick
def stream_response(state):
    if state.llm.has_new_tokens():
        console.add(state.llm.consume_tokens())


@hook.on_render_status
def show_status(state):
    if state.llm.busy:
        return "[dim]Thinking...[/dim]"
    return None
