from ex6 import hook, command


@hook.on_startup
def init():
    from ex6 import state
    with state as s:
        s.console.append("Chat plugin loaded.")


@command("clear")
def clear_cmd():
    from ex6 import state
    with state as s:
        s.console.clear()
    return True


@command("echo", args=[("text", str)])
def echo_cmd(text):
    from ex6 import state
    with state as s:
        s.console.append(f"Echo: {text}")
    return True


@hook.on_submit
def handle_input(text, s):
    s.console.append(f"User: {text}")
    s.console.append(f"AI: You said '{text}'")  # dummy response
    return True
