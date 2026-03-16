import ex6

@ex6.command
def hello(name):
    """Say hello."""
    ex6.debug_print(f"Hello, {name}!")
