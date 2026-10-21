from _ex6.agents_sound import make_sound_agent


def setup(app):
    app.tui.add_context(make_sound_agent(
        app,
        project="a demon/skull pixel-art hack-and-slash game.",
        sfx_dir="assets/sfx",
        credits_file="CREDITS.txt",
    ))
