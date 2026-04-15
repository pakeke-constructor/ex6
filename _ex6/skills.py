

"""

skills:
Progressive disclosure, allows LLMs to read/load skills.

Structure of skill md file:
<structure>
# ui
short description

---

(^^^ this serves as a separator, anything below this line is part of the "body")
long description, examples
</structure>

"""



# TODO: implement this by looping over _ex6/skills/*.md
# should look like:
skill_list_str = """
ui: short description
other_skill: blah blah description
"""



def load_skill(ctx, skill_id):
    f"""
    load skills with load_skill(id).
    List of skills:
    {skill_list_str}
    """

    # TODO loads skill!
    # injects markdown of skill into ctx window.
    pass





