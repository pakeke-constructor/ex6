from blessed import Terminal

term = Terminal()

print(f"Terminal supports {term.number_of_colors} colors\n")

for color in range(term.number_of_colors):
    print(f"{term.color(color)} {color:3} {term.normal}", end="")
    if (color + 1) % 16 == 0:
        print()

print()
