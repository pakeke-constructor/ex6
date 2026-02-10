


'''

Basic tools for ex6.
includes:

- reading/writing/updating files
- reading function bodies
- reading class/func headers
- glob


'''

import ex6
import os
import glob as _glob
import importlib
import tree_sitter


LANG_MODULES = {
    '.py': 'tree_sitter_python', '.pyw': 'tree_sitter_python',
    '.js': 'tree_sitter_javascript', '.mjs': 'tree_sitter_javascript',
    '.ts': 'tree_sitter_typescript', '.tsx': 'tree_sitter_typescript',
    '.jsx': 'tree_sitter_javascript',
    '.go': 'tree_sitter_go',
    '.rs': 'tree_sitter_rust',
    '.c': 'tree_sitter_c', '.h': 'tree_sitter_c',
    '.cpp': 'tree_sitter_cpp', '.hpp': 'tree_sitter_cpp', '.cc': 'tree_sitter_cpp',
    '.java': 'tree_sitter_java',
    '.rb': 'tree_sitter_ruby',
    '.cs': 'tree_sitter_c_sharp',
}

DEFINITION_TYPES = {
    'tree_sitter_python': ['function_definition', 'class_definition'],
    'tree_sitter_javascript': ['function_declaration', 'class_declaration', 'method_definition'],
    'tree_sitter_typescript': ['function_declaration', 'class_declaration', 'method_definition'],
    'tree_sitter_go': ['function_declaration', 'method_declaration', 'type_declaration'],
    'tree_sitter_rust': ['function_item', 'struct_item', 'enum_item', 'impl_item', 'trait_item'],
    'tree_sitter_c': ['function_definition', 'struct_specifier'],
    'tree_sitter_cpp': ['function_definition', 'class_specifier', 'struct_specifier'],
    'tree_sitter_java': ['method_declaration', 'class_declaration', 'interface_declaration'],
    'tree_sitter_ruby': ['method', 'class', 'module'],
    'tree_sitter_c_sharp': ['method_declaration', 'class_declaration', 'interface_declaration'],
}


def _parse_file(file):
    ext = os.path.splitext(file)[1].lower()
    mod_name = LANG_MODULES.get(ext)
    if not mod_name:
        raise ValueError(f"Unsupported file type: {ext}")
    mod = importlib.import_module(mod_name)
    lang = tree_sitter.Language(mod.language())
    parser = tree_sitter.Parser(lang)
    with open(file, "rb") as f:
        source = f.read()
    return parser.parse(source), source, mod_name


def _get_name(node):
    """Get the name of a definition node."""
    n = node.child_by_field_name('name')
    return n.text.decode() if n else None


def _signature(node, source):
    """Extract signature (start of node to start of body), plus first docstring line."""
    body = node.child_by_field_name('body')
    if body:
        sig = source[node.start_byte:body.start_byte].decode().rstrip().rstrip(':')
    else:
        sig = source[node.start_byte:node.end_byte].decode().split('\n')[0]
    # append first docstring/comment if present
    if body and body.children:
        first = body.children[0]
        if first.type in ('expression_statement', 'comment'):
            child = first.children[0] if first.children else first
            if child.type == 'string' or child.type == 'comment':
                doc = child.text.decode().strip().strip('"\' \n')
                first_line = doc.split('\n')[0].strip()
                if first_line:
                    sig += f'  # {first_line}'
    return sig


def read_file(ctx: ex6.Context, file: str) -> str:
    "Reads a file and returns its contents."
    with open(file, "r") as f:
        return f.read()


def write_file(ctx: ex6.Context, file: str, content: str) -> str:
    """
    Writes content to a file.
    If the file exists, it is cleared.
    If the file doesn't exist, a new file is created.
    """
    d = os.path.dirname(file)
    if d: os.makedirs(d, exist_ok=True)
    with open(file, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {file}"


def update_file(ctx: ex6.Context, file: str, search: str, replace: str) -> str:
    """Updates a file by searching and replacing a string."""
    with open(file, "r") as f:
        content = f.read()
    if search not in content:
        return f"ERROR: search string not found in {file}"
    content = content.replace(search, replace, 1)
    with open(file, "w") as f:
        f.write(content)
    return f"Updated {file}"


def find_files(ctx: ex6.Context, pattern: str) -> str:
    """Glob for files matching a pattern. Returns newline-separated paths."""
    matches = _glob.glob(pattern, recursive=True)
    return "\n".join(matches) if matches else "No matches."


def read_headers(ctx: ex6.Context, file: str) -> str:
    """Read class/function signatures from a file (no bodies)."""
    tree, source, mod_name = _parse_file(file)
    def_types = DEFINITION_TYPES.get(mod_name, [])
    out = []

    def collect(node):
        for child in node.children:
            if child.type in def_types:
                out.append(_signature(child, source))
            collect(child)

    collect(tree.root_node)
    return "\n".join(out) if out else "No classes/functions found."


def read_function(ctx: ex6.Context, file: str, name: str) -> str:
    """Read a function or class body by name from a file."""
    tree, source, mod_name = _parse_file(file)
    def_types = DEFINITION_TYPES.get(mod_name, [])

    def find(node):
        for child in node.children:
            if child.type in def_types and _get_name(child) == name:
                return source[child.start_byte:child.end_byte].decode()
            result = find(child)
            if result:
                return result
        return None

    result = find(tree.root_node)
    return result if result else f"ERROR: '{name}' not found in {file}"

