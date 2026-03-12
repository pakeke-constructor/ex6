


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
import re
import difflib
import glob as _glob
import importlib
import tree_sitter
import time
import fnmatch


def _load_gitignore():
    patterns = []
    if os.path.isfile(".gitignore"):
        with open(".gitignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line.rstrip("/"))
    return patterns

_GITIGNORE_PATTERNS = _load_gitignore()

def _is_gitignored(path):
    rel = os.path.relpath(path).replace("\\", "/")
    parts = rel.split("/")
    for pat in _GITIGNORE_PATTERNS:
        # match against basename or any path component or full relative path
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(os.path.basename(rel), pat):
            return True
        if any(fnmatch.fnmatch(p, pat) for p in parts):
            return True
    return False

def _check_gitignore(path):
    if _is_gitignored(path):
        raise ValueError(f"Refused: '{path}' is gitignored.")

def _check_read(ctx, path):
    if not ctx.has_read_file(path):
        raise ValueError(f"Must read file '{path}' before editing it.")


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
    '.lua': 'tree_sitter_lua',
    '.kt': 'tree_sitter_kotlin', '.kts': 'tree_sitter_kotlin',
}

CONTAINER_TYPES = {
    'class_definition', 'class_declaration', 'class_specifier',
    'interface_declaration',
    'struct_specifier', 'struct_item',
    'enum_item', 'impl_item', 'trait_item',
    'type_declaration',
    'module',
    'object_declaration', 'companion_object',
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
    'tree_sitter_lua': ['function_declaration', 'variable_declaration', 'assignment_statement'],
    'tree_sitter_kotlin': ['function_declaration', 'class_declaration', 'object_declaration', 'companion_object'],
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


def _signature_generic(node, source):
    body = node.child_by_field_name('body')
    if body:
        return source[node.start_byte:body.start_byte].decode().rstrip().rstrip(':')
    return source[node.start_byte:node.end_byte].decode().split('\n')[0]


def _signature_python(node, source):
    sig = _signature_generic(node, source)
    body = node.child_by_field_name('body')
    if body and body.children:
        first = body.children[0]
        if first.type in ('expression_statement', 'comment'):
            child = first.children[0] if first.children else first
            if child.type in ('string', 'comment'):
                doc = child.text.decode().strip().strip('"\' \n')
                first_line = doc.split('\n')[0].strip()
                if first_line:
                    sig += f'  # {first_line}'
    return sig


def _signature_lua(node, source):
    sig = source[node.start_byte:node.end_byte].decode().split('\n')[0]
    annotations = []
    sib = node.prev_sibling
    while sib and sib.type == 'comment' and sib.text.decode().startswith('---'):
        annotations.append(sib.text.decode())
        sib = sib.prev_sibling
    if annotations:
        annotations.reverse()
        return '\n'.join(annotations) + '\n' + sig
    return sig


def _signature_kotlin(node, source):
    for child in node.children:
        if child.type in ('function_body', 'class_body', 'enum_class_body'):
            return source[node.start_byte:child.start_byte].decode().rstrip()
    return source[node.start_byte:node.end_byte].decode().split('\n')[0]


_SIGNATURE_FNS = {
    'tree_sitter_python': _signature_python,
    'tree_sitter_lua': _signature_lua,
    'tree_sitter_kotlin': _signature_kotlin,
}


def _signature(node, source, mod_name):
    return _SIGNATURE_FNS.get(mod_name, _signature_generic)(node, source)



def write_file(ctx: ex6.Context, file: str, content: str) -> str:
    """Write content to a file, creating it if needed. Existing files must be read first."""
    if os.path.exists(file):
        _check_read(ctx, file)
        with open(file, "r") as f:
            old = f.read()
    else:
        old = ""
    diff = _make_diff(old, content)
    denial = approve(ctx, f"Write file: {file}", render_extra=lambda buf, x, y, w, h: _render_diff(buf, diff, x, y, w, h))
    if denial: raise ValueError(f"Denied: {denial}")
    d = os.path.dirname(file)
    if d: os.makedirs(d, exist_ok=True)
    with open(file, "w") as f:
        f.write(content)
    ctx.mark_file_read(file)
    return f"Wrote {len(content)} chars to {file}"




def _ws_normalize(s):
    return re.sub(r'[ \t]+', ' ', s).strip()

def edit_file(ctx: ex6.Context, file: str, search: str, replace: str) -> str:
    """Edit a file by searching and replacing a unique string.
    Tries exact match, then whitespace-insensitive, then fuzzy (80% threshold).
    search must match exactly one location. errors if zero or multiple matches.

    Use this tool when you need surgical edits, ESPECIALLY edits to 1-3 lines.
    Prefer edit_file_lines for larger edits or insertions where you know the line numbers.
    Prefer write_file if the entire file needs to be rewritten, or if the file is small (less than 50 lines)
    """
    _check_read(ctx, file)

    with open(file, "r") as f:
        content = f.read()

    def do_edit(original):
        new_content = content.replace(original, replace, 1)
        diff = _make_diff(original, replace)
        denial = approve(ctx, f"Edit file: {file}", render_extra=lambda buf, x, y, w, h: _render_diff(buf, diff, x, y, w, h))
        if denial: raise ValueError(f"Denied: {denial}")
        with open(file, "w") as f:
            f.write(new_content)
        ctx.mark_file_read(file)
        return f"Updated {file}"

    # 1. exact match
    if search in content:
        if content.count(search) > 1:
            raise ValueError(f"search string has {content.count(search)} matches in {file}; must be unique")
        return do_edit(search)
    search_lines = search.splitlines()
    content_lines = content.splitlines()
    n = len(search_lines)

    # 2. whitespace-insensitive match
    search_ws = [_ws_normalize(l) for l in search_lines]
    ws_matches = []
    for i in range(len(content_lines) - n + 1):
        if [_ws_normalize(l) for l in content_lines[i:i + n]] == search_ws:
            ws_matches.append(i)
    if len(ws_matches) == 1:
        original = "\n".join(content_lines[ws_matches[0]:ws_matches[0] + n])
        return do_edit(original)
    if len(ws_matches) > 1:
        raise ValueError(f"{len(ws_matches)} whitespace-normalized matches in {file}; must be unique")

    # 3. fuzzy line-level match
    THRESHOLD = 0.8
    matches = []
    for i in range(len(content_lines) - n + 1):
        chunk = content_lines[i:i + n]
        ratio = difflib.SequenceMatcher(None, search_lines, chunk).ratio()
        if ratio >= THRESHOLD:
            matches.append((ratio, i))
    if len(matches) == 1:
        ratio, start = matches[0]
        original = "\n".join(content_lines[start:start + n])
        return do_edit(original)
    if len(matches) > 1:
        raise ValueError(f"{len(matches)} fuzzy matches in {file}; must be unique. Add more context to disambiguate.")
    best = max((difflib.SequenceMatcher(None, search_lines, content_lines[i:i+n]).ratio()
                for i in range(len(content_lines) - n + 1)), default=0)
    raise ValueError(f"search string not found in {file} (best match: {best:.0%})")



def edit_file_lines(ctx: ex6.Context, file: str, start: int, end: int, content: str) -> str:
    """
    Replace lines start..end (inclusive, 1-indexed) with content.
    Prefer this over edit_file if you know the line numbers and are editing more than 2 lines,
    or if you want to insert code between function blocks/definitions.
    To insert without removing lines, set end=0 and start=the line to insert before.
    Content should NOT end with a trailing newline — one is added automatically.
    WARNING: Do NOT call this twice in a row; line numbers shift after the first edit. Use edit_file for subsequent edits, or re-read the file headers first.
    """
    _check_read(ctx, file)
    with open(file, "r") as f:
        lines = f.readlines()
    if end == 0:
        if start < 1 or start > len(lines) + 1:
            raise ValueError(f"Invalid insert position {start} (file has {len(lines)} lines)")
        new_lines = lines[:]
        new_lines.insert(start - 1, content + '\n')
    else:
        if start < 1 or end > len(lines) or start > end:
            raise ValueError(f"Invalid range {start}..{end} (file has {len(lines)} lines)")
        new_lines = lines[:]
        new_lines[start - 1:end] = [content + '\n']
    old_text = "".join(lines)
    new_text = "".join(new_lines)
    diff = _make_diff(old_text, new_text)
    denial = approve(ctx, f"Edit file: {file} (lines {start}-{end})", render_extra=lambda buf, x, y, w, h: _render_diff(buf, diff, x, y, w, h))
    if denial: raise ValueError(f"Denied: {denial}")
    with open(file, "w") as f:
        f.writelines(new_lines)
    ctx.mark_file_read(file)
    return f"Edited {file}"


def glob(ctx: ex6.Context, pattern: str) -> str:
    """Find files matching a glob pattern (recursive). Returns newline-separated paths."""
    matches = _glob.glob(pattern, recursive=True)
    return "\n".join(matches) if matches else "No matches."



_SKIP_DIRS = set(['.git', 'node_modules', '__pycache__', '.venv', 'venv', '.tox', '.mypy_cache', '.pytest_cache', 'dist', 'build', '.egg-info'])


def search(ctx: ex6.Context, pattern: str, match: str = "**/*", max_results: int = 15, line_numbers: bool = True) -> str:
    """Search file contents for a regex pattern, filtered by glob.
    Returns matching lines with file:line: prefix.
    Use line_numbers=False when you only care about the content of matches, not their location.
    When you just want to check whether a pattern exists, (e.g. after a refactor) use max_results=1 to save context.
    """
    regex = re.compile(pattern)
    matched_files = _glob.glob(match, recursive=True)
    results = []
    for f in matched_files:
        if not os.path.isfile(f):
            continue
        parts = f.replace("\\", "/").split("/")
        if any(p in _SKIP_DIRS for p in parts):
            continue
        try:
            with open(f, "r", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if regex.search(line):
                        prefix = f"{f}:{i}: " if line_numbers else ""
                        results.append(f"{prefix}{line.rstrip()}")
                        if len(results) >= max_results:
                            return "\n".join(results) + f"\n... (capped at {max_results} results)"
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(results) if results else "No matches."




def _read_headers_lua(tree, source):
    def_types = DEFINITION_TYPES['tree_sitter_lua']
    out = []

    def collect(node):
        for child in node.children:
            if child.type in def_types:
                if out:
                    out.append("")
                out.append(_signature_lua(child, source))
                if child.type not in ('variable_declaration', 'assignment_statement'):
                    collect(child)
            else:
                collect(child)

    collect(tree.root_node)
    return "\n".join(out) if out else "No classes/functions found."




def _add_line_numbers(text: str, start: int = 1) -> str:
    lines = text.split('\n')
    w = len(str(start + len(lines) - 1))
    return "\n".join(f"{i:>{w}}: {line}" for i, line in enumerate(lines, start))

def read_file(ctx: ex6.Context, path: str, line_numbers: bool = False) -> str:
    """
    Read and return contents of a file at the given path.
    - Prefer line_numbers=False to avoid bloat. 
    - Use line_numbers=True if you are doing deep work with this file.
    - It's okay to use this tool liberally if the files are small (e.g less than 100 lines)
    """
    time.sleep(3)
    with open(path, "r") as f:
        content = f.read()
    ctx.mark_file_read(path)
    if line_numbers:
        return _add_line_numbers(content)
    return content


def read_headers(ctx: ex6.Context, file: str, line_numbers: bool = False) -> str:
    """
    Read class/function signatures from a file (no bodies).
    Prefer line_numbers=True if you need to reference specific lines, or edit the file after.

    You should prefer using this tool first before reading an entire file.
    read_headers is more context-efficient, so unless you are very sure you need the entire file, use this.
    """
    ctx.mark_file_read(file)
    tree, source, mod_name = _parse_file(file)
    if mod_name == 'tree_sitter_lua':
        result = _read_headers_lua(tree, source)
        if line_numbers:
            return _add_line_numbers(result)
        return result
    def_types = DEFINITION_TYPES.get(mod_name, [])
    out = []

    def collect(node, indent=0):
        prefix = "  " * indent
        for child in node.children:
            if child.type in def_types:
                if indent == 0 and out:
                    out.append("")  # gap between top-level defs
                line_no = source[:child.start_byte].count(b'\n') + 1
                sig = _signature(child, source, mod_name).strip()
                if line_numbers:
                    out.append(f"{line_no}: {prefix}{sig}")
                else:
                    out.append(prefix + sig)
                if child.type in CONTAINER_TYPES:
                    collect(child, indent + 1)
            else:
                collect(child, indent)

    collect(tree.root_node)
    return "\n".join(out) if out else "No classes/functions found."


def web_search(ctx: ex6.Context, query: str) -> str:
    """Search the web. Returns top results as text."""
    import urllib.request, urllib.parse, json
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        html = r.read().decode()
    results = re.findall(r'class="result__a"[^>]*href="(.*?)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</span>', html, re.DOTALL)
    if not results:
        return "No results found."
    out = []
    for href, title, snippet in results[:8]:
        title = re.sub(r'<[^>]+>', '', title).strip()
        snippet = re.sub(r'<[^>]+>', '', snippet).strip()
        out.append(f"{title}\n  {href}\n  {snippet}")
    return "\n\n".join(out)


def read_function(ctx: ex6.Context, file: str, name: str, line_numbers: bool = False) -> str:
    """
    Read a function or class body by name from a file.
    - Prefer line_numbers=True if you want to edit the function, or refererence line-numbers to the user.
    - Use this tool when you only need bits of information, like details about a particular function
    """
    ctx.mark_file_read(file)
    tree, source, mod_name = _parse_file(file)
    def_types = DEFINITION_TYPES.get(mod_name, [])

    def find(node):
        for child in node.children:
            if child.type in def_types and _get_name(child) == name:
                start_line = source[:child.start_byte].count(b'\n') + 1
                text = source[child.start_byte:child.end_byte].decode()
                if line_numbers:
                    return _add_line_numbers(text, start_line)
                return text
            result = find(child)
            if result:
                return result
        return None

    result = find(tree.root_node)
    if not result:
        raise ValueError(f"'{name}' not found in {file}")
    return result





def ask_user(ctx: ex6.Context, question: str) -> str:
    """Ask user a question and wait for their response. Blocks until answered."""
    result = [None]

    def on_submit(text):
        result[0] = text
        ctx.input_stack.pop()

    input_draw = ex6.make_input(on_submit)

    def draw(buf: ex6.ScreenBuffer, inpt, r):
        x, y, w, h = r
        buf.puts(x, y, f"? {question}", txt_color='yellow')
        input_draw(buf, inpt, (x + 2, y + 1, w - 2, 1))

    ctx.push_ui(draw)

    while draw in ctx.input_stack:
        time.sleep(0.05)

    return result[0] or ""


def _diff_color(line):
    if line.startswith('+'): return 'green'
    if line.startswith('-'): return 'red'
    if line.startswith('@@'): return 'cyan'
    return 'white'

def _make_diff(old: str, new: str) -> list:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    lines = list(difflib.unified_diff(old_lines, new_lines, lineterm=''))
    # strip the --- +++ header (first 2 lines)
    return lines[2:] if len(lines) > 2 else lines

def _render_diff(buf, diff_lines, x, y, w, h):
    """Render diff lines into a region. Returns rows used."""
    max_lines = h
    truncated = len(diff_lines) > max_lines
    visible = diff_lines[:max_lines - 1] if truncated else diff_lines
    for i, line in enumerate(visible):
        color = _diff_color(line)
        buf.puts(x, y + i, line[:w], txt_color=color, bg_color='bright_black')
    if truncated:
        remainder = len(diff_lines) - len(visible)
        buf.puts(x, y + len(visible), f"... {remainder} more lines", txt_color='bright_black', bg_color='bright_black')
    return len(visible) + (1 if truncated else 0)


def approve(ctx: ex6.Context, description: str, render_extra=None) -> str | None:
    """Show approval dialog. ENTER=approve (returns None), text+ENTER=deny (returns reason).
    render_extra: optional fn(buf, x, y, w, h) called below the chrome to render extra info."""
    result = [False, None]  # [answered, denial_reason]

    def on_submit(text):
        result[0] = True
        result[1] = text if text.strip() else None
        ctx.input_stack.pop()

    input_draw = ex6.make_input(on_submit)

    def draw(buf: ex6.ScreenBuffer, inpt, r):
        x, y, w, h = r
        buf.fill(r, char=' ', bg_color='bright_black')
        buf.rect_line(r, txt_color='cyan', bg_color='bright_black')
        cx = x + 3
        cy = y + 1
        buf.puts(cx, cy,   description, txt_color='cyan', bg_color='bright_black')
        buf.puts(cx, cy+1, "ENTER approve | type reason + ENTER to deny", txt_color='white', bg_color='bright_black')
        if inpt.consume('KEY_ENTER'):
            result[0] = True
            ctx.input_stack.pop()
            return
        input_draw(buf, inpt, (cx, cy + 2, w - 6, 1))
        if render_extra:
            extra_y = cy + 4
            extra_h = h - extra_y + y - 1
            if extra_h > 0:
                render_extra(buf, cx, extra_y, w - 6, extra_h)

    ctx.push_ui(draw)

    while draw in ctx.input_stack:
        time.sleep(0.05)

    return result[1]



def _get_claude_md_content(ctx):
    for p in ["CLAUDE.md", ".claude/CLAUDE.md"]:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    return "(no CLAUDE.md found)"

CLAUDE_MD = ex6.Message(role="system", content=_get_claude_md_content)

