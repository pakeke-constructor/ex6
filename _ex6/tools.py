

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
import json
import difflib
import glob as _glob
import importlib
import tree_sitter
import time
import fnmatch
import threading
import subprocess
import shutil
import datetime
import platform
import sys
import git
import inspect
import functools
from _ex6.models import M
from ex6 import Context, Message
from typing import Optional



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

_file_locks = {}
_file_locks_lock = threading.Lock()

def _get_file_lock(path):
    key = os.path.normpath(os.path.abspath(path))
    with _file_locks_lock:
        if key not in _file_locks:
            _file_locks[key] = threading.Lock()
        return _file_locks[key]

def _is_gitignored(path):
    rel = os.path.relpath(path).replace("\\", "/")
    parts = rel.split("/")
    if any(p in _SKIP_DIRS for p in parts):
        return True
    for pat in _GITIGNORE_PATTERNS:
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
    if not n:
        n = node.child_by_field_name('left')
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
    if node.type != 'function_declaration' and node.end_point[0] > node.start_point[0]:
        sig = sig.rstrip() + ' ...'
    sb, _ = _node_range_lua(node, source)
    if sb < node.start_byte:
        annotations = source[sb:node.start_byte].decode().rstrip('\n')
        return annotations + '\n' + sig
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


def _node_range_lua(node, source):
    """Return (start_byte, end_byte) including preceding ---@ annotations."""
    start_byte = node.start_byte
    sib = node.prev_sibling
    while sib and sib.type == 'comment' and sib.text.decode().startswith('---'):
        start_byte = sib.start_byte
        sib = sib.prev_sibling
    return start_byte, node.end_byte

def _node_range_default(node, source):
    return node.start_byte, node.end_byte

_NODE_RANGE_FNS = {
    'tree_sitter_lua': _node_range_lua,
}

def _node_range(node, source, mod_name):
    return _NODE_RANGE_FNS.get(mod_name, _node_range_default)(node, source)


def _signature(node, source, mod_name):
    return _SIGNATURE_FNS.get(mod_name, _signature_generic)(node, source)



def write_file(ctx: ex6.Context, file: str, content: str) -> str:
    """Write content to a file, creating it if needed. Existing files must be read first."""
    p = ctx.resolve(file)
    with _get_file_lock(p):
        if os.path.exists(p):
            _check_read(ctx, file)
            with open(p, "r") as f:
                old = f.read()
        else:
            old = ""
        diff = _make_diff(old, content)
        denial = approve(ctx, f"Write file: {file}", render_extra=lambda buf, x, y, w, h: _render_diff(buf, diff, x, y, w, h, file))
        if denial: raise ValueError(f"User denied your write-file request, with reason: {denial}")
        d = os.path.dirname(p)
        if d: os.makedirs(d, exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
        ctx.mark_file_read(file)
        return f"Wrote {len(content)} chars to {file}"




def _ws_normalize(s):
    return re.sub(r'[ \t]+', ' ', s).strip()


def edit_file(ctx: ex6.Context, file: str, search: str, replace: str) -> str:
    """Edit a file by searching and replacing a unique string.
    Tries exact match, then whitespace-insensitive, then fuzzy (80% threshold).
    search must match exactly one location. errors if zero or multiple matches.

    Usage:
    - Use this tool when you need surgical edits, ESPECIALLY edits to 1-2 lines.
    - ALWAYS Prefer edit_file_lines for edits larger than 3 lines, but only when you know the lines. If you need to delete a lot of code, edit_file_lines is better because it avoids you rewriting the entire code.
    - ALWAYS Prefer write_file if the entire file needs to be rewritten, or if the file is small (less than 50 lines)
    """
    return edit_file_codemode(ctx, file, search, replace)


def edit_file_codemode(ctx: ex6.Context, file: str, search: str, replace: str) -> str:
    """Edit a file by searching and replacing a unique string.
    Tries exact match, then whitespace-insensitive, then fuzzy (80% threshold).
    search must match exactly one location. errors if zero or multiple matches.

    Usage:
    - Use this tool when you need surgical edits, ESPECIALLY edits to 1-2 lines.
    - ALWAYS Prefer edit_file_lines for edits larger than 3 lines. but only when you know the lines. If you need to delete a lot of code; edit_file_lines is better because it avoids you rewriting the entire code.
    - ALWAYS Prefer write_file if the entire file needs to be rewritten, or if the file is small (less than 50 lines)

    Argument Formatting:
    - For multiline editing, you MUST use raw triple-backtick strings, (using r'''). Otherwise, \\n sequences will wreck the strings, and you will find it hard to code.
    - Do NOT use random \\ characters to escape ' or " characters. Python allows you to use ' or " characters in r''' strings without escaping.
    - For multiline edits, you MUST format over multiple lines. DO NOT use a string like r'foo\\nbar\\nbaz'.

    Correct usage:
    edit_file("file.txt",
    r'''search''',
    r'''replace''')
    """
    _check_read(ctx, file)
    p = ctx.resolve(file)

    with _get_file_lock(p):
        with open(p, "r") as f:
            content = f.read()

        def do_edit(original):
            new_content = content.replace(original, replace, 1)
            diff = _make_diff(original, replace)
            denial = approve(ctx, f"Edit file: {file}", render_extra=lambda buf, x, y, w, h: _render_diff(buf, diff, x, y, w, h, file))
            if denial: raise ValueError(f"User denied your edit_file request, with reason: {denial}")
            with open(p, "w") as f:
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
    To delete lines, pass content="" (empty string).
    A trailing newline is added automatically after your edit.

    Prefer this over edit_file for deleting large code blocks, or inserting between definitions.
    ALWAYS Prefer edit_file_lines for edits larger than 3 lines. but only when you know the lines. If you need to delete a lot of code; edit_file_lines is better because it avoids you rewriting the entire code.

    If you have not read the lines or headers, your edits will automatically be rejected.
    Use this in conjunction with read_headers and/or read_body to replace/rewrite entire functions or classes.
    """
    _check_read(ctx, file)
    p = ctx.resolve(file)

    with _get_file_lock(p):
        with open(p, "r") as f:
            lines = f.readlines()

        # snapshot check
        snapshot = ctx.get_line_snapshot(file)
        def assert_line(L):
            actual = lines[L - 1].rstrip('\n') if L <= len(lines) else ""
            if actual == "":
                return
            if L not in snapshot:
                raise ValueError(f"Line {L} not in any snapshot for {file}, re-read the file first.")
            expected = snapshot[L]
            if expected != actual:
                raise ValueError(f"Line {L} has shifted since last read, re-read the file.")

        if end == 0:
            if start < 1 or start > len(lines) + 1:
                raise ValueError(f"Invalid insert position {start} (file has {len(lines)} lines)")
            assert_line(start)
            new_lines = lines[:]
            new_lines.insert(start - 1, content + '\n')
        else:
            if start < 1 or end > len(lines) or start > end:
                raise ValueError(f"Invalid range {start}..{end} (file has {len(lines)} lines)")
            assert_line(start)
            assert_line(end)
            new_lines = lines[:]
            if content == "":
                new_lines[start - 1:end] = []
            else:
                new_lines[start - 1:end] = [content + '\n']

        old_text = "".join(lines)
        new_text = "".join(new_lines)
        diff = _make_diff(old_text, new_text)
        denial = approve(ctx, f"Edit file: {file} (lines {start}-{end})", render_extra=lambda buf, x, y, w, h: _render_diff(buf, diff, x, y, w, h, file))
        if denial: raise ValueError(f"The user denied your edit request, with reason: {denial}")
        with open(p, "w") as f:
            f.writelines(new_lines)
        ctx.mark_file_read(file)
        snapshot = ctx.get_line_snapshot(file)
        for line_no in list(snapshot):
            if line_no > start:
                del snapshot[line_no]
        return f"Edited {file}"

def glob(ctx: ex6.Context, pattern: str) -> str:
    """Find files matching a glob pattern (recursive). Returns newline-separated paths."""
    root = ctx.cwd or os.getcwd()
    matches = [m for m in _glob.glob(pattern, recursive=True, root_dir=root) if not _is_gitignored(m)]
    return "\n".join(matches) if matches else "No matches."



_SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', '.tox', '.mypy_cache', '.pytest_cache', 'dist', 'build', '.egg-info'}


def search(ctx: ex6.Context, pattern: str, file_glob: str = "**/*", max_results: int = 15, page: int = 1) -> str:
    """Search file contents for a regex pattern, filtered by glob.
    Returns matching lines with file:line: prefix.
    Pagination: page is 1-indexed and returns at most max_results matches for that page.
    When you just want to check whether a pattern exists, (e.g. after a refactor) use max_results=1 to save context.
    - Make sure to use regex patterns correctly. WRONG: search("func("), malformed regex pattern. CORRECT: search("func\\(")
    """
    if page < 1:
        raise ValueError("page must be >= 1")
    if max_results < 1:
        raise ValueError("max_results must be >= 1")

    regex = re.compile(pattern)
    root = ctx.cwd or os.getcwd()
    start_idx = (page - 1) * max_results
    end_idx = start_idx + max_results
    seen = 0
    results = []

    for f in _glob.glob(file_glob, recursive=True, root_dir=root):
        if _is_gitignored(f):
            continue
        fp = os.path.join(root, f)
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, "r", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if not regex.search(line):
                        continue
                    if seen < start_idx:
                        seen += 1
                        continue
                    if seen >= end_idx:
                        return "\n".join(results) + f"\n... (page {page}, max_results {max_results})"
                    prefix = f"{f}:{i}: "
                    results.append(f"{prefix}{line.rstrip()}")
                    seen += 1
        except (OSError, UnicodeDecodeError):
            continue

    return "\n".join(results) if results else "No matches."



def _read_headers_lua(tree, source, line_numbers=False):
    def_types = DEFINITION_TYPES['tree_sitter_lua']
    out = []
    sig_line_nos = []

    for child in tree.root_node.children:
        if child.type not in def_types:
            continue
        if out:
            out.append("")
        sb, _ = _node_range_lua(child, source)
        line_no = source[:sb].count(b'\n') + 1
        sig_line_nos.append(line_no)
        sig = _signature_lua(child, source)
        if line_numbers:
            sig_lines = sig.split('\n')
            for i, sl in enumerate(sig_lines):
                out.append(f"{line_no + i}: {sl}")
        else:
            out.append(sig)

    text = "\n".join(out) if out else "No classes/functions found."
    return text, sig_line_nos




def _add_line_numbers(text: str, start: int = 1) -> str:
    lines = text.split('\n')
    w = len(str(start + len(lines) - 1))
    return "\n".join(f"{i:>{w}}: {line}" for i, line in enumerate(lines, start))

def read_file(ctx: ex6.Context, path: str, line_numbers: bool = False, lines: tuple[int,int] = None) -> str:
    """
    Read and return contents of a file at the given path.
    - Prefer line_numbers=False to avoid bloat. 
    - Use line_numbers=True if you are doing deep work with this file.
    - It's okay to use this tool liberally if the files are small (e.g less than 100 lines)
    - lines=(start,end) to read a subset (1-indexed, inclusive). (Forces line_numbers=True)
    """
    _check_gitignore(path)
    p = ctx.resolve(path)
    with open(p, "r") as f:
        all_lines = f.readlines()
    if lines:
        start, end = lines
        end = min(end, len(all_lines)) # end-line can't go beyond file
        selected = all_lines[start-1:end]
        ctx.mark_file_read(path, list(range(start, end + 1)))
        return _add_line_numbers("".join(selected), start=start)
    content = "".join(all_lines)
    if ex6.get_token_estimate(content) > ex6.MAX_TOOL_OUTPUT_CHARACTERS:
        # short-circuit for code-mode, so the LLM can still see other tool-calls in this block.
        raise ValueError("File is too big!")
    ctx.mark_file_read(path, list(range(1, len(all_lines) + 1)))
    if line_numbers:
        return _add_line_numbers(content)
    return content


def read_headers(ctx: ex6.Context, file: str, line_numbers: bool = True) -> str:
    """
    Read class/function signatures from a file (no bodies).
    Prefer line_numbers=True if you need to reference specific lines, or edit the file after.

    You should prefer using this tool first before reading an entire file.
    read_headers is more context-efficient, so unless you are very sure you need the entire file, use this.
    """
    _check_gitignore(file)
    p = ctx.resolve(file)
    tree, source, mod_name = _parse_file(p)
    if mod_name == 'tree_sitter_lua':
        result, sig_line_nos = _read_headers_lua(tree, source, line_numbers)
        ctx.mark_file_read(file, sig_line_nos)
        return result
    def_types = DEFINITION_TYPES.get(mod_name, [])
    out = []
    sig_line_nos = []

    def collect(node, indent=0):
        prefix = "  " * indent
        for child in node.children:
            if child.type in def_types:
                if indent == 0 and out:
                    out.append("")  # gap between top-level defs
                line_no = source[:child.start_byte].count(b'\n') + 1
                sig_line_nos.append(line_no)
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
    ctx.mark_file_read(file, sig_line_nos)
    return "\n".join(out) if out else "No classes/functions found."


def read_body(ctx: ex6.Context, file: str, name: str, line_numbers: bool = True) -> str:
    """
    Read a function or class body by name from a file.
    - Prefer line_numbers=True if you want to edit the function, or refererence line-numbers to the user.
    - Use this tool when you only need bits of information, like details about a particular function
    """
    p = ctx.resolve(file)
    tree, source, mod_name = _parse_file(p)
    def_types = set(DEFINITION_TYPES.get(mod_name, []))

    def find(node, target):
        for child in node.children:
            if child.type in def_types and _get_name(child) == target:
                return child
            result = find(child, target)
            if result:
                return result
        return None

    # exact match on full name (handles Lua's "module.func" style)
    hit = find(tree.root_node, name)

    # dotted nested lookup (handles Python's Class.method style)
    if not hit and '.' in name:
        parts = name.split('.')
        node = tree.root_node
        for part in parts:
            node = find(node, part)
            if not node:
                break
        hit = node

    # permissive fallback: try leaf name directly
    if not hit and '.' in name:
        hit = find(tree.root_node, name.rsplit('.', 1)[-1])

    if not hit:
        raise ValueError(f"'{name}' not found in {file}")

    sb, eb = _node_range(hit, source, mod_name)
    start_line = source[:sb].count(b'\n') + 1
    end_line = source[:eb].count(b'\n') + 1
    text = source[sb:eb].decode()
    if line_numbers:
        all_lines = source.decode().splitlines()
        s = max(start_line - 1, 1)
        e = min(end_line + 1, len(all_lines))
        ctx.mark_file_read(file, list(range(s, e + 1)))
        chunk = "\n".join(all_lines[s-1:e])
        return _add_line_numbers(chunk, s)
    ctx.mark_file_read(file, list(range(start_line, end_line + 1)))
    return text





def ask_user(ctx: ex6.Context, question: str) -> str:
    """Ask user a question and wait for their response. Blocks until answered."""
    result = [None]

    def on_submit(text):
        result[0] = text
        ctx.ui_stack.pop()

    input_draw = ex6.make_input(on_submit)

    def draw(buf: ex6.ScreenBuffer, inpt, r):
        x, y, w, h = r
        th = ex6.get_theme()
        buf.puts(x, y, f"? {question}", txt_color=th.warning)
        input_draw(buf, inpt, (x + 2, y + 1, w - 2, 1))

    ctx.push_ui(draw)

    while draw in ctx.ui_stack:
        time.sleep(0.05)

    return result[0] or ""



def ask_user_question(ctx: ex6.Context, question: str, opt: Optional[list[str]] = None) -> str:
    """Ask user a question with optional selectable answers. Returns: Q: <question>\nA: <answer>"""
    options = [str(o) for o in (opt or [])]
    selected = [0]
    typed_mode = [len(options) == 0]
    result = [None]

    def on_submit(text):
        if not text:
            return
        result[0] = text
        ctx.ui_stack.pop()

    input_draw = ex6.make_input(on_submit)

    def _wrap_words(text: str, width: int) -> list[str]:
        if width < 1:
            return [""]
        out = []
        for part in (text or "").split("\n"):
            words = part.split()
            if not words:
                out.append("")
                continue
            line = ""
            for word in words:
                if not line:
                    line = word
                elif len(line) + 1 + len(word) <= width:
                    line += " " + word
                else:
                    out.append(line)
                    line = word
            if line:
                out.append(line)
        return out or [""]

    def draw(buf: ex6.ScreenBuffer, inpt, r):
        x, y, w, h = r
        th = ex6.get_theme()

        show_options = bool(options) and not typed_mode[0]
        q_lines = _wrap_words(question, max(1, w - 8))
        base_h = 6 + len(q_lines) + (len(options) + 2 if show_options else 3)
        box_h = min(h, max(h // 2, base_h))
        box = ex6.Region(x, y, w, box_h)

        buf.fill(box, char=' ', bg_color=None)
        buf.rect_line(box, txt_color=th.accent)

        cx = x + 3
        cy = y + 1
        buf.puts(cx, cy, "AGENT QUESTION", txt_color=th.accent_alt, style='bold')
        cy += 2

        for line in q_lines:
            if cy >= y + box_h - 2:
                break
            buf.puts(cx, cy, line[:max(1, w - 6)], txt_color=th.warning)
            cy += 1

        cy += 1

        if show_options:
            if inpt.consume('KEY_UP'):
                selected[0] = (selected[0] - 1) % len(options)
            if inpt.consume('KEY_DOWN'):
                selected[0] = (selected[0] + 1) % len(options)
            if inpt.consume('KEY_ENTER'):
                result[0] = options[selected[0]]
                ctx.ui_stack.pop()
                return
            if inpt._keys:
                typed_mode[0] = True
                show_options = False

        if show_options:
            buf.puts(cx, cy, "UP/DOWN choose | ENTER submit | type custom answer", txt_color=th.text)
            cy += 1
            for i, option in enumerate(options):
                if cy >= y + box_h - 1:
                    break
                is_sel = i == selected[0]
                prefix = "›" if is_sel else " "
                clr = th.success if is_sel else th.text
                buf.puts(cx, cy, f"{prefix} {option}"[:max(1, w - 6)], txt_color=clr, style='bold' if is_sel else None)
                cy += 1
            return

        buf.puts(cx, cy, "Type custom answer, ENTER submit", txt_color=th.text)
        cy += 1
        input_h = max(1, y + box_h - 1 - cy)
        input_draw(buf, inpt, (cx, cy, max(1, w - 6), input_h))
        if options and getattr(input_draw, 'text', '') == "":
            typed_mode[0] = False

    ctx.push_ui(draw)

    while draw in ctx.ui_stack:
        time.sleep(0.05)

    answer = result[0] or ""
    return f"Q: {question}\nA: {answer}"


class EscalationError(Exception):
    def __init__(self, reason, severity=1):
        self.reason = reason
        self.severity = severity
        super().__init__(reason)


def escalate(ctx: ex6.Context, reason: str, severity: int = 1) -> str:
    """Escalates an issue to the human operator, or to the parent agent.
    Use when: no simple solution exists, the task seems malformed, or you are unable to complete it.
    - Example: prompt asks to improve 
    severity: 1=informational, 2=blocking, 3=critical."""
    if ctx.parent:
        raise EscalationError(reason, severity)
    result = [None]

    def on_submit(text):
        result[0] = text
        ctx.ui_stack.pop()

    input_draw = ex6.make_input(on_submit)
    sev_labels = {1: "INFO", 2: "BLOCKING", 3: "CRITICAL"}
    label = sev_labels.get(severity, f"SEV-{severity}")

    def draw(buf: ex6.ScreenBuffer, inpt, r):
        x, y, w, h = r
        th = ex6.get_theme()
        buf.fill(r, char=' ', bg_color=None)
        buf.rect(r, txt_color=th.muted)
        cx = x + 3
        cy = y + 1
        buf.puts(cx, cy, f"[{label}] ESCALATION", txt_color=th.error, bg_color=None)
        words = reason.split()
        lines, line = [], ""
        for word in words:
            if line and len(line) + 1 + len(word) > w - 6:
                lines.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            lines.append(line)
        for i, l in enumerate(lines):
            if cy + 1 + i >= y + h - 2:
                break
            buf.puts(cx, cy + 1 + i, l, txt_color=th.warning, bg_color=None)
        prompt_y = cy + 1 + len(lines) + 1
        buf.puts(cx, prompt_y - 1, "Respond to agent:", txt_color=th.text, bg_color=None)
        input_draw(buf, inpt, (cx, prompt_y, w - 6, 1))

    ctx.push_ui(draw)

    while draw in ctx.ui_stack:
        time.sleep(0.05)

    return result[0] or ""


def _make_diff(old: str, new: str) -> list:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    lines = list(difflib.unified_diff(old_lines, new_lines, lineterm=''))
    # strip the --- +++ header (first 2 lines)
    lines = lines[2:] if len(lines) > 2 else lines
    return [l.replace('\n', ' ').replace('\r', '') for l in lines]


def _render_diff(buf, diff_lines, x, y, w, h, filename=None):
    """Render diff lines into a region, with optional syntax highlighting."""
    from _ex6.z_highlight_codeblock import render_highlighted_line
    try:
        from pygments.lexers import get_lexer_for_filename
        lexer = get_lexer_for_filename(filename) if filename else None
    except:
        lexer = False

    th = ex6.get_theme()
    max_lines = h
    truncated = len(diff_lines) > max_lines
    visible = diff_lines[:max_lines - 1] if truncated else diff_lines
    for i, line in enumerate(visible):
        if line.startswith('@@'):
            buf.puts(x, y + i, line[:w], txt_color=th.accent_alt); continue
        bg = th.diff_add_bg if line.startswith('+') else th.diff_del_bg if line.startswith('-') else None
        if lexer:
            render_highlighted_line(buf, x, y + i, w, line, lexer, bg_color=bg)
        else:
            buf.puts(x, y + i, line[:w], txt_color=th.text, bg_color=bg); continue
    if truncated:
        remainder = len(diff_lines) - len(visible)
        buf.puts(x, y + len(visible), f"... {remainder} more lines", txt_color=th.muted)
    return len(visible) + (1 if truncated else 0)


def approve(ctx: ex6.Context, description: str, render_extra=None, height=None, bottom=False) -> str | None:
    """Show approval dialog. ENTER=approve (returns None), text+ENTER=deny (returns reason).
    render_extra: optional fn(buf, x, y, w, h) called below the chrome to render extra info."""
    if ctx.yolo:
        return None
    result = [False, None]  # [answered, denial_reason]

    def on_submit(text):
        result[0] = True
        result[1] = text if text.strip() else None
        ctx.ui_stack.pop()

    input_draw = ex6.make_input(on_submit)

    def draw(buf: ex6.ScreenBuffer, inpt, r):
        th = ex6.get_theme()
        panel = ex6.Region(*r)
        if height is not None:
            panel_h = min(panel[3], height)
            panel_y = panel[1] + panel[3] - panel_h if bottom else panel[1]
            panel = ex6.Region(panel[0], panel_y, panel[2], panel_h)
        content = panel.shrink(3, 1)

        buf.fill(panel, char=' ', bg_color=None)
        buf.rect_line(panel, txt_color=th.accent)
        buf.puts(content[0], content[1], description[:content[2]], txt_color=th.accent_alt, style='bold')
        buf.puts(content[0], content[1] + 1, "ENTER approve | type reason + ENTER to deny"[:content[2]], txt_color=th.muted)
        if (not input_draw.get_text()) and inpt.consume('KEY_ENTER'):
            result[0] = True
            ctx.ui_stack.pop()
            return
        input_r = ex6.Region(content[0], content[1] + 2, content[2], 1)
        input_draw(buf, inpt, input_r, txt_color=th.warning)
        if render_extra:
            extra_r = ex6.Region(content[0], content[1] + 4, content[2], max(0, content[3] - 4))
            if extra_r[3] > 0:
                render_extra(buf, *extra_r)

    ctx.push_ui(draw)

    while draw in ctx.ui_stack:
        if ctx.stop_early:
            if draw in ctx.ui_stack: ctx.ui_stack.remove(draw)
            return "stopped"
        time.sleep(0.05)

    return result[1]



def _get_claude_md_content(ctx):
    root = ctx.cwd or os.getcwd()
    for p in ["CLAUDE.md", ".claude/CLAUDE.md", "AGENTS.md"]:
        fp = os.path.join(root, p)
        if os.path.isfile(fp):
            with open(fp, "r", encoding="utf-8") as f:
                return f.read()
    return "(no AGENTS.md or CLAUDE.md found)"

CLAUDE_MD = ex6.Message(role="system", content=_get_claude_md_content, overview="AGENTS.md")
AGENTS_MD = CLAUDE_MD



def _env_content(ctx):
    cwd = ctx.cwd or os.getcwd()
    plat = platform.system()
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL, cwd=cwd).strip()
    except Exception:
        branch = "unknown"
    return f"<environment>\n- cwd: {cwd}\n- platform: {plat}\n- date: {now}\n- git branch: {branch}\n</environment>"

ENV_PROMPT = ex6.Message(role="system", content=_env_content, overview="env")



_IS_WINDOWS = sys.platform == "win32"
_bash_location = None

def _get_bash():
    global _bash_location
    if _bash_location:
        return _bash_location
    if not _IS_WINDOWS:
        _bash_location = "bash"
        return _bash_location
    # Windows: try shutil.which first, then known install paths
    b = shutil.which("bash")
    if b:
        _bash_location = b
        return _bash_location
    git_path = shutil.which("git")
    if git_path:
        # git.exe lives in <install>/cmd/ or <install>/bin/ — walk up and check
        install_dir = os.path.dirname(os.path.dirname(git_path))
        for sub in ("bin", "usr\\bin"):
            p = os.path.join(install_dir, sub, "bash.exe")
            if os.path.isfile(p):
                _bash_location = p
                return _bash_location
    for d in (r"C:\Program Files\Git", r"C:\Program Files (x86)\Git"):
        for sub in ("bin", "usr\\bin"):
            p = os.path.join(d, sub, "bash.exe")
            if os.path.isfile(p):
                _bash_location = p
                return _bash_location
    return None



def make_safe_cwd(folders: dict[str, str]):
    if len(folders) < 2:
        raise ValueError("make_safe_cwd requires at least 2 folders")
    reverse = {os.path.normpath(v): k for k, v in folders.items()}

    def safe_cwd(ctx: ex6.Context, tag: str) -> str:
        if tag not in folders:
            raise ValueError(f"ERROR: unknown cwd tag '{tag}'. Valid tags: {', '.join(folders.keys())}")
        old_cwd = ctx.cwd or os.getcwd()
        old_key = reverse.get(os.path.normpath(old_cwd), old_cwd)
        new_path = os.path.normpath(folders[tag])
        ctx.cwd = new_path
        return f"{old_key} -> {tag} ({new_path})"

    folder_list = "\n".join(f"  {k} => {v}" for k, v in folders.items())
    safe_cwd.__doc__ = f"""\
Switch working directory to a named folder.

Available folders:
{folder_list}

Usage: safe_cwd("tag_name")"""
    safe_cwd.__name__ = "safe_cwd"

    return safe_cwd


def _approve_command(ctx: ex6.Context, shell: str, command: str) -> str | None:
    command_lines = command.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    def render_command(buf, x, y, w, h):
        th = ex6.get_theme()
        area = ex6.Region(x, y, w, h)
        label_r, command_r = area.split_vertical(1, max(2, len(command_lines) + 2))
        buf.puts(label_r[0], label_r[1], "COMMAND", txt_color=th.muted, style='bold')
        command_r = ex6.Region(command_r[0], command_r[1], command_r[2], min(command_r[3], len(command_lines) + 2))
        buf.rect_line(command_r, txt_color=th.warning)
        buf.print_contained('\n'.join(command_lines), command_r.shrink(2, 1), txt_color=th.warning, wrap=False)

    return approve(ctx, f"{shell.upper()} APPROVAL", render_extra=render_command, height=len(command_lines) + 9, bottom=True)


def bash(ctx: ex6.Context, command: str, timeout: int = 30) -> str:
    """Run a bash/shell command and return its output (stdout + stderr combined).
    Use for: running tests, checking git status, installing packages, etc.
    timeout: max seconds to wait (default 30)."""
    bp = _get_bash()
    if not bp:
        return "ERROR: bash not found (install Git for Windows)"
    denial = _approve_command(ctx, "bash", command)
    if denial: raise ValueError(f"User denied your bash request, with reason: {denial}")
    try:
        result = subprocess.run([bp, "-c", command], capture_output=True, text=True, timeout=timeout, cwd=ctx.cwd)
        out = result.stdout + result.stderr
        if result.returncode != 0:
            out = f"[exit code {result.returncode}]\n" + out
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"



def powershell(ctx: ex6.Context, command: str, timeout: int = 30) -> str:
    """Run a PowerShell command and return its output (stdout + stderr combined). Windows only.
    Use for: running tests, checking git status, installing packages, etc.
    timeout: max seconds to wait (default 30)."""
    exe = shutil.which("pwsh") or shutil.which("powershell")
    if not exe:
        return "ERROR: powershell not found"
    denial = _approve_command(ctx, "PowerShell", command)
    if denial: raise ValueError(f"User denied your powershell request, with reason: {denial}")
    try:
        result = subprocess.run([exe, "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=timeout, cwd=ctx.cwd)
        out = result.stdout + result.stderr
        if result.returncode != 0:
            out = f"[exit code {result.returncode}]\n" + out
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"


COMMANDLINE_TOOL = powershell if _IS_WINDOWS else bash



def read_warnings(ctx: ex6.Context, path: str) -> str:
    """Run pyright on a file and return its warnings and errors."""
    result = subprocess.run(
        ["pyright", ctx.resolve(path)],
        capture_output=True,
        text=True,
        cwd=ctx.cwd,
    )
    return (result.stdout + result.stderr).strip() or "No warnings."


def git_working_tree(ctx: ex6.Context) -> str:
    """Show working tree changes: status + unstaged/staged diffs."""
    try:
        repo = git.Repo((ctx and ctx.cwd) or os.getcwd(), search_parent_directories=True)
    except Exception:
        return "ERROR: not in git repository"

    branch = repo.active_branch.name if not repo.head.is_detached else "(detached)"
    status_lines = [f"## {branch}"]

    staged_paths = [d.a_path for d in repo.index.diff("HEAD")]
    unstaged_paths = [d.a_path for d in repo.index.diff(None)]
    untracked_paths = list(repo.untracked_files)

    for p in staged_paths:
        status_lines.append(f"M  {p}")
    for p in unstaged_paths:
        status_lines.append(f" M {p}")
    for p in untracked_paths:
        status_lines.append(f"?? {p}")
    if len(status_lines) == 1:
        status_lines.append("(clean)")

    unstaged_diff = repo.git.diff("--no-ext-diff", "--", ".").strip() or "(no unstaged diff)"
    staged_diff = repo.git.diff("--cached", "--no-ext-diff", "--", ".").strip() or "(no staged diff)"

    return (
        "=== git status --short --branch ===\n"
        + "\n".join(status_lines)
        + "\n\n=== git diff (unstaged) ===\n"
        + unstaged_diff
        + "\n\n=== git diff --cached (staged) ===\n"
        + staged_diff
    )




EXPLORE_MODEL = M.GEMINI3_FLASH.id

EXPLORE_SYSTEM_PROMPT = Message(role="system", overview="explore-system", content="""\
You are a fast, read-only exploration agent. Your output is given to another agent - plain text only, no markdown headers, no tables, no emojis.

<goal>
Understand the code, then return a tight, information-dense summary. No fluff. Match length to information content.
</goal>

<strategy>
- Start broad, go deep. Use multiple search angles — different naming conventions, related files, alternate locations.
- Maximize parallel tool calls. Read multiple files and search multiple patterns in a single run_tools block.
- Start tools like `search` / `glob` to find out where to go, then `read_file` for going deep.
- IMPORTANT: YOU MUST KEEP ROUND-TRIPS TO A MINIMUM, SINCE YOU ARE ON A TIME LIMIT. DO A MAXIMUM OF 5 ROUND-TRIPS, AND ALWAYS CALL TOOLS IN BATCHES.
</strategy>

<output>
- Bullet points over paragraphs. Code references (file:function_name) over prose.
- Concrete facts, relevant paths, function names, relationships.
- Favour conciseness at all costs. Conciseness is much more important than grammatical correctness.
- Be EXTREMELY CONCISE. Do NOT use "the", "a", "it looks like", or anything else that bloats the output.
- Be fast. Try to write 1 line if possible, more than 1 line only if needed.
</output>
""",
tools = [read_file, glob, search, read_headers, read_body]
)


def explore_agent(ctx: ex6.Context, prompt: str, files: list = None) -> str:
    """Spawn a read-only subagent to explore the codebase. Returns its findings.
    files: optional file paths to pre-read and include in the prompt.
    
    DO ask questions that are specific and general.
    Do NOT ask questions that delegate your entire task.
    
    <example>
    User: How does the data pipeline work when the microservices are in debug mode?

    ASSISTANT: vexplore("How does the data pipeline work when the microservices are in debug mode?") 
    (BAD: delegating entire task; subagent will be overwhelmed:)

    ASSISTANT: explore("Where does the data pipeline start?")
    - explore("What is microservice debug mode and what does it do?")
    - read_headers("data_service.py")
    (GOOD: being more specific, splitting it up, using hybrid )
    </example>
    """
    if files:
        parts = []
        for f in files:
            fp = ctx.resolve(f)
            with open(fp, "r", encoding="utf-8") as fh:
                parts.append(f'<file path="{f}">\n{fh.read()}\n</file>')
        prompt = "\n".join(parts) + "\n\n" + prompt

    sub_name = f"explore_{int(time.time() * 1000)}"
    sub = Context(sub_name, model=EXPLORE_MODEL, reasoning="none", cwd=ctx.cwd, messages=[EXPLORE_SYSTEM_PROMPT])
    add_tool_repetition_guard(sub, [read_file, read_headers, read_body, search, glob])
    sub.parent = ctx.name
    sub.invoke(prompt)
    while sub.llm_is_running:
        time.sleep(0.05)

    try:
        if sub.llm_result and sub.llm_result.error:
            raise RuntimeError(f"explore_agent failed: {sub.llm_result.error}")
        messages = sub.get_messages()
        return messages[-1].content if messages else ""
    finally:
        ex6.remove_context(sub)


def _normalize_guard_value(v):
    if isinstance(v, dict):
        return {str(k): _normalize_guard_value(vv) for k, vv in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_normalize_guard_value(x) for x in v]
    if isinstance(v, str):
        return v.replace("\\", "/")
    return v


def _guard_fingerprint(tool_name: str, kwargs: dict) -> str:
    payload = {
        "tool": tool_name,
        "args": _normalize_guard_value(kwargs),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _count_matching_tool_outputs(ctx: ex6.Context, fp: str, output: str) -> int:
    tc_map = {}
    count = 0
    for m in ctx.get_messages():
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                tc_fp = _guard_fingerprint(tc.get("name", ""), tc.get("args", {}))
                tc_map[tc.get("id")] = tc_fp
            continue
        if m.role != "tool":
            continue
        if tc_map.get(m.tool_call_id) != fp:
            continue
        if str(m.content or "") == output:
            count += 1
    return count


def guard_repeat_calls(fn):
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if not params:
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        ctx = bound.arguments.get(params[0].name)
        if not isinstance(ctx, ex6.Context):
            return fn(*args, **kwargs)

        call_kwargs = {k: v for k, v in bound.arguments.items() if k != params[0].name}
        fp = _guard_fingerprint(fn.__name__, call_kwargs)

        out = fn(*args, **kwargs)
        out_str = str(out or "")
        if _count_matching_tool_outputs(ctx, fp, out_str) >= 2:
            return f"ERROR: blocked repeated tool call ({fn.__name__}) with same args+output. Use previous tool output already in context."
        return out

    wrapped.__signature__ = sig
    wrapped.__name__ = fn.__name__
    wrapped.__qualname__ = fn.__qualname__
    wrapped.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    wrapped._ex6_repeat_guard_wrapped = True
    return wrapped


def add_tool_repetition_guard(ctx: ex6.Context, guard: Optional[list] = None):
    """Wrap selected tools in this context to block repeated identical calls (3rd+)."""
    names = None if guard is None else {fn.__name__ for fn in guard}
    for m in ctx.get_messages():
        if not getattr(m, "tools", None):
            continue
        new_tools = []
        for fn in m.tools:
            should_wrap = names is None or fn.__name__ in names
            if should_wrap and not getattr(fn, "_ex6_repeat_guard_wrapped", False):
                fn = guard_repeat_calls(fn)
            new_tools.append(fn)
        m.tools = new_tools
    return ctx

