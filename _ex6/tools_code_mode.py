import ex6
import re
import ast
import threading
import inspect



def _build_tool_docs(ctx: ex6.Context) -> str:
    """Dynamic content: generates tool docs from all messages in ctx."""
    tools = ctx.get_tools()
    if not tools:
        return "No tools available."

    lines = ["You have access to the following tools:", "<tools>"]
    for name, fn in tools.items():
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())[1:]  # skip ctx
        args = ", ".join(p.name for p in params)
        doc = (fn.__doc__ or "").split('\n')[0].strip()
        lines.append(f"  {name}({args}) - {doc}")
    lines.append("</tools>")
    lines.append("")
    lines.append("To call tools, output a ```tools block:")
    lines.append("```tools")
    lines.append('read_file("path/to/file")')
    lines.append('write_file("out.txt", "content")')
    lines.append("```")
    return "\n".join(lines)


# This message auto-aggregates tools when added to a context
tool_system_prompt = ex6.Message(role="system", content=_build_tool_docs)


def extract_tools_block(content: str) -> str | None:
    m = re.search(r'```tools\s*\n(.*?)```', content, re.DOTALL)
    return m.group(1).strip() if m else None


@ex6.override
def call_tools(ctx: ex6.Context, llmres: ex6.LLMResult) -> bool:
    content = ""
    for msg in reversed(ctx.messages):
        if msg.role == "assistant":
            content = msg.content if isinstance(msg.content, str) else ""
            break

    code = extract_tools_block(content)
    if not code: return False

    tools = ctx.get_tools()
    tree = ast.parse(code)
    threads = []
    results = []

    for node in tree.body:
        call = None
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
        if not call: continue

        fn_name = call.func.id if isinstance(call.func, ast.Name) else None
        if not fn_name or fn_name not in tools: continue

        args = [ast.literal_eval(a) for a in call.args]
        kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        call_str = f'{fn_name}({", ".join(repr(a) for a in args)})'
        result = {"value": None}
        results.append((call_str, result))

        fn = tools[fn_name]
        def run_tool(fn=fn, args=args, kwargs=kwargs, result=result):
            result["value"] = fn(ctx, *args, **kwargs)
        t = threading.Thread(target=run_tool)
        t.start()
        threads.append(t)

    for t in threads: t.join()

    if results:
        parts = [f"<tool_result {cs}>\n{r['value']}\n</tool_result>" for cs, r in results]
        ctx.messages.append(ex6.Message(role="user", content="\n\n".join(parts)))

    return len(results) > 0
