
import os, json, base64, urllib.parse, urllib.request
import ex6
import time
from _ex6.models import M
from _ex6.tools import add_tool_repetition_guard


_API = "https://api.zyte.com/v1/extract"

def _zyte(payload: dict) -> dict:
    key = os.environ.get("ZYTE_API_KEY", "")
    if not key:
        raise RuntimeError("ZYTE_API_KEY not set")
    auth = base64.b64encode(f"{key}:".encode()).decode()
    req = urllib.request.Request(
        _API,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def web_search(ctx: ex6.Context, query: str) -> str:
    """Search the web. Returns top results as text."""
    url = "https://www.google.com/search?" + urllib.parse.urlencode({"q": query})
    resp = _zyte({"url": url, "serp": True})
    out = []
    for r in resp.get("serp", {}).get("organicResults", [])[:8]:
        title = r.get("name", "").strip()
        href = r.get("url", "").strip()
        snippet = r.get("description", "").strip()
        out.append(f"{title}\n  {href}\n  {snippet}")
    if not out:
        raise ValueError("No results found! Do NOT make up results; inform your user that there was likely a search error.")
    return "\n\n".join(out)




def web_scrape(ctx: ex6.Context, url: str, max_chars: int = 50_000) -> str:
    """
    Fetch and return the readable text content of a webpage.
    - Use this after web_search() to read the full content of a result
    - max_chars: truncate output to this many characters (default 50k)
    - Will fail gracefully on paywalled, bot-protected, or JS-only pages
    - Avoid scraping the same URL repeatedly in one session
    """
    resp = _zyte({"url": url, "pageContent": True, "pageContentOptions": {"extractFrom": "httpResponseBody"}})
    md = resp.get("pageContent", {}).get("itemMain", "")
    if isinstance(md, dict):
        md = md.get("markdown") or md.get("text") or ""
    if not md:
        raise RuntimeError("web_scrape returned no content")
    if len(md) > max_chars:
        md = md[:max_chars] + "\n\n[...truncated]"
    return md






WEBSEARCH_SYSTEM_PROMPT = ex6.Message(role="system", content="""\
You are a focused web research agent. Answer the given question using web_search and web_scrape tools.

<strategy>
- Search snippets are SHORT and often STALE. For anything technical, API-related, or detailed: you MUST scrape the actual page.
- Only skip scraping for trivial factual questions (e.g. "what year was X founded").
- If the first search doesn't find what you need, refine your query and search again. Try different keywords, site: filters, or quoted phrases.
- If a scraped page doesn't have the answer, scrape another result or search with different terms.
- You may need 2-4 tool calls to get a good answer. Don't give up after one search.
</strategy>

<search_tips>
- Use specific terms, not vague questions. "openai responses API python client.responses.create" > "how to use openai API".
- Use site: filters for official docs: site:docs.python.org, site:developer.mozilla.org, etc.
- Use quotes for exact phrases: "client.responses.create".
</search_tips>

<output>
- Concise — facts only, no padding. Conciseness >> grammatical correctness.
- Plain text only, no markdown.
- Include specific details: code examples, parameter names, URLs. Don't summarize away the useful parts.
</output>
""")


WEBSEARCH_TOOLS_MSG = ex6.Message(role="system", overview="tools",
    content="Use web_search to find pages, and web_scrape to read a page in full.",
    tools=[web_search, web_scrape])

def websearch_agent(ctx: ex6.Context, question: str) -> str:
    """Spawn a websearch subagent to research a question. Returns a concise answer.
    Use this when you need up-to-date information from the web."""
    sub = ex6.Context("websearch", model=M.GEMINI_LATEST.id, messages=[WEBSEARCH_SYSTEM_PROMPT, WEBSEARCH_TOOLS_MSG], reasoning="none")
    add_tool_repetition_guard(sub, [web_search, web_scrape])
    sub.parent = ctx.name
    sub.invoke(question)
    while sub.llm_is_running:
        time.sleep(0.05)
    messages = sub.get_messages()
    result = messages[-1].content if messages else "No answer."
    del ex6.state.contexts[sub.name]
    return result

