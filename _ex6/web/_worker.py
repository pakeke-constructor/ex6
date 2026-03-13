"""Runs inside the isolated venv. Takes URL as argv[1], prints JSON to stdout."""
import sys, json, asyncio, os

# crawl4ai prints progress to stdout; redirect to devnull
_real_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")


async def _scrape(url: str, max_chars: int = 50_000):
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    config = CrawlerRunConfig(
        css_selector="article, main, [role='main']",
        excluded_tags=["nav", "footer", "header", "aside", "form"],
        exclude_all_images=True,
        exclude_external_links=True,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
        md = result.markdown.raw_markdown.strip()
        # fallback: if css_selector matched nothing, retry without it
        if len(md) < 100:
            config = CrawlerRunConfig(
                excluded_tags=["nav", "footer", "header", "aside", "form"],
                exclude_all_images=True,
                exclude_external_links=True,
            )
            result = await crawler.arun(url=url, config=config)
            md = result.markdown.raw_markdown.strip()
        if len(md) > max_chars:
            md = md[:max_chars] + "\n\n[...truncated]"
        return md

try:
    max_chars = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    md = asyncio.run(_scrape(sys.argv[1], max_chars))
    sys.stdout = _real_stdout
    print(json.dumps({"ok": True, "markdown": md}))
except Exception as e:
    sys.stdout = _real_stdout
    print(json.dumps({"ok": False, "error": str(e)}))
