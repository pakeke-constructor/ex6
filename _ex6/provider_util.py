# Shared provider helpers: invoke logging (invokes.db) and /usage.

import json
import hashlib
import os
import sqlite3
import ex6


@ex6.command
def usage(tui):
    """Show today's spending."""
    from _ex6.commands import _text_panel
    lines = [
        f"Today: ${tui.app.budget.get_cost():.4f} / ${tui.app.budget.get_limit():.2f}",
    ]
    _text_panel(tui, lines)


def _log_safe(value):
    if isinstance(value, dict) and value.get("type") in ("base64", "image") and "data" in value:  # anthropic / MCP images
        return {**value, "data": "<base64 data omitted>"}
    if isinstance(value, dict):
        return {k: _log_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_log_safe(v) for v in value]
    if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
        return "<base64 data omitted>"
    return value


_DB_LIMIT = 1_000_000_000  # bytes of live data in invokes.db; oldest invokes are culled past this


def _db():
    db = sqlite3.connect(ex6.get_folder() / "invokes.db", timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS blobs(hash TEXT PRIMARY KEY, data TEXT);
        CREATE TABLE IF NOT EXISTS invokes(id INTEGER PRIMARY KEY, ts, cwd, ctx, model, provider,
            input_tokens, output_tokens, cached_tokens, cache_write_tokens, cost, finish_reason, error,
            messages TEXT, tools TEXT, output TEXT);
    """)
    return db


def _blob(db, value):
    data = json.dumps(_log_safe(value), sort_keys=True)
    h = hashlib.sha256(data.encode()).hexdigest()
    db.execute("INSERT OR IGNORE INTO blobs VALUES (?, ?)", (h, data))
    return h


def _db_size(db):
    return db.execute("SELECT (page_count - freelist_count) * page_size "
                      "FROM pragma_page_count(), pragma_freelist_count(), pragma_page_size()").fetchone()[0]


def _cull(db):
    """Over the limit: drop oldest invokes and their orphaned blobs until under 90% of it."""
    if _db_size(db) <= _DB_LIMIT:
        return
    while _db_size(db) > _DB_LIMIT * 0.9:
        db.execute("DELETE FROM invokes WHERE id IN "
                   "(SELECT id FROM invokes ORDER BY id LIMIT (SELECT COUNT(*) / 10 + 1 FROM invokes))")
        db.execute("DELETE FROM blobs WHERE hash NOT IN "
                   "(SELECT value FROM invokes, json_each(invokes.messages) UNION SELECT tools FROM invokes)")


def log_invoke(ctx, provider, messages, tools, result, text=""):
    """Record a finished invoke. messages/tools: the provider-format payload actually sent."""
    from datetime import datetime
    output = {"text": text, "tool_calls": result.tool_calls, "native_calls": result.native_calls}
    db = _db()
    try:
        with db:  # one transaction per invoke
            hashes = [_blob(db, m) for m in messages]
            db.execute(
                "INSERT INTO invokes (ts, cwd, ctx, model, provider, input_tokens, output_tokens, cached_tokens,"
                " cache_write_tokens, cost, finish_reason, error, messages, tools, output)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), ctx.cwd or os.getcwd(), ctx.name, ctx.model,
                 provider, result.input_tokens, result.output_tokens, result.cached_tokens,
                 result.cache_write_tokens, result.cost, result.finish_reason, result.error,
                 json.dumps(hashes), _blob(db, tools or []), json.dumps(_log_safe(output))))
            _cull(db)
    finally:
        db.close()
