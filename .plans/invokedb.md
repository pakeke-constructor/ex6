# invokedb: smaller invokes.db, same foolproof structure
Status: active

## Motivation (user's words)
"optimizing the invokes db a bit more (wiping it and changing the structure)."
"Most importantly, it should be incredibly faultless and simple."
"the current way where it duplicates hashes (quadratic growth) is slightly inefficient, but my god it's simple and so foolproof."

Rule: every invoke row stays self-contained (full list of message refs). No delta/parent chains. Those break when we cull, fork, truncate, or rewind, and they'd make every read walk a chain.

## Data (old db, 9 days, 18.6k invokes, 144MB)
blobs.data 73MB, invokes.messages (hash lists) 31MB, output 12.5MB. Of the blob data, 13.7MB is old flattened Claude transcripts, which can't happen any more. zlib per blob: 73MB -> 30MB. The output column goes 12.5MB -> 6.6MB.

## Changes (provider_util.py only)
1. Shorter hash: keep only the first 16 hex chars of the sha256 (64 bits). The hash lists get 4x smaller, and so does the blobs primary key index. A collision is about 1e-8 likely at 1M blobs, and the damage would just be one wrong log entry.
2. Compress blob data: `data BLOB = zlib.compress(json)`. Hashing still uses the uncompressed json. Nothing in `_cull` reads data, so it stays the same.
3. New columns:
   - `duration_ms`: wall time of the invoke (pass the start time into log_invoke, or measure it in the providers).
   - `ctx_id`: a stable per-context uuid in `ctx.data["provider_util:ctx_id"]`. Names like `anth_OPUS` get reused, so this tells threads apart. It survives forks, so a fork stays grouped with its parent, which is fine.
4. Leave out: delta encoding, integer blob ids (would need an insert-then-select step), and image extraction (only ~3MB).

Expected size: about 144MB -> ~65MB for the same traffic.

## Wipe / migration
No migration. Delete `invokes.db` and `analysis.db` (`ex6.get_folder("provider_util")`). `CREATE TABLE IF NOT EXISTS` would keep using an old-schema file without complaint, so the user must wipe before running the new code (or rename the file to `invokes2.db` so old and new can never mix).

## Readers
- `scripts/invoke_insights.py`: decompress blobs. In Python, register `db.create_function("unz", 1, ...)` so ad-hoc SQL like `unz(data) LIKE ...` still works.
- `/usage` in provider_util.py: check that it only reads the token/cost columns.

## Files
- _ex6/provider_util.py (`_db`, `_blob`, `log_invoke`)
- scripts/invoke_insights.py
- _ex6/provider_anthropic.py, provider_openai.py, provider_openrouter.py (log_invoke call sites, if duration is measured there)
