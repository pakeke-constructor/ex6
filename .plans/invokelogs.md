# invokelogs: agent improvements found in invoke logs (leftovers)
Status: parked

## Motivation (user's words)
"go over all the logs for the past few days ... see if there are any obvious improvements we can make to our agents."
"look for common patterns that agents do that could be done with way less tokens. (Via smarter tools, or more condensed output.)"
Only anth_OPUS (claude-opus-5.5) and oai_ASTRA (gpt-6-astra) matter; gpt-6-luna is ignored.

## Pipeline
`python scripts/invoke_insights.py` builds `analysis.db` next to `invokes.db`, with one row per tool call: its arguments, its result, and how many later invokes re-sent that result. `... dump <invoke_id>` prints a single transcript. 95% of Opus input is cache hits, so cost is roughly context size times number of invokes, and repeated results are the main thing to cut.

Done: AfterOtherTools, AGENTS.md re-reads, condense, bash cwd hint. Split out: `.plans/batching.md`, `.plans/outputfmt.md`. Full-file reads: no real fix.

## Remaining small items
- `handoff` with `ready=False` costs an extra round trip at about 90k context (31 times). Put the guidelines in the tool description.
- Images are re-sent for a median of 11 invokes (about 3% of Opus input). Elide them after K rounds.
- The MEV knowledge-base subagents fetch web pages through python heredocs (147 calls, 11.5M carried). Needs a project-level `web_fetch(url, grep, offset)` tool.
- 9 errors came from subagents trying to start their own subagents. Remove the tool from subagents.
- `read_audio` could raise `ex6.AfterOtherTools` on a missing file, like read_image.
