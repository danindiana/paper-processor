# openclaw ↔ Ollama hookup (resumed after Bun crash)

**Host:** morlok-AM4
**Working dir:** `/run/media/morlok/WORLOCK/home/jeb/programs/python_programs/paper_processor`
**Started:** 2026-05-09 ~05:56 UTC (prior session, crashed mid-final-test)
**Resumed:** 2026-05-09 06:08 UTC
**openclaw version:** 2026.5.7

## Goal

Wire openclaw's gateway to talk to local Ollama models so paper_processor's
`--backend openclaw` path has something to route through.

## What the prior session accomplished

1. Confirmed openclaw recognises the `ollama` provider API (schema includes
   `injectNumCtxForOpenAICompat` for OpenAI-compat shimming).
2. Wrote a JSON5 patch installing an `ollama-local` provider in
   `~/.openclaw/openclaw.json` with all 11 locally-installed Ollama models:

   | Model id                          | Context |
   | --------------------------------- | ------- |
   | `qwen3.5:9b`                      | 262144  |
   | `qwen3.5:4b-q4_K_M`               | 262144  |
   | `qwen3.5:2b`                      | 262144  |
   | `qwen3.5:2b-q8_0`                 | 262144  |
   | `deepseek-r1:8b`                  | 131072  |
   | `ministral-3:14b`                 | 131072  |
   | `ministral-3:8b`                  | 131072  |
   | `gemma4:e4b`                      | 131072  |
   | `nemotron-3-nano:latest`          | 131072  |
   | `nemotron-3-nano:4b`              | 131072  |
   | `nemotron-3-nano:4b-q8_0`         | 131072  |

   Provider stanza: `baseUrl: http://localhost:11434`, `auth: api-key`,
   `apiKey: "ollama"`, `timeoutSeconds: 1200`.

3. Set the agent default model to `ollama-local/qwen3.5:9b` via
   `openclaw models set`.
4. Hit a wall on the gateway path: `Model override "ollama-local/qwen3.5:2b"
   is not allowed for agent "main"`. Root-caused by reading the gateway code
   (`hasAllowlist = agentCfg?.models && Object.keys(...).length > 0`) and
   the live config — the `models set` command had only added the *default*
   model to `agents.defaults.models`, leaving 10 of the 11 unauthorised.
5. Wrote a second JSON5 patch adding all 11 models to
   `agents.defaults.models`, applied it, restarted the gateway.

**Crash point:** Bun (or the `openclaw` Node process — log says `node`,
not `bun`, but the user reported a Bun crash; possibly a wrapping CLI)
exited mid-`echo`/`timeout` chain that was about to run the
gateway-routed smoke test. Last log entry was the agent-cfg dump.

## Resumption (this session)

### State on resume — clean

- `~/.openclaw/openclaw.json`: all 11 models present in
  `agents.defaults.models` and in `models.providers.ollama-local.models`.
  Gateway auth token unchanged.
- `openclaw-gateway.service`: active since 01:06:59 CDT, exactly the timestamp
  of the prior session's restart — the service survived; the CLI crashed.
- Last gateway log: `agent model: ollama-local/qwen3.5:9b (thinking=medium, fast=off)`.

No re-patching needed.

### Smoke tests

```bash
# Embedded local agent path — bypasses gateway auth gate
$ openclaw agent --local --agent main --model "ollama-local/qwen3.5:2b" \
    --message "Reply with the single word PONG and nothing else."
PONG

# Gateway-routed path WITH model override — still gated
$ openclaw agent --agent main --model "ollama-local/qwen3.5:2b" \
    --message "Reply with the single word PONG and nothing else."
GatewayClientRequestError: provider/model overrides are not authorized for this caller.

# Gateway-routed path using agent default (no override) — works
$ openclaw agent --agent main \
    --message "Reply with the single word PONG and nothing else."
PONG

# Health
$ openclaw health
Gateway event loop: degraded reasons=event_loop_utilization,...
Agents: main (default)
- agent:main:main (1m ago)
$ echo $?
0
```

## Conclusions

- **Hookup is functional.** Local Ollama models reach openclaw via two paths.
- **Gateway path: agent-default only.** Per-call model overrides via the
  gateway hit a separate auth gate (caller authorization), distinct from
  the per-agent allowlist that we just unblocked. To switch models at
  runtime through the gateway you must either:
  1. Make the caller an authorised override-capable client, **or**
  2. Use embedded `--local` mode (no gateway auth gate), **or**
  3. Define multiple agents, one per model, and pick by `--agent`, **or**
  4. Rotate the agent's default via `openclaw models set` between runs.
- **Embedded `--local` is unrestricted** for any model in the agent allowlist.

## Outstanding bug found in paper_processor

`paper_processor.py:453` — `cmd = ["openclaw", "agent", "--message", prompt]`.
Missing the required `--agent` (or `--to`/`--session-id`) selector — every
call errors with:

```
Error: Pass --to <E.164>, --session-id, or --agent to choose a session
```

`health_check_openclaw()` calls `openclaw health` (which exits 0 regardless),
so pre-flight passes and the pipeline only fails once it tries to summarise.

**Suggested fix (to be confirmed):** change the command to
`["openclaw", "agent", "--agent", "main", "--message", prompt]`. Note that
`OPENCLAW_MODEL` env-var-based per-page-count routing is a no-op against the
gateway path — the gateway uses the agent default. To get true per-call
model selection, switch the subprocess to `openclaw agent --local --agent
main --model "<provider/model>" --message "<prompt>"`.

## Files touched (this session)

- `paper_processor.py` (`_call_openclaw`, ~line 442–453) — added `--agent main`
  to the subprocess command and updated docstring/comment to record that
  `OPENCLAW_MODEL` is a no-op via the gateway path. Verified with the exact
  Python subprocess shape: `exit=0`, `stdout=PONG`.
- `~/.openclaw/openclaw.json` — already patched by the prior session before
  the crash; verified intact, no changes this session.
- This document.

## Status

Hookup complete. paper_processor `--backend openclaw` will now route every
LLM call through the gateway to `ollama-local/qwen3.5:9b` (the agent default).
Per-page-count model routing is intentionally disabled along this path; if
that becomes important, switch the subprocess to `openclaw agent --local
--agent main --model <m> --message <p>`.
