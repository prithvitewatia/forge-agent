# Forge

A small, local-first coding agent built on [Ollama](https://ollama.com). It reads and edits files in
a project directory and runs allowlisted commands (tests, linters, read-only git) to verify its own
work, with no cloud API and no data leaving your machine.

Forge is deliberately compact. It is meant to be readable end to end and to show the parts of an
agent that matter in practice: a provider-agnostic model interface, validated tool calls, a safety
layer, loop guards, context budgeting, and replayable run traces.

> **Status:** alpha. The agent loop, tools and safety layer are covered by an automated test suite
> that uses a scripted fake model. Behaviour against a real local model varies by model size; see
> [Limitations](#limitations).

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally, with a tool-capable model pulled:
  `ollama pull qwen3:8b`

## Quick start

```bash
uv sync
uv run forge doctor                  # verifies server, model, and tool-calling support
uv run forge run "Add a function is_prime(n) to math_utils.py with tests" --workspace ./my-project
```

By default every file write and command asks for your approval. Useful flags:

| Flag | Meaning |
| --- | --- |
| `--approval ask\|auto\|deny` | `ask` prompts per action (default); `auto` approves writes and allowlisted commands; `deny` is read-only |
| `--model TAG` | Any Ollama model that supports tools |
| `--max-steps N` | Hard cap on agent iterations (default 25) |
| `--think / --no-think` | Toggle model thinking mode, if the model supports it |
| `--show-thinking` | Print the model's reasoning |
| `--no-trace` | Skip writing the JSONL trace |

Every setting can also be given as an environment variable or `.env` entry with a `FORGE_` prefix
(for example `FORGE_MODEL`, `FORGE_NUM_CTX`, `FORGE_MAX_STEPS`). See `src/forge_agent/config.py`.

## How it works

```
 task ──> Agent loop ──> LLMClient (Ollama adapter) ──> model
             │  ▲
   tool calls│  │observations
             ▼  │
        ToolRegistry ──> validate args ──> approval gate ──> tool
                                              │
                         Workspace jail + CommandPolicy
```

The loop calls the model, executes any tool calls it requests, feeds the results back, and repeats
until the model answers without a tool call. Tool failures are returned to the model as text so it
can correct itself. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and threat model.

### Tools

`list_dir`, `read_file`, `write_file`, `edit_file` (exact-match replace), `run_command`.

### Safety, in brief

- File access is confined to the workspace (symlinks are resolved before the check; `.git`,
  `.venv` and `.forge` are write-protected).
- Commands run without a shell, from an allowlist, with a scrubbed environment, a timeout, and no
  path arguments outside the workspace. Only read-only `git` subcommands are permitted.
- Writes and commands need approval unless you opt into `--approval auto`.

**This is a guardrail against mistakes, not a security sandbox.** An allowlisted interpreter such as
`python` can run arbitrary code with your user's permissions. For untrusted tasks or models, run
Forge inside a container or VM.

### Observability

Each run writes a JSONL trace (assistant text, tool calls, results) to `<workspace>/.forge/runs/`.
`--log-level DEBUG` enables standard logging.

## Development

```bash
uv sync
uv run pytest                 # tests + coverage
uv run ruff check . && uv run ruff format --check .
uv run mypy                   # strict
```

The tests never need Ollama: they drive the agent with a scripted fake `LLMClient`. To add a model
provider, implement the one-method `LLMClient` protocol in `src/forge_agent/llm/base.py`.

## Limitations

- Small local models (7-8B) are inconsistent at multi-step coding. Expect malformed tool calls,
  premature stops, and the occasional loop; the guards turn these into clean failures, not hangs.
- Context is limited by your hardware (`FORGE_NUM_CTX`, default 16384). Old tool output is
  elided when the history exceeds the budget.
- No shell features (pipes, redirection, `&&`), so inline `python -c "..."` is refused. The agent
  writes a script file and runs it instead.
- Single agent, single model. No retrieval over large codebases yet.

## License

MIT. See [LICENSE](LICENSE).
