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

| Requirement | Notes |
| --- | --- |
| Python 3.12+ | `uv` can download and manage this for you, so a system install is optional |
| [uv](https://docs.astral.sh/uv/) | Package and environment manager |
| [Ollama](https://ollama.com) | Local model server |
| A tool-capable model | Default is `qwen3:8b` (about 5 GB). Needs the `tools` capability |
| RAM | 16 GB recommended for an 8B model at the default 16k context |

## Installation

### 1. Install the prerequisites

```bash
# uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh      # or: brew install uv

# Ollama: download from https://ollama.com/download, or on macOS:
brew install ollama
```

### 2. Pull a model

```bash
ollama pull qwen3:8b
```

### 3. Get Forge and install its dependencies

```bash
git clone https://github.com/<your-username>/forge-agent.git
cd forge-agent
uv sync
```

`uv sync` creates a `.venv`, installs Python 3.12 if it is not already available, and installs
Forge with its dependencies. Nothing is installed system-wide.

### 4. Verify the setup

Make sure the Ollama server is running (the desktop app starts it automatically; otherwise run
`ollama serve` in a separate terminal), then:

```bash
uv run forge doctor
```

You should see `OK` for the server, the model, and tool calling. If anything shows `FAIL`, see
[Troubleshooting](#troubleshooting).

### Optional: install `forge` as a global command

To run `forge` from any project directory without `uv run`:

```bash
uv tool install .
forge --version
```

Re-run `uv tool install . --force` after pulling new changes to update it.

## Usage

### Quick start

Create a scratch project and give the agent a task:

```bash
mkdir -p /tmp/forge-demo
uv run forge run "Create hello.py that prints hello, then run it to check it works" \
  --workspace /tmp/forge-demo
```

By default, every file write and command stops and asks for your approval. Review each prompt and
answer `y` or `n`.

> **Pick the workspace deliberately.** The agent can read and modify files in the workspace, which
> defaults to your current directory. When running from inside the `forge-agent` repository,
> always pass `--workspace` or the agent will be working on Forge's own source. If you installed
> the global command, simply `cd` into the project you want to work on first.

### Example tasks

```bash
# Explore without changing anything (read-only)
forge run "Explain how the tests in this project are organised" --approval deny

# Fix a bug and verify it
forge run "Fix the failing test in tests/test_math.py and run pytest to confirm"

# Add a feature, auto-approving edits (allowlisted commands only; use on a git-tracked project)
forge run "Add input validation to parse_config() and add tests" --approval auto
```

### Approval modes

| Mode | Behaviour |
| --- | --- |
| `ask` (default) | Prompts before every file write and command |
| `auto` | Approves writes inside the workspace and allowlisted commands without asking |
| `deny` | Read-only: any write or command is refused |

Reading and listing files never needs approval. Commands outside the allowlist are always refused,
in every mode.

### Command-line options

| Option | Meaning |
| --- | --- |
| `--workspace PATH` | Project directory the agent may work in (default: current directory) |
| `--model TAG` | Any Ollama model that supports tools (default `qwen3:8b`) |
| `--host URL` | Ollama server URL (default `http://localhost:11434`) |
| `--approval ask\|auto\|deny` | See [Approval modes](#approval-modes) |
| `--max-steps N` | Hard cap on agent iterations (default 25) |
| `--think / --no-think` | Toggle model thinking mode, if the model supports it |
| `--show-thinking` | Print the model's reasoning |
| `--no-trace` | Skip writing the JSONL trace |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING` (default), and so on |

Run `forge run --help` for the full list. You can also invoke the module directly with
`python -m forge_agent`.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Task completed |
| `1` | The run did not complete (step limit, loop detected, or model error) |
| `2` | Invalid configuration, such as a workspace that does not exist |

### Configuration

Every setting can also be provided as an environment variable or in a `.env` file in the directory
you run from, using a `FORGE_` prefix. Command-line options take priority over these.

```bash
# .env
FORGE_MODEL=qwen3:8b
FORGE_NUM_CTX=16384        # context window; raise carefully, it costs RAM
FORGE_MAX_STEPS=30
FORGE_APPROVAL_MODE=ask
FORGE_THINK=true
```

All available settings, with their defaults and validation rules, are in
`src/forge_agent/config.py`.

### Run traces

Each run writes a JSONL trace to `<workspace>/.forge/runs/run-<timestamp>.jsonl`, recording the
assistant's messages, every tool call, and every result. It is the first thing to look at when a
run goes wrong. Add `.forge/` to the project's `.gitignore`.

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `Cannot reach Ollama at http://localhost:11434` | The server is not running. Open the Ollama app or run `ollama serve` |
| `Model 'X' is not available locally` | Run `ollama pull X`, or pick an installed model with `--model` (see `ollama list`) |
| `does not report tool support` | The model cannot call tools. Choose one that lists `tools` under `ollama show <model>` |
| `think is enabled but ... lacks 'thinking'` | Drop `--think`, or use a model that supports thinking |
| `Stopping: 'X' was called 3 times in a row` | The model got stuck repeating itself. Rephrase the task, make it smaller, or try a larger model |
| `Stopped after N steps without finishing` | Task too large for the step budget. Split it up, or raise `--max-steps` |
| A command is refused as "not allowed" | By design. Only allowlisted commands run, with no pipes, redirects, `&&`, or inline `python -c` |
| Very slow responses or memory pressure | Lower `FORGE_NUM_CTX` (for example 8192), close other apps, or use a smaller model |
| `python` not found inside the agent | Commands run with a minimal environment. Make sure `python` is on your `PATH`, or have the agent use `uv run` |

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