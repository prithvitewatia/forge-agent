# Architecture

## Goals

1. Small enough to read in one sitting.
2. Safe by default, because the model is untrusted input.
3. Testable without a model: every external boundary is an interface with a fake.
4. Honest failure: guards turn confusion into a clear status, never a hang.

## Module map

| Module | Responsibility |
| --- | --- |
| `agent.py` | The loop, `RunResult`, and the step/repetition guards |
| `messages.py` | Provider-neutral `Message`, `ToolCall`, `ToolSpec`, `AssistantTurn` |
| `llm/base.py` | `LLMClient` protocol, the only thing the agent knows about a model |
| `llm/ollama_client.py` | Ollama adapter; all wire-format and error translation lives here |
| `tools/base.py` | `Tool` ABC, `ToolRegistry` (validation, approval, truncation, error containment) |
| `tools/filesystem.py`, `tools/shell.py` | Concrete tools |
| `tools/validate.py` | Heuristics that catch mangled files (double-escaped newlines, bad syntax) |
| `safety/paths.py` | `Workspace`: path confinement |
| `safety/policy.py` | `CommandPolicy` allowlist, `ModeApprover` approval modes |
| `context.py` | History budgeting (elide oldest tool output first) |
| `events.py` | `EventSink` observer, JSONL trace writer |
| `config.py` | Validated, immutable `Settings` (defaults, `.env`, `FORGE_*`) |
| `doctor.py` | Environment diagnostics |
| `cli.py` | Typer CLI and terminal rendering |

Dependencies point inward: the agent depends on protocols (`LLMClient`, `EventSink`, `Approver`),
never on Ollama, Rich or Typer.

## One iteration of the loop

1. Trim history to the token budget (`context.fit_to_budget`).
2. Call `LLMClient.chat(messages, tool_specs)`.
3. No tool calls: the run is complete (one nudge is given if the reply is empty).
4. For each tool call: check the repetition guard, then `ToolRegistry.execute`.
5. Append each result as a `tool` message and loop, up to `max_steps`.

### `ToolRegistry.execute` pipeline

unknown tool? -> validate args with the tool's pydantic model -> `precheck()` -> `describe()` the
action -> approval gate (skipped for read-only tools) -> run -> truncate output.

`precheck()` is an optional per-tool hook that rejects obviously bad input before a human is asked
to approve it, so nobody is prompted for an action that would be refused anyway.

Every expected failure becomes a `ToolResult(is_error=True)` string for the model. An unexpected
exception is logged and also returned as text, so a buggy tool cannot crash a run.

## Design decisions

- **Provider interface, not a library call.** A one-method protocol keeps the agent independent of
  Ollama and makes a scripted fake trivial.
- **Args validated with pydantic, schemas generated from the same models.** One source of truth for
  what the model is told and what the code accepts. Validation errors go back to the model.
- **Errors as observations.** Small models recover from clear error text far more often than they
  recover from a crashed run.
- **Immutable settings.** One validated object, no globals.
- **Observer for output.** The same loop drives the terminal, the JSONL trace, and the tests.
- **No shell.** Commands are parsed with `shlex` and run as an argv list. Shell metacharacters are
  refused, which removes injection and chaining as a class.
- **Guards for mangled files** (`tools/validate.py`). Small models sometimes double-escape newlines
  in tool arguments, so a multi-line file arrives as one line. `write_file` refuses content with no
  real line break and two or more literal `\n` sequences, and tells the model how to resend. A
  genuine single-line file passes if it ends with one real newline. After any write or edit, TOML,
  JSON and Python files are parsed and a failure is appended as a `WARNING` rather than an error,
  because intermediate edits can be legitimately invalid. Write results also report a line count.
  These are heuristics that make failures visible to the model; they do not make a weak model
  reliable.

## Threat model

The model is treated as untrusted: it may hallucinate, be confused, or be steered by text inside
files it reads (prompt injection).

| Threat | Mitigation |
| --- | --- |
| Reading or writing outside the project | `Workspace` resolves symlinks and rejects escapes; command path arguments are checked too |
| Destructive or networked commands | Executable allowlist; read-only git subcommands only; global git options and file-writing flags refused |
| Command chaining and injection | No shell; metacharacters rejected |
| Secret leakage via child processes | Environment scrubbed to a small passthrough list |
| Runaway cost or loops | `max_steps`, repetition guard, command timeout, output truncation |
| Unreviewed changes | Approval gate on every write and command (default `ask`) |

### Known gaps

- **Not a sandbox.** Allowlisted interpreters (`python`, `pytest`, `uv`) can execute arbitrary code
  as your user, including code the agent just wrote. The allowlist limits accidents and casual
  misuse, not a determined attacker. Use a container or VM for untrusted work.
- **Prompt injection is not solved.** File contents are fed to the model. The approval gate and the
  confinement rules are the defence; keep approval on when working with files you do not trust.
- **Token estimation is approximate** (characters / 4), so the budget is a guardrail, not exact.

## Extending

- **New provider:** implement `LLMClient.chat` and translate to and from `Message` / `ToolSpec`.
- **New tool:** subclass `Tool`, declare `name`, `description`, `risk`, and a pydantic `args_model`,
  then register it in `build_default_registry`.
- **New front end:** implement `EventSink` and `Approver`.
