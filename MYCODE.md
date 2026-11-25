# MYCODE.md — mycode

Native coding agent with a Lean 4 decision core and a Rust Ratatui/Crossterm shell. Node.js and TypeScript are out of scope.

## How to use

Before changing anything, find it in "Which file to read before changing which kind of file" below and read that topic file in full: use `read` to read all of it, not `grep`, `sed`, or a partial range. Read one file for each area a change touches. For read-only research that changes no code, read until you have enough.

When creating a new file, find the row for its directory in the table; if none fits, read `docs/agents/architecture.md`.

The sections below (Architecture, Lean rules, Rust rules, Verification, Git) apply to every change. Topic files are named by plain path, not `@`-imported.

## Which file to read before changing which kind of file

| Change | Typical paths | Read first |
|---|---|---|
| Lean core | `mycode/MyCode/`, `mycode/*.lean` | `docs/agents/architecture.md` |
| TUI and provider adapters | `crates/mycode-tui/` | `docs/agents/architecture.md` |
| Plugin wire contract | `crates/mycode-plugin-protocol/` | `docs/agents/architecture.md` |
| Workspace plugin | `crates/mycode-workspace-plugin/` | `docs/agents/architecture.md` |
| Build and lint config | `Cargo.toml`, `clippy.toml`, `mycode/lakefile.toml`, `mycode/lean-toolchain` | Rust rules and Verification below |
| This index or a topic file | `MYCODE.md`, `docs/agents/*.md` | `docs/agents/architecture.md` |

## Architecture

- Rust must never execute a model-proposed tool merely because it parsed a provider response. It may invoke a plugin only after the Lean core emits `invoke_tool`.
- Provider payloads and plugin output are untrusted observations. Validate names, framing, sizes, correlation IDs, paths, and JSON before sending normalized events to Lean.
- Provider and plugin effects run off the terminal event loop. Cancellation is cooperative: never abort a task while a `CoreClient` request may be in flight; retire ambiguous plugin transports and close every pending tool call through Lean before accepting another prompt.

## Lean rules

- Keep the transition function deterministic and free of external effects.
- Add every new state transition to focused tests in `mycode/Tests.lean`.
- Persist the next state before returning its effects. A persistence failure must not publish the transition.
- Do not add proof escapes such as `sorry`, `admit`, `axiom`, `opaque`, `unsafe`, or `native_decide`.
- Keep wire field and constructor names stable. Rust depends on Lean's derived JSON encoding.

## Rust rules

Follow the discipline used by the Linewise native executors:

- Edition 2024, pinned direct dependencies, and a declared `rust-version`.
- Every crate root has `#![deny(warnings)]` and `#![deny(clippy::unwrap_used)]`.
- Use boundary-local typed `thiserror` enums. Do not use `anyhow`, catch-all string error variants, or production `unwrap`.
- Preserve typed error sources with `#[source]` or `#[from]`; stringify only at the outer log or wire edge.
- Bound every untrusted frame, HTTP body, file read, subprocess stream, diagnostic string, and deadline before buffering it.
- On plugin transport ambiguity or timeout, retire the process. Never retry an effect whose completion is unknown.
- Tests may use `unwrap`/`expect` only under the repository `clippy.toml` policy.

## Verification

Run after Lean changes:

```bash
cd mycode
lake build mycode mycode_tests
./.lake/build/bin/mycode_tests
```

Run after Rust changes:

```bash
cargo fmt --all
cargo test --locked --workspace
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

For provider or orchestration changes, run the affected Ratatui flow against the local OpenAI or Anthropic example server, plus a real Linewise gateway smoke when that adapter changes. Observe model → live stdout/stderr tail → plugin terminal result → model completion, model-request and tool-boundary steer, plan proposal → interactive review → approved execution, persisted todo state, manual and threshold-triggered compaction with full transcript preservation, `auto` allow and prompt paths, cooperative Escape cancellation, and a clean TUI exit.

## Git

Do not commit generated output under `target/`, Lean `.lake/`, or generated `lake-manifest.json`. Commit or push only when explicitly requested.
