# MYCODE.md — mycode

Native coding agent with a Lean 4 decision core and a Rust Ratatui/Crossterm shell. Node.js and TypeScript are out of scope.

## Architecture

- `mycode/` owns the authoritative agent state machine, tool ordering, approval decisions, session snapshots, and restart recovery.
- `crates/mycode-tui/` owns terminal input/rendering, direct OpenAI and Anthropic adapters, the OMP-backed Linewise OpenAI gateway adapter, and asynchronous realization of Lean effects.
- `crates/mycode-plugin-protocol/` owns the versioned external-plugin wire contract: four-byte big-endian length followed by bounded UTF-8 JSON.
- `crates/mycode-workspace-plugin/` is the first-party workspace plugin. `read` and `grep` are automatically safe; `write`, `edit`, and `bash` require a Lean approval effect.
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

For provider or orchestration changes, run the affected Ratatui flow against the local OpenAI or Anthropic example server, plus a real Linewise gateway smoke when that adapter changes. Observe model → plugin tool → model completion, cooperative Escape cancellation, and a clean TUI exit.

## Git

Do not commit generated output under `target/`, Lean `.lake/`, or generated `lake-manifest.json`. Commit or push only when explicitly requested.
