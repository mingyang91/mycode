# Architecture: crate ownership, Lean-owned control state, and TUI behavior

Read this in full before changing the Lean core, the TUI, the plugin protocol, or the workspace plugin. The safety boundaries that every change must keep stay in `MYCODE.md` under "Architecture".

## Architecture

- `mycode/` owns the authoritative agent state machine, tool ordering, approval decisions, session snapshots, and restart recovery.
- `crates/mycode-tui/` owns terminal input/rendering, direct OpenAI and Anthropic adapters, the OMP-backed Linewise OpenAI gateway adapter, and asynchronous realization of Lean effects.
- `crates/mycode-plugin-protocol/` owns the versioned external-plugin wire contract: four-byte big-endian length followed by bounded UTF-8 JSON.
- `crates/mycode-workspace-plugin/` is the first-party workspace plugin. `read` and `grep` are always safe; Lean permission modes decide whether `write`, `edit`, and `bash` run automatically or require approval. Bash emits bounded, correlated stdout/stderr progress frames before its terminal result.
- Steer belongs to the main Lean state, unlike the independent BTW sidechain. A steer replaces an in-flight model request; during a tool it is persisted until the current result is known, then Lean synthesizes results for every skipped tool before appending the new user instruction.
- Plan and todo are built-in control state, not plugin effects. Lean owns the canonical Markdown plan, its revision and review phase, plus phased todo items. In plan mode Lean permits only configured safe tools and closed read-only commands, even under `yolo`.
- `/plan <goal>` starts read-only planning. The model updates todos and submits a Markdown plan through the built-in `plan` tool; the TUI lets the user approve, refine, cancel, or edit it with `$VISUAL`/`$EDITOR`. `/todo` round-trips the same canonical todo state through a bounded Markdown editor.
- Compaction is built-in control state. Lean preserves the complete canonical transcript, chooses user-turn cut points, persists pending compaction before emitting `request_compaction`, and stores the bounded summary plus the first kept message. Rust sends providers only the latest summary and retained suffix.
- `/compact [instructions]` runs manual context compaction. Automatic compaction uses provider-reported input tokens before a model request and after a completed turn, recovers classified context-overflow responses, and clears its trigger after success or failure to prevent retry loops.
- Permission policy is Lean-owned. `ask` permits only configured safe tools, `auto` additionally permits closed read-only commands (`pwd`, and non-dereferencing `ls` in the current directory), and `yolo` permits every declared tool.
- The TUI renders live command stdout/stderr tails while details are collapsed. `Ctrl+O` toggles the complete command and tool output.
- Transcript text is mouse-selectable while mouse-wheel scrolling remains active. A left-button drag highlights and copies through OSC 52; double-click selects a code word, `Ctrl`+double-click selects one visual line, and triple-click selects the contiguous paragraph block. Copied text excludes rounded box borders, and a click without a drag leaves the clipboard unchanged.
- The input editor keeps a Unicode grapheme cursor and a cursor-following wrapped viewport. `Left`/`Right` move by grapheme; `Alt`/`Ctrl` plus arrows and `Alt+B`/`Alt+F` move by code word; `Ctrl+B`/`Ctrl+F`, `Ctrl+A`/`Ctrl+E`, `Home`/`End`, `Delete`, `Ctrl+W`, and `Alt+Backspace` provide readline-style editing.
