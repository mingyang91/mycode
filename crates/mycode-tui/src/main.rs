#![deny(warnings)]
#![deny(clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    env,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use clap::{Parser, ValueEnum};
use crossterm::clipboard::CopyToClipboard;
use crossterm::cursor::Show;
use crossterm::{
    event::{
        DisableMouseCapture, EnableMouseCapture, Event as CrosstermEvent, EventStream, KeyCode,
        KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures_util::StreamExt;
use mycode_plugin_protocol::{
    CallToolParams, DEFAULT_MAX_FRAME_BYTES, DEFAULT_MAX_TOOL_OUTPUT_BYTES, InitializeParams,
    InitializeResult, Limits, PluginMessage, RequestEnvelope as PluginRequest,
    RequestOperation as PluginOperation, ToolListResult, ToolOutputStream, ToolProgress,
    ToolResult, ToolSpec, read_plugin_message, read_response, write_request,
};
#[cfg(unix)]
use nix::{
    sys::signal::{Signal, killpg},
    unistd::Pid,
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    buffer::Buffer,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Padding, Paragraph, Widget, Wrap},
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, BufWriter},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{Mutex, mpsc},
    task::JoinHandle,
    time::{MissedTickBehavior, interval, timeout},
};
use tokio_util::sync::CancellationToken;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
use uuid::Uuid;

const CORE_PROTOCOL_VERSION: u64 = 4;
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a careful coding agent. Use the declared tools when needed. Explain changes briefly.";
const MAX_PROVIDER_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const MAX_BTW_SIDECHAIN_BYTES: usize = 4 * 1024 * 1024;
const MAX_EDITOR_BYTES: usize = 128 * 1024;
const FRAME_INTERVAL: Duration = Duration::from_millis(16);
const COLOR_ACCENT: Color = Color::Rgb(139, 92, 246);
const COLOR_USER: Color = Color::Rgb(56, 189, 248);
const COLOR_ASSISTANT: Color = Color::Rgb(74, 222, 128);
const COLOR_TOOL: Color = Color::Rgb(250, 204, 21);
const COLOR_TEXT: Color = Color::Rgb(228, 228, 231);
const COLOR_MUTED: Color = Color::Rgb(113, 113, 122);
const COLOR_ERROR: Color = Color::Rgb(248, 113, 113);
// Human messages are deliberately rendered as black-backed blocks. Assistant prose
// uses the terminal's default background so it remains visually distinct from tools.
const COLOR_USER_BACKGROUND: Color = Color::Black;
const COLOR_ASSISTANT_BACKGROUND: Color = Color::Reset;
const COLOR_TOOL_BACKGROUND: Color = Color::Rgb(42, 36, 13);
// File contents are code-like output. Keep this deliberately plain until syntax
// highlighting is added; the black surface also separates it from command output.
const COLOR_FILE_BACKGROUND: Color = Color::Black;
const COLOR_GREP_BACKGROUND: Color = Color::Rgb(18, 34, 48);
const COLOR_ERROR_BACKGROUND: Color = Color::Rgb(49, 20, 24);
const COLOR_MESSAGE_BACKGROUND: Color = Color::Rgb(30, 30, 34);
const COLLAPSED_LIVE_TAIL_LINES: usize = 4;
const COLLAPSED_COMMAND_LINES: usize = 2;
const COLLAPSED_FILE_OUTPUT_LINES: usize = 8;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Provider {
    Openai,
    Anthropic,
    Linewise,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PermissionMode {
    Ask,
    Auto,
    Yolo,
}

impl PermissionMode {
    fn from_wire(value: &str) -> Self {
        match value {
            "auto" => Self::Auto,
            "yolo" => Self::Yolo,
            _ => Self::Ask,
        }
    }

    fn wire_value(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Auto => "auto",
            Self::Yolo => "yolo",
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "mycode-tui",
    about = "Lean 4 native coding agent with a Rust Ratatui shell"
)]
struct Args {
    #[arg(long, value_enum)]
    provider: Provider,
    #[arg(long)]
    model: String,
    #[arg(long)]
    plugin: PathBuf,
    #[arg(long)]
    git_plugin: Option<PathBuf>,
    #[arg(long)]
    core: Option<PathBuf>,
    #[arg(long)]
    session: Option<PathBuf>,
    #[arg(long)]
    btw_sidechain: Option<PathBuf>,
    #[arg(long)]
    base_url: Option<String>,
    #[arg(long, default_value = DEFAULT_SYSTEM_PROMPT)]
    system_prompt: String,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, value_enum)]
    permission_mode: Option<PermissionMode>,
    #[arg(
        long,
        default_value_t = 128_000,
        value_parser = clap::value_parser!(u64).range(1024..)
    )]
    context_window: u64,
    #[arg(
        long,
        default_value_t = 80,
        value_parser = clap::value_parser!(u8).range(1..=95)
    )]
    auto_compact_threshold_percent: u8,
    #[arg(long)]
    no_auto_compact: bool,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("terminal operation failed: {0}")]
    Terminal(#[from] std::io::Error),
    #[error("core process failed: {0}")]
    Core(#[from] CoreClientError),
    #[error("plugin process failed: {0}")]
    Plugin(#[from] PluginClientError),
    #[error("provider request failed: {0}")]
    Provider(#[from] ProviderError),
    #[error("Lean core effect `{kind}` omitted its tool call payload")]
    MissingCoreEffectCall { kind: String },
    #[error("Lean core emitted an unknown effect kind: {kind}")]
    UnknownCoreEffect { kind: String },
    #[error("session path has no parent directory")]
    SessionPathWithoutParent,
    #[error("cannot submit --prompt while restoring a {phase} session")]
    PromptWhileResuming { phase: String },
    #[error("persisted {phase} session has no current pending tool call")]
    MissingPendingTool { phase: String },
    #[error("persisted session has an unsupported phase: {phase}")]
    UnsupportedSessionPhase { phase: String },
    #[error("plugin progress carried invalid base64: {0}")]
    ToolProgressEncoding(#[from] base64::DecodeError),
    #[error("failed to access BTW sidechain {path}: {source}")]
    SidechainIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("BTW sidechain {path} was not valid JSON: {source}")]
    SidechainJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("BTW sidechain {path} exceeded {limit} bytes")]
    SidechainTooLarge { path: PathBuf, limit: usize },
    #[error("BTW sidechain {path} uses unsupported version {version}")]
    SidechainVersion { path: PathBuf, version: u32 },
    #[error("could not determine a home directory for BTW sidechain storage")]
    MissingHomeDirectory,
    #[error("external editor exited with status {status}")]
    EditorFailed { status: std::process::ExitStatus },
    #[error("edited content exceeded {limit} bytes")]
    EditedContentTooLarge { limit: usize },
    #[error("external editor produced non-UTF-8 text: {0}")]
    EditorEncoding(#[from] std::string::FromUtf8Error),
    #[error("invalid todo document at line {line}: {message}")]
    TodoDocument { line: usize, message: String },
}

#[derive(Debug, Error, PartialEq, Eq)]
enum SlashCommandError {
    #[error("usage: {0}")]
    InvalidUsage(&'static str),
    #[error("unknown slash command: /{0}")]
    Unknown(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlashCommandKind {
    Btw,
    Compact,
    Exit,
    Plan,
    Steer,
    Todo,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SlashCommandSpec {
    name: &'static str,
    usage: &'static str,
    description: &'static str,
    kind: SlashCommandKind,
    takes_arguments: bool,
}

const SLASH_COMMANDS: [SlashCommandSpec; 6] = [
    SlashCommandSpec {
        name: "btw",
        usage: "/btw <question>",
        description: "Ask outside the main conversation",
        kind: SlashCommandKind::Btw,
        takes_arguments: true,
    },
    SlashCommandSpec {
        name: "compact",
        usage: "/compact [instructions]",
        description: "Summarize old context and keep recent turns",
        kind: SlashCommandKind::Compact,
        takes_arguments: true,
    },
    SlashCommandSpec {
        name: "exit",
        usage: "/exit",
        description: "Exit MyCode cleanly",
        kind: SlashCommandKind::Exit,
        takes_arguments: false,
    },
    SlashCommandSpec {
        name: "plan",
        usage: "/plan <goal>",
        description: "Research and submit a plan for review",
        kind: SlashCommandKind::Plan,
        takes_arguments: true,
    },
    SlashCommandSpec {
        name: "steer",
        usage: "/steer <instruction>",
        description: "Redirect the active main task",
        kind: SlashCommandKind::Steer,
        takes_arguments: true,
    },
    SlashCommandSpec {
        name: "todo",
        usage: "/todo",
        description: "Edit the session todo list",
        kind: SlashCommandKind::Todo,
        takes_arguments: false,
    },
];

fn slash_command_prefix(input: &str) -> Option<&str> {
    let command = input.strip_prefix('/')?;
    if command.starts_with('/') || command.chars().any(char::is_whitespace) {
        return None;
    }
    Some(command)
}

fn slash_command_candidates(input: &str) -> impl Iterator<Item = &'static SlashCommandSpec> + '_ {
    let prefix = slash_command_prefix(input);
    SLASH_COMMANDS
        .iter()
        .filter(move |spec| prefix.is_some_and(|prefix| spec.name.starts_with(prefix)))
}

#[derive(Debug, PartialEq, Eq)]
enum SlashCommand {
    Btw(String),
    Compact(Option<String>),
    Exit,
    Plan(String),
    Steer(String),
    Todo,
}

#[derive(Debug, PartialEq, Eq)]
enum UserSubmission {
    Prompt(String),
    Command(SlashCommand),
}

fn parse_submission(text: String) -> Result<UserSubmission, SlashCommandError> {
    if let Some(literal) = text.strip_prefix("//") {
        return Ok(UserSubmission::Prompt(format!("/{literal}")));
    }
    let Some(command) = text.strip_prefix('/') else {
        return Ok(UserSubmission::Prompt(text));
    };
    let (name, arguments) = command
        .split_once(char::is_whitespace)
        .map_or((command, ""), |(name, arguments)| (name, arguments.trim()));
    let spec = SLASH_COMMANDS
        .iter()
        .find(|spec| spec.name == name)
        .ok_or_else(|| SlashCommandError::Unknown(name.to_owned()))?;
    match spec.kind {
        SlashCommandKind::Btw if arguments.is_empty() => {
            Err(SlashCommandError::InvalidUsage(spec.usage))
        }
        SlashCommandKind::Btw => Ok(UserSubmission::Command(SlashCommand::Btw(
            arguments.to_owned(),
        ))),
        SlashCommandKind::Compact => Ok(UserSubmission::Command(SlashCommand::Compact(
            if arguments.is_empty() {
                None
            } else {
                Some(arguments.to_owned())
            },
        ))),
        SlashCommandKind::Exit if arguments.is_empty() => {
            Ok(UserSubmission::Command(SlashCommand::Exit))
        }
        SlashCommandKind::Exit => Err(SlashCommandError::InvalidUsage(spec.usage)),
        SlashCommandKind::Plan if arguments.is_empty() => {
            Err(SlashCommandError::InvalidUsage(spec.usage))
        }
        SlashCommandKind::Plan => Ok(UserSubmission::Command(SlashCommand::Plan(
            arguments.to_owned(),
        ))),
        SlashCommandKind::Steer if arguments.is_empty() => {
            Err(SlashCommandError::InvalidUsage(spec.usage))
        }
        SlashCommandKind::Steer => Ok(UserSubmission::Command(SlashCommand::Steer(
            arguments.to_owned(),
        ))),
        SlashCommandKind::Todo if arguments.is_empty() => {
            Ok(UserSubmission::Command(SlashCommand::Todo))
        }
        SlashCommandKind::Todo => Err(SlashCommandError::InvalidUsage(spec.usage)),
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct CoreToolCall {
    call_id: String,
    name: String,
    arguments: Value,
}

impl CoreToolCall {
    fn display_label(&self) -> &str {
        if self.name == "git_read" || self.name == "git_write" {
            self.arguments
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or(&self.name)
        } else {
            &self.name
        }
    }

    fn transcript_summary(&self) -> String {
        if self.name == "grep" {
            let pattern = self
                .arguments
                .get("pattern")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let path = self
                .arguments
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or(".");
            let case = self
                .arguments
                .get("caseSensitive")
                .and_then(Value::as_bool)
                .map_or("", |case_sensitive| {
                    if case_sensitive {
                        ""
                    } else {
                        " · ignore case"
                    }
                });
            return format!("grep  🔎 /{pattern}/  {path}{case}");
        }
        let path_summary =
            self.arguments
                .get("path")
                .and_then(Value::as_str)
                .map(|path| match self.name.as_str() {
                    "read" => format!("read  📄 {path}"),
                    "grep" => format!("grep  🔎 {path}"),
                    "write" => format!("write  📝 {path}"),
                    "edit" => format!("edit  ✎ {path}"),
                    _ => String::new(),
                });
        if let Some(summary) = path_summary.filter(|summary| !summary.is_empty()) {
            return summary;
        }
        match self.arguments.get("command").and_then(Value::as_str) {
            Some(command) => format!("{}  {command}", self.name),
            None if self.name == "todo" => format!("{}  update task list", self.name),
            None => format!("{}  {}", self.name, self.arguments),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct CoreMessage {
    role: String,
    content: String,
    #[serde(default)]
    tool_calls: Vec<CoreToolCall>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    is_error: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct CoreTodoItem {
    content: String,
    status: String,
    #[serde(default)]
    blocker: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct CoreTodoPhase {
    name: String,
    #[serde(default)]
    tasks: Vec<CoreTodoItem>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct CorePlanState {
    enabled: bool,
    revision: u64,
    status: String,
    content: String,
}

impl Default for CorePlanState {
    fn default() -> Self {
        Self {
            enabled: false,
            revision: 0,
            status: "none".to_owned(),
            content: String::new(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct CorePendingCompaction {
    first_kept_message: u64,
    tokens_before: u64,
    #[serde(default)]
    instructions: Option<String>,
    automatic: bool,
    continue_after: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct CoreCompactionState {
    revision: u64,
    summary: String,
    first_kept_message: u64,
    tokens_before: u64,
    last_input_tokens: u64,
    #[serde(default)]
    pending: Option<CorePendingCompaction>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct CoreEvent {
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(default)]
    tool_calls: Vec<CoreToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    approved: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_error: Option<bool>,
    #[serde(default)]
    safe_tools: Vec<String>,
    #[serde(default = "default_permission_mode")]
    permission_mode: String,
    #[serde(default)]
    todos: Vec<CoreTodoPhase>,
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    automatic: bool,
}

fn default_permission_mode() -> String {
    "auto".to_owned()
}

impl CoreEvent {
    fn new(kind: &str) -> Self {
        Self {
            kind: kind.to_owned(),
            text: None,
            tool_calls: Vec::new(),
            call_id: None,
            approved: None,
            content: None,
            is_error: None,
            safe_tools: Vec::new(),
            permission_mode: "auto".to_owned(),
            todos: Vec::new(),
            input_tokens: 0,
            automatic: false,
        }
    }

    fn configure_tools(safe_tools: Vec<String>, permission_mode: PermissionMode) -> Self {
        Self {
            safe_tools,
            permission_mode: permission_mode.wire_value().to_owned(),
            ..Self::new("configure_tools")
        }
    }

    fn submit(text: String) -> Self {
        Self {
            text: Some(text),
            ..Self::new("submit")
        }
    }

    fn enter_plan(text: String) -> Self {
        Self {
            text: Some(text),
            ..Self::new("enter_plan")
        }
    }

    fn steer(text: String) -> Self {
        Self {
            text: Some(text),
            ..Self::new("steer")
        }
    }

    fn model_completed(content: String, tool_calls: Vec<CoreToolCall>, input_tokens: u64) -> Self {
        Self {
            tool_calls,
            content: Some(content),
            input_tokens,
            ..Self::new("model_completed")
        }
    }

    fn approval(call_id: String, approved: bool) -> Self {
        Self {
            call_id: Some(call_id),
            approved: Some(approved),
            ..Self::new("approval_result")
        }
    }

    fn tool_completed(call_id: String, content: String, is_error: bool) -> Self {
        Self {
            call_id: Some(call_id),
            content: Some(content),
            is_error: Some(is_error),
            ..Self::new("tool_completed")
        }
    }

    fn replace_todos(todos: Vec<CoreTodoPhase>) -> Self {
        Self {
            todos,
            ..Self::new("replace_todos")
        }
    }

    fn approve_plan() -> Self {
        Self {
            approved: Some(true),
            ..Self::new("plan_review_result")
        }
    }

    fn refine_plan(feedback: String) -> Self {
        Self {
            text: Some(feedback),
            ..Self::new("plan_review_result")
        }
    }

    fn edit_plan(content: String) -> Self {
        Self {
            content: Some(content),
            ..Self::new("plan_review_result")
        }
    }

    fn cancel_plan_review() -> Self {
        Self {
            approved: Some(false),
            ..Self::new("plan_review_result")
        }
    }

    fn start_compaction(instructions: Option<String>, input_tokens: u64, automatic: bool) -> Self {
        Self {
            text: instructions,
            input_tokens,
            automatic,
            ..Self::new("start_compaction")
        }
    }

    fn compaction_completed(summary: String) -> Self {
        Self {
            content: Some(summary),
            ..Self::new("compaction_completed")
        }
    }

    fn compaction_failed() -> Self {
        Self::new("compaction_failed")
    }

    fn abort() -> Self {
        Self::new("abort")
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CoreEffect {
    kind: String,
    #[serde(default)]
    call: Option<CoreToolCall>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CoreSnapshot {
    phase: String,
    #[serde(default)]
    messages: Vec<CoreMessage>,
    #[serde(default)]
    pending_calls: Vec<CoreToolCall>,
    current_call: u64,
    #[serde(default)]
    safe_tools: Vec<String>,
    #[serde(default = "default_permission_mode")]
    permission_mode: String,
    #[serde(default)]
    pending_steers: Vec<String>,
    #[serde(default)]
    plan: CorePlanState,
    #[serde(default)]
    todos: Vec<CoreTodoPhase>,
    #[serde(default)]
    compaction: CoreCompactionState,
}
impl Default for CoreSnapshot {
    fn default() -> Self {
        Self {
            phase: String::new(),
            messages: Vec::new(),
            pending_calls: Vec::new(),
            current_call: 0,
            safe_tools: Vec::new(),
            permission_mode: default_permission_mode(),
            pending_steers: Vec::new(),
            plan: CorePlanState::default(),
            todos: Vec::new(),
            compaction: CoreCompactionState::default(),
        }
    }
}

fn snapshot_accepts_steer(snapshot: &CoreSnapshot) -> bool {
    matches!(
        snapshot.phase.as_str(),
        "waiting_model" | "waiting_approval" | "waiting_tool"
    )
}

fn startup_permission_mode(
    restored_session: bool,
    requested: Option<PermissionMode>,
    snapshot: &CoreSnapshot,
) -> PermissionMode {
    requested.unwrap_or_else(|| {
        if restored_session {
            PermissionMode::from_wire(&snapshot.permission_mode)
        } else {
            PermissionMode::Auto
        }
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CoreResponse {
    version: u64,
    request_id: String,
    ok: bool,
    snapshot: CoreSnapshot,
    #[serde(default)]
    effects: Vec<CoreEffect>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CoreRequest<'a> {
    version: u64,
    request_id: &'a str,
    op: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    event: Option<&'a CoreEvent>,
}

#[derive(Debug, Error)]
enum CoreClientError {
    #[error("failed to start the Lean core: {0}")]
    Spawn(#[source] std::io::Error),
    #[error("failed to communicate with the Lean core: {0}")]
    Io(#[source] std::io::Error),
    #[error("Lean core emitted malformed JSON: {0}")]
    Json(#[source] serde_json::Error),
    #[error("Lean core closed stdout before replying")]
    UnexpectedEof,
    #[error("Lean core response id did not match the request")]
    CorrelationMismatch,
    #[error("Lean core response uses an unsupported version {0}")]
    VersionMismatch(u64),
    #[error("Lean core rejected the event: {0}")]
    Rejected(String),
    #[error("Lean core did not provide piped standard I/O")]
    MissingPipe,
}

struct CoreClient {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl CoreClient {
    async fn spawn(core: &Path, session: Option<&Path>) -> Result<Self, CoreClientError> {
        let mut command = Command::new(core);
        if let Some(session) = session {
            command.arg("--session").arg(session);
        }
        let mut child = command
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(CoreClientError::Spawn)?;
        let input = child.stdin.take().ok_or(CoreClientError::MissingPipe)?;
        let output = child.stdout.take().ok_or(CoreClientError::MissingPipe)?;
        Ok(Self {
            child,
            input: BufWriter::new(input),
            output: BufReader::new(output),
        })
    }

    async fn event(&mut self, event: &CoreEvent) -> Result<CoreResponse, CoreClientError> {
        self.request("event", Some(event)).await
    }

    async fn snapshot(&mut self) -> Result<CoreResponse, CoreClientError> {
        self.request("snapshot", None).await
    }

    async fn shutdown(&mut self) -> Result<(), CoreClientError> {
        let _ = self.request("shutdown", None).await?;
        let _ = self.child.wait().await.map_err(CoreClientError::Io)?;
        Ok(())
    }

    async fn request(
        &mut self,
        op: &str,
        event: Option<&CoreEvent>,
    ) -> Result<CoreResponse, CoreClientError> {
        let request_id = Uuid::new_v4().simple().to_string();
        let encoded = serde_json::to_string(&CoreRequest {
            version: CORE_PROTOCOL_VERSION,
            request_id: &request_id,
            op,
            event,
        })
        .map_err(CoreClientError::Json)?;
        self.input
            .write_all(encoded.as_bytes())
            .await
            .map_err(CoreClientError::Io)?;
        self.input
            .write_all(b"\n")
            .await
            .map_err(CoreClientError::Io)?;
        self.input.flush().await.map_err(CoreClientError::Io)?;
        let mut response_line = String::new();
        let size = self
            .output
            .read_line(&mut response_line)
            .await
            .map_err(CoreClientError::Io)?;
        if size == 0 {
            return Err(CoreClientError::UnexpectedEof);
        }
        let response: CoreResponse =
            serde_json::from_str(&response_line).map_err(CoreClientError::Json)?;
        if response.version != CORE_PROTOCOL_VERSION {
            return Err(CoreClientError::VersionMismatch(response.version));
        }
        if response.request_id != request_id {
            return Err(CoreClientError::CorrelationMismatch);
        }
        if response.ok {
            Ok(response)
        } else {
            Err(CoreClientError::Rejected(
                response
                    .error
                    .unwrap_or_else(|| "unknown core error".to_owned()),
            ))
        }
    }
}

#[derive(Debug, Error)]
enum PluginClientError {
    #[error("failed to start plugin: {0}")]
    Spawn(#[source] std::io::Error),
    #[error("plugin did not provide piped standard I/O")]
    MissingPipe,
    #[error("plugin protocol failed: {0}")]
    Protocol(#[from] mycode_plugin_protocol::ProtocolError),
    #[error("plugin reply id did not match the request")]
    CorrelationMismatch,
    #[error("plugin did not support protocol version one")]
    UnsupportedVersion,
    #[error("plugin tool catalogue contained a duplicate name")]
    DuplicateTool,
    #[error("plugin tool call exceeded its deadline")]
    DeadlineExceeded,
    #[error("plugin was retired after a transport failure or deadline")]
    Retired,
    #[error("plugin tool catalogue changed after restart")]
    CatalogueChanged,
    #[error("no plugin declared tool {0}")]
    UnknownTool(String),
    #[error("multiple plugins declared tool {0}")]
    DuplicateToolAcrossPlugins(String),
}

struct PluginClient {
    child: Child,
    input: ChildStdin,
    output: ChildStdout,
    tools: Vec<ToolSpec>,
    available: bool,
}

impl PluginClient {
    async fn spawn(executable: &Path) -> Result<Self, PluginClientError> {
        let mut command = Command::new(executable);
        #[cfg(unix)]
        command.process_group(0);
        let mut child = command
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(PluginClientError::Spawn)?;
        let mut input = child.stdin.take().ok_or(PluginClientError::MissingPipe)?;
        let mut output = child.stdout.take().ok_or(PluginClientError::MissingPipe)?;
        let init_id = "init_1".to_owned();
        write_request(
            &mut input,
            &PluginRequest {
                v: 1,
                id: init_id.clone(),
                operation: PluginOperation::Initialize(InitializeParams {
                    host: mycode_plugin_protocol::HostIdentity {
                        name: "mycode".to_owned(),
                        version: env!("CARGO_PKG_VERSION").to_owned(),
                    },
                    limits: Limits::default(),
                }),
            },
        )
        .await?;
        let init = read_response(&mut output, DEFAULT_MAX_FRAME_BYTES).await?;
        if init.id != init_id {
            return Err(PluginClientError::CorrelationMismatch);
        }
        let init: InitializeResult = init.into_result()?;
        if !(init.protocol.min_version..=init.protocol.max_version).contains(&1) {
            return Err(PluginClientError::UnsupportedVersion);
        }
        let list_id = "tools_1".to_owned();
        write_request(
            &mut input,
            &PluginRequest {
                v: 1,
                id: list_id.clone(),
                operation: PluginOperation::ListTools(mycode_plugin_protocol::EmptyParams {}),
            },
        )
        .await?;
        let list = read_response(&mut output, DEFAULT_MAX_FRAME_BYTES).await?;
        if list.id != list_id {
            return Err(PluginClientError::CorrelationMismatch);
        }
        let tools: ToolListResult = list.into_result()?;
        let mut names = BTreeMap::new();
        for tool in &tools.tools {
            if names.insert(tool.name.clone(), ()).is_some() {
                return Err(PluginClientError::DuplicateTool);
            }
        }
        Ok(Self {
            child,
            input,
            output,
            tools: tools.tools,
            available: true,
        })
    }

    async fn call<F>(
        &mut self,
        call: &CoreToolCall,
        mut on_progress: F,
    ) -> Result<ToolResult, PluginClientError>
    where
        F: FnMut(ToolProgress),
    {
        if !self.available {
            return Err(PluginClientError::Retired);
        }
        let request_id = format!("call_{}", Uuid::new_v4().simple());
        if let Err(error) = write_request(
            &mut self.input,
            &PluginRequest {
                v: 1,
                id: request_id.clone(),
                operation: PluginOperation::CallTool(CallToolParams {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                }),
            },
        )
        .await
        {
            self.retire().await;
            return Err(error.into());
        }
        loop {
            let message = match timeout(
                Duration::from_secs(300),
                read_plugin_message(&mut self.output, DEFAULT_MAX_FRAME_BYTES),
            )
            .await
            {
                Ok(Ok(message)) => message,
                Ok(Err(error)) => {
                    self.retire().await;
                    return Err(error.into());
                }
                Err(_) => {
                    self.retire().await;
                    return Err(PluginClientError::DeadlineExceeded);
                }
            };
            match message {
                PluginMessage::Progress(progress) => {
                    if progress.id != request_id {
                        self.retire().await;
                        return Err(PluginClientError::CorrelationMismatch);
                    }
                    on_progress(progress.progress);
                }
                PluginMessage::Response(response) => {
                    if response.id != request_id {
                        self.retire().await;
                        return Err(PluginClientError::CorrelationMismatch);
                    }
                    return response.into_result().map_err(PluginClientError::from);
                }
            }
        }
    }

    async fn retire(&mut self) {
        self.available = false;
        self.kill_tree().await;
    }

    async fn kill_tree(&mut self) {
        #[cfg(unix)]
        if let Some(pid) = self.child.id().and_then(|pid| i32::try_from(pid).ok()) {
            let _ = killpg(Pid::from_raw(pid), Signal::SIGKILL);
            let _ = self.child.wait().await;
            return;
        }
        let _ = self.child.kill().await;
    }

    async fn shutdown(&mut self) {
        if !self.available {
            return;
        }
        let request = PluginRequest {
            v: 1,
            id: "shutdown_1".to_owned(),
            operation: PluginOperation::Shutdown(mycode_plugin_protocol::EmptyParams {}),
        };
        let clean_shutdown = match write_request(&mut self.input, &request).await {
            Ok(()) => match timeout(
                Duration::from_secs(2),
                read_response(&mut self.output, DEFAULT_MAX_FRAME_BYTES),
            )
            .await
            {
                Ok(Ok(response)) => {
                    response.id == request.id && response.into_result::<Value>().is_ok()
                }
                Ok(Err(_)) | Err(_) => false,
            },
            Err(_) => false,
        };
        self.available = false;
        if clean_shutdown
            && matches!(
                timeout(Duration::from_secs(2), self.child.wait()).await,
                Ok(Ok(_))
            )
        {
            return;
        }
        self.kill_tree().await;
    }
}

struct ManagedPlugin {
    path: PathBuf,
    expected_tool_names: Vec<String>,
    client: PluginClient,
}

struct PluginManager {
    plugins: Vec<ManagedPlugin>,
    routes: BTreeMap<String, usize>,
    model_tools: Vec<ToolSpec>,
}

impl PluginManager {
    async fn spawn(workspace: &Path, git: &Path) -> Result<Self, PluginClientError> {
        let declarations = [(workspace, true), (git, false)];
        let mut plugins = Vec::with_capacity(declarations.len());
        let mut routes = BTreeMap::new();
        let mut model_tools = Vec::new();
        for (path, expose_to_model) in declarations {
            let client = PluginClient::spawn(path).await?;
            let plugin_index = plugins.len();
            let expected_tool_names = client.tools.iter().map(|tool| tool.name.clone()).collect();
            for tool in &client.tools {
                if routes.insert(tool.name.clone(), plugin_index).is_some() {
                    return Err(PluginClientError::DuplicateToolAcrossPlugins(
                        tool.name.clone(),
                    ));
                }
                if expose_to_model {
                    model_tools.push(tool.clone());
                }
            }
            plugins.push(ManagedPlugin {
                path: path.to_path_buf(),
                expected_tool_names,
                client,
            });
        }
        Ok(Self {
            plugins,
            routes,
            model_tools,
        })
    }

    fn model_tools(&self) -> &[ToolSpec] {
        &self.model_tools
    }

    fn has_tool(&self, name: &str) -> bool {
        self.routes.contains_key(name)
    }

    async fn ensure_plugin(&mut self, index: usize) -> Result<(), PluginClientError> {
        if self.plugins[index].client.available {
            return Ok(());
        }
        let path = self.plugins[index].path.clone();
        let expected_names = self.plugins[index].expected_tool_names.clone();
        let replacement = PluginClient::spawn(&path).await?;
        let replacement_names: Vec<String> = replacement
            .tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect();
        if replacement_names != expected_names {
            return Err(PluginClientError::CatalogueChanged);
        }
        self.plugins[index].client = replacement;
        Ok(())
    }

    async fn call<F>(
        &mut self,
        call: &CoreToolCall,
        on_progress: F,
    ) -> Result<ToolResult, PluginClientError>
    where
        F: FnMut(ToolProgress),
    {
        let index = self
            .routes
            .get(&call.name)
            .copied()
            .ok_or_else(|| PluginClientError::UnknownTool(call.name.clone()))?;
        self.ensure_plugin(index).await?;
        self.plugins[index].client.call(call, on_progress).await
    }

    async fn retire(&mut self) {
        for plugin in &mut self.plugins {
            plugin.client.retire().await;
        }
    }

    async fn shutdown(&mut self) {
        for plugin in &mut self.plugins {
            plugin.client.shutdown().await;
        }
    }
}

#[derive(Debug, Error)]
enum ProviderError {
    #[error("HTTP transport failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("provider returned HTTP {status}: {body}")]
    Status {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("provider response was not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("provider response exceeded {limit} bytes")]
    ResponseTooLarge { limit: usize },
    #[error("provider response did not contain a usable completion")]
    MissingCompletion,
    #[error("provider supplied malformed tool arguments")]
    InvalidToolArguments,
    #[error("provider generated an undeclared tool name")]
    UnknownTool,
    #[error("missing {provider} API key in {environment}")]
    MissingApiKey {
        provider: &'static str,
        environment: &'static str,
    },
    #[error("OMP token lookup timed out")]
    OmpTokenTimeout,
    #[error("failed to execute `omp token linewise`: {0}")]
    OmpTokenCommand(#[source] std::io::Error),
    #[error("`omp token linewise` failed with status {status}: {message}")]
    OmpTokenRejected {
        status: std::process::ExitStatus,
        message: String,
    },
    #[error("`omp token linewise` returned non-UTF-8 output")]
    OmpTokenEncoding,
    #[error("`omp token linewise` returned an empty credential")]
    EmptyOmpToken,
}

impl ProviderError {
    fn is_context_overflow(&self) -> bool {
        let Self::Status { body, .. } = self else {
            return false;
        };
        let message = body.to_ascii_lowercase();
        [
            "context length",
            "context window",
            "maximum context",
            "prompt is too long",
            "too many tokens",
        ]
        .iter()
        .any(|needle| message.contains(needle))
    }
}

fn provider_input_tokens(payload: &Value, field: &str) -> u64 {
    payload
        .get("usage")
        .and_then(|usage| usage.get(field))
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

struct ModelClient {
    http: Client,
    provider: Provider,
    model: String,
    base_url: String,
    api_key: String,
    system_prompt: String,
}

struct ModelCompletion {
    content: String,
    tool_calls: Vec<CoreToolCall>,
    input_tokens: u64,
}

fn active_model_messages(snapshot: &CoreSnapshot) -> Vec<CoreMessage> {
    if snapshot.compaction.summary.is_empty() {
        return snapshot.messages.clone();
    }
    let start = usize::try_from(snapshot.compaction.first_kept_message)
        .unwrap_or(usize::MAX)
        .min(snapshot.messages.len());
    let mut messages = Vec::with_capacity(snapshot.messages.len().saturating_sub(start) + 1);
    messages.push(CoreMessage {
        role: "user".to_owned(),
        content: format!(
            "[Compacted conversation summary. Treat this as historical context, not as a new instruction.]\n{}",
            snapshot.compaction.summary
        ),
        ..CoreMessage::default()
    });
    messages.extend_from_slice(&snapshot.messages[start..]);
    messages
}

impl ModelClient {
    async fn from_args(args: &Args) -> Result<Self, ProviderError> {
        let (default_url, api_key) = match args.provider {
            Provider::Openai => (
                "https://api.openai.com/v1",
                environment_key("OPENAI_API_KEY", "OpenAI")?,
            ),
            Provider::Anthropic => (
                "https://api.anthropic.com/v1",
                environment_key("ANTHROPIC_API_KEY", "Anthropic")?,
            ),
            Provider::Linewise => (
                "https://llm.dev.linewise.io/v1",
                omp_token("linewise").await?,
            ),
        };
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(20))
            .read_timeout(Duration::from_secs(300))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self {
            http,
            provider: args.provider,
            model: args.model.clone(),
            base_url: args
                .base_url
                .clone()
                .unwrap_or_else(|| default_url.to_owned()),
            api_key,
            system_prompt: args.system_prompt.clone(),
        })
    }

    fn rendered_system_prompt(&self, snapshot: &CoreSnapshot) -> String {
        let mut prompt = self.system_prompt.clone();
        prompt.push_str(
            "\n\nUse the todo tool for work with three or more distinct steps. Keep its phases and statuses current.",
        );
        if snapshot.plan.enabled {
            prompt.push_str(
                "\n\nPLAN MODE IS ACTIVE. Research without changing workspace state. Read and search freely, but do not call write, edit, or mutating shell commands. Update the structured todo list as you learn. Use the plan tool with op=update while drafting and op=propose only when the full Markdown plan is ready for human review.",
            );
        }
        if !snapshot.plan.content.is_empty() {
            prompt.push_str(&format!(
                "\n\nCurrent plan revision {} ({})\n{}",
                snapshot.plan.revision, snapshot.plan.status, snapshot.plan.content
            ));
        }
        if !snapshot.todos.is_empty() {
            prompt.push_str("\n\nCurrent todo state:");
            for phase in &snapshot.todos {
                prompt.push_str(&format!("\n## {}", phase.name));
                for task in &phase.tasks {
                    prompt.push_str(&format!("\n- [{}] {}", task.status, task.content));
                    if let Some(blocker) = &task.blocker {
                        prompt.push_str(&format!(" ({blocker})"));
                    }
                }
            }
        }
        prompt
    }

    async fn complete(
        &self,
        snapshot: &CoreSnapshot,
        tools: &[ToolSpec],
        declared_tools: &[ToolSpec],
    ) -> Result<ModelCompletion, ProviderError> {
        let system_prompt = self.rendered_system_prompt(snapshot);
        let completion = match self.provider {
            Provider::Openai | Provider::Linewise => {
                self.complete_openai(snapshot, tools, &system_prompt)
                    .await?
            }
            Provider::Anthropic => {
                self.complete_anthropic(snapshot, tools, &system_prompt)
                    .await?
            }
        };
        if completion
            .tool_calls
            .iter()
            .any(|call| !declared_tools.iter().any(|tool| tool.name == call.name))
        {
            return Err(ProviderError::UnknownTool);
        }
        Ok(completion)
    }

    async fn complete_text_only(
        &self,
        snapshot: &CoreSnapshot,
        sidechain: &[LocalTranscriptEntry],
        question: &str,
    ) -> Result<String, ProviderError> {
        let mut side_snapshot = snapshot.clone();
        side_snapshot.messages.push(CoreMessage {
            role: "user".to_owned(),
            content: "[BTW sidechain. Keep this discussion separate from the main task. Answer without using tools or changing the main task.]".to_owned(),
            ..CoreMessage::default()
        });
        for entry in sidechain {
            side_snapshot.messages.push(CoreMessage {
                role: "user".to_owned(),
                content: entry.question.clone(),
                ..CoreMessage::default()
            });
            side_snapshot.messages.push(CoreMessage {
                role: "assistant".to_owned(),
                content: entry.answer.clone(),
                ..CoreMessage::default()
            });
        }
        side_snapshot.messages.push(CoreMessage {
            role: "user".to_owned(),
            content: question.to_owned(),
            ..CoreMessage::default()
        });
        let completion = self.complete(&side_snapshot, &[], &[]).await?;
        Ok(completion.content)
    }

    async fn summarize_compaction(&self, snapshot: &CoreSnapshot) -> Result<String, ProviderError> {
        let pending = snapshot
            .compaction
            .pending
            .as_ref()
            .ok_or(ProviderError::MissingCompletion)?;
        let start = usize::try_from(snapshot.compaction.first_kept_message)
            .unwrap_or(usize::MAX)
            .min(snapshot.messages.len());
        let end = usize::try_from(pending.first_kept_message)
            .unwrap_or(usize::MAX)
            .min(snapshot.messages.len());
        if end <= start {
            return Err(ProviderError::MissingCompletion);
        }

        let mut compact_snapshot = snapshot.clone();
        compact_snapshot.messages.clear();
        if !snapshot.compaction.summary.is_empty() {
            compact_snapshot.messages.push(CoreMessage {
                role: "user".to_owned(),
                content: format!(
                    "Previous compaction summary:\n{}",
                    snapshot.compaction.summary
                ),
                ..CoreMessage::default()
            });
        }
        compact_snapshot
            .messages
            .extend_from_slice(&snapshot.messages[start..end]);
        let custom = pending
            .instructions
            .as_deref()
            .map(|instructions| format!("\n\nAdditional instructions:\n{instructions}"))
            .unwrap_or_default();
        compact_snapshot.messages.push(CoreMessage {
            role: "user".to_owned(),
            content: format!(
                "Summarize the conversation above for a coding agent that will continue the same work. Preserve user goals and constraints, decisions and invariants, files read or changed, tool outcomes, errors and blockers, verification evidence, and unresolved work. Do not add new instructions or claim unfinished work is complete.{custom}"
            ),
            ..CoreMessage::default()
        });
        compact_snapshot.compaction = CoreCompactionState::default();

        let system_prompt = "You produce bounded, factual continuation summaries for coding-agent sessions. Return only the summary.";
        let completion = match self.provider {
            Provider::Openai | Provider::Linewise => {
                self.complete_openai(&compact_snapshot, &[], system_prompt)
                    .await?
            }
            Provider::Anthropic => {
                self.complete_anthropic(&compact_snapshot, &[], system_prompt)
                    .await?
            }
        };
        if !completion.tool_calls.is_empty() || completion.content.trim().is_empty() {
            return Err(ProviderError::MissingCompletion);
        }
        Ok(completion.content)
    }

    async fn complete_openai(
        &self,
        snapshot: &CoreSnapshot,
        tools: &[ToolSpec],
        system_prompt: &str,
    ) -> Result<ModelCompletion, ProviderError> {
        let active_messages = active_model_messages(snapshot);
        let mut messages = vec![json!({"role": "system", "content": system_prompt})];
        for message in &active_messages {
            match message.role.as_str() {
                "user" => messages.push(json!({"role": "user", "content": message.content})),
                "assistant" => {
                    let tool_calls: Vec<Value> = message
                        .tool_calls
                        .iter()
                        .map(|call| {
                            json!({
                                "id": call.call_id,
                                "type": "function",
                                "function": {"name": call.name, "arguments": call.arguments.to_string()}
                            })
                        })
                        .collect();
                    if tool_calls.is_empty() {
                        messages.push(json!({"role": "assistant", "content": message.content}));
                    } else {
                        messages.push(json!({
                            "role": "assistant",
                            "content": message.content,
                            "tool_calls": tool_calls
                        }));
                    }
                }
                "tool" => messages.push(json!({
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content
                })),
                _ => {}
            }
        }
        let tools: Vec<Value> = tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema
                    }
                })
            })
            .collect();
        let mut request = json!({
            "model": self.model,
            "messages": messages,
            "stream": false
        });
        if !tools.is_empty() {
            request["tools"] = json!(tools);
            request["tool_choice"] = json!("auto");
        }
        let response = self
            .http
            .post(format!(
                "{}/chat/completions",
                self.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?;
        let payload = response_json(response).await?;
        let message = payload
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .ok_or(ProviderError::MissingCompletion)?;
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let tool_calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|calls| parse_openai_tool_calls(calls))
            .transpose()?
            .unwrap_or_default();
        let input_tokens = provider_input_tokens(&payload, "prompt_tokens");
        Ok(ModelCompletion {
            content,
            tool_calls,
            input_tokens,
        })
    }

    async fn complete_anthropic(
        &self,
        snapshot: &CoreSnapshot,
        tools: &[ToolSpec],
        system_prompt: &str,
    ) -> Result<ModelCompletion, ProviderError> {
        let active_messages = active_model_messages(snapshot);
        let messages = anthropic_messages(&active_messages);
        let tools: Vec<Value> = tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema
                })
            })
            .collect();
        let mut request = json!({
            "model": self.model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages
        });
        if !tools.is_empty() {
            request["tools"] = json!(tools);
        }
        let response = self
            .http
            .post(format!("{}/messages", self.base_url.trim_end_matches('/')))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&request)
            .send()
            .await?;
        let payload = response_json(response).await?;
        let blocks = payload
            .get("content")
            .and_then(Value::as_array)
            .ok_or(ProviderError::MissingCompletion)?;
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(fragment) = block.get("text").and_then(Value::as_str) {
                        text.push_str(fragment);
                    }
                }
                Some("tool_use") => {
                    let id = block
                        .get("id")
                        .and_then(Value::as_str)
                        .ok_or(ProviderError::MissingCompletion)?;
                    let name = block
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or(ProviderError::MissingCompletion)?;
                    let arguments = block
                        .get("input")
                        .cloned()
                        .ok_or(ProviderError::InvalidToolArguments)?;
                    if !arguments.is_object() {
                        return Err(ProviderError::InvalidToolArguments);
                    }
                    tool_calls.push(CoreToolCall {
                        call_id: id.to_owned(),
                        name: name.to_owned(),
                        arguments,
                    });
                }
                _ => {}
            }
        }
        let input_tokens = provider_input_tokens(&payload, "input_tokens");
        Ok(ModelCompletion {
            content: text,
            tool_calls,
            input_tokens,
        })
    }
}
fn environment_key(
    environment: &'static str,
    provider: &'static str,
) -> Result<String, ProviderError> {
    env::var(environment).map_err(|_| ProviderError::MissingApiKey {
        provider,
        environment,
    })
}

async fn omp_token(provider: &str) -> Result<String, ProviderError> {
    let output = timeout(
        Duration::from_secs(10),
        Command::new("omp").arg("token").arg(provider).output(),
    )
    .await
    .map_err(|_| ProviderError::OmpTokenTimeout)?
    .map_err(ProviderError::OmpTokenCommand)?;
    if !output.status.success() {
        return Err(ProviderError::OmpTokenRejected {
            status: output.status,
            message: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    let token = String::from_utf8(output.stdout).map_err(|_| ProviderError::OmpTokenEncoding)?;
    let token = token.trim().to_owned();
    if token.is_empty() {
        Err(ProviderError::EmptyOmpToken)
    } else {
        Ok(token)
    }
}

async fn response_json(response: reqwest::Response) -> Result<Value, ProviderError> {
    let status = response.status();
    let mut body = Vec::with_capacity(8192);
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if body.len() + chunk.len() > MAX_PROVIDER_RESPONSE_BYTES {
            return Err(ProviderError::ResponseTooLarge {
                limit: MAX_PROVIDER_RESPONSE_BYTES,
            });
        }
        body.extend_from_slice(&chunk);
    }
    if !status.is_success() {
        return Err(ProviderError::Status {
            status,
            body: String::from_utf8_lossy(&body).into_owned(),
        });
    }
    Ok(serde_json::from_slice(&body)?)
}

fn parse_openai_tool_calls(calls: &[Value]) -> Result<Vec<CoreToolCall>, ProviderError> {
    calls
        .iter()
        .map(|call| {
            let call_id = call
                .get("id")
                .and_then(Value::as_str)
                .ok_or(ProviderError::MissingCompletion)?;
            let function = call
                .get("function")
                .ok_or(ProviderError::MissingCompletion)?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or(ProviderError::MissingCompletion)?;
            let arguments_text = function
                .get("arguments")
                .and_then(Value::as_str)
                .ok_or(ProviderError::InvalidToolArguments)?;
            let arguments: Value = serde_json::from_str(arguments_text)?;
            if !arguments.is_object() {
                return Err(ProviderError::InvalidToolArguments);
            }
            Ok(CoreToolCall {
                call_id: call_id.to_owned(),
                name: name.to_owned(),
                arguments,
            })
        })
        .collect()
}

fn anthropic_messages(messages: &[CoreMessage]) -> Vec<Value> {
    let mut result: Vec<Value> = Vec::new();
    for message in messages {
        let next = match message.role.as_str() {
            "user" => json!({
                "role": "user",
                "content": [{"type": "text", "text": message.content}]
            }),
            "assistant" => {
                let mut content = Vec::new();
                if !message.content.is_empty() {
                    content.push(json!({"type": "text", "text": message.content}));
                }
                for call in &message.tool_calls {
                    content.push(json!({
                        "type": "tool_use",
                        "id": call.call_id,
                        "name": call.name,
                        "input": call.arguments
                    }));
                }
                json!({"role": "assistant", "content": content})
            }
            "tool" => json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                    "is_error": message.is_error
                }]
            }),
            _ => continue,
        };
        if matches!(next.get("role"), Some(Value::String(role)) if role == "user")
            && let Some(last) = result.last_mut()
            && last.get("role") == next.get("role")
            && let (Some(Value::Array(existing)), Some(Value::Array(mut addition))) =
                (last.get_mut("content"), next.get("content").cloned())
        {
            existing.append(&mut addition);
            continue;
        }
        result.push(next);
    }
    result
}

struct TranscriptLayoutCache {
    revision: u64,
    width: u16,
    lines: Vec<Line<'static>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
struct LocalTranscriptEntry {
    question: String,
    answer: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SidechainFile {
    version: u32,
    entries: Vec<LocalTranscriptEntry>,
}

struct BtwSidechainStore {
    path: PathBuf,
    entries: Vec<LocalTranscriptEntry>,
}

impl BtwSidechainStore {
    async fn load(path: PathBuf) -> Result<Self, AppError> {
        let mut file = match tokio::fs::File::open(&path).await {
            Ok(file) => file,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self {
                    path,
                    entries: Vec::new(),
                });
            }
            Err(source) => {
                return Err(AppError::SidechainIo { path, source });
            }
        };
        let mut bytes = Vec::with_capacity(8192);
        let mut chunk = [0_u8; 8192];
        loop {
            let count = file
                .read(&mut chunk)
                .await
                .map_err(|source| AppError::SidechainIo {
                    path: path.clone(),
                    source,
                })?;
            if count == 0 {
                break;
            }
            if bytes.len() + count > MAX_BTW_SIDECHAIN_BYTES {
                return Err(AppError::SidechainTooLarge {
                    path,
                    limit: MAX_BTW_SIDECHAIN_BYTES,
                });
            }
            bytes.extend_from_slice(&chunk[..count]);
        }
        let stored: SidechainFile =
            serde_json::from_slice(&bytes).map_err(|source| AppError::SidechainJson {
                path: path.clone(),
                source,
            })?;
        if stored.version != 1 {
            return Err(AppError::SidechainVersion {
                path,
                version: stored.version,
            });
        }
        Ok(Self {
            path,
            entries: stored.entries,
        })
    }

    async fn append(&mut self, entry: LocalTranscriptEntry) -> Result<(), AppError> {
        self.entries.push(entry);
        if let Err(error) = self.save().await {
            self.entries.pop();
            return Err(error);
        }
        Ok(())
    }

    async fn save(&self) -> Result<(), AppError> {
        let payload = serde_json::to_vec(&SidechainFile {
            version: 1,
            entries: self.entries.clone(),
        })
        .map_err(|source| AppError::SidechainJson {
            path: self.path.clone(),
            source,
        })?;
        if payload.len() > MAX_BTW_SIDECHAIN_BYTES {
            return Err(AppError::SidechainTooLarge {
                path: self.path.clone(),
                limit: MAX_BTW_SIDECHAIN_BYTES,
            });
        }
        let parent = self.path.parent().ok_or_else(|| AppError::SidechainIo {
            path: self.path.clone(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "sidechain path has no parent",
            ),
        })?;
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|source| AppError::SidechainIo {
                path: self.path.clone(),
                source,
            })?;
        let file_name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("btw-sidechain");
        let temporary = self
            .path
            .with_file_name(format!(".{file_name}.{}.tmp", Uuid::new_v4().simple()));
        let result = async {
            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .await?;
            file.write_all(&payload).await?;
            file.flush().await?;
            file.sync_all().await?;
            drop(file);
            tokio::fs::rename(&temporary, &self.path).await
        }
        .await;
        if let Err(source) = result {
            let _ = tokio::fs::remove_file(&temporary).await;
            return Err(AppError::SidechainIo {
                path: self.path.clone(),
                source,
            });
        }
        Ok(())
    }
}

fn stable_path_hash(path: &Path) -> u64 {
    path.as_os_str()
        .as_encoded_bytes()
        .iter()
        .fold(0xcbf29ce484222325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}

fn btw_sidechain_path(
    configured: Option<&Path>,
    session: Option<&Path>,
) -> Result<PathBuf, AppError> {
    if let Some(path) = configured {
        return Ok(path.to_path_buf());
    }
    if let Some(session) = session {
        let mut sidecar = session.as_os_str().to_os_string();
        sidecar.push(".btw.json");
        return Ok(PathBuf::from(sidecar));
    }
    let home = env::var_os("MYCODE_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".mycode")))
        .ok_or(AppError::MissingHomeDirectory)?;
    let workspace = env::current_dir()?.canonicalize()?;
    Ok(home
        .join("sidechains")
        .join(format!("{:016x}.json", stable_path_hash(&workspace))))
}

fn push_wrapped_line(lines: &mut Vec<Line<'static>>, text: &str, style: Style, width: u16) {
    if width == 0 {
        return;
    }
    for source_line in text.split('\n') {
        if source_line.is_empty() {
            lines.push(Line::default());
            continue;
        }
        for wrapped in textwrap::wrap(source_line, usize::from(width)) {
            lines.push(Line::from(Span::styled(wrapped.into_owned(), style)));
        }
    }
}
enum TranscriptContentPart {
    Text(String),
    Code(String),
}

fn transcript_content_parts(content: &str) -> Vec<TranscriptContentPart> {
    let mut parts = Vec::new();
    let mut buffer = Vec::new();
    let mut in_code = false;
    for line in content.split('\n') {
        if line.trim_start().starts_with("```") {
            let buffered = buffer.join("\n");
            if in_code {
                parts.push(TranscriptContentPart::Code(buffered));
            } else if !buffered.is_empty() {
                parts.push(TranscriptContentPart::Text(buffered));
            }
            buffer.clear();
            in_code = !in_code;
        } else {
            buffer.push(line);
        }
    }
    let buffered = buffer.join("\n");
    if in_code {
        parts.push(TranscriptContentPart::Code(buffered));
    } else if !buffered.is_empty() {
        parts.push(TranscriptContentPart::Text(buffered));
    }
    parts
}

fn fill_transcript_background(
    mut line: Line<'static>,
    background: Color,
    width: u16,
) -> Line<'static> {
    let width = usize::from(width);
    let occupied = line.width();
    if occupied < width {
        line.spans.push(Span::raw(" ".repeat(width - occupied)));
    }
    line.style(Style::default().bg(background))
}

fn push_compact_text(
    lines: &mut Vec<Line<'static>>,
    text: &str,
    marker: &'static str,
    marker_color: Color,
    text_color: Color,
    background: Color,
    width: u16,
) {
    if width == 0 {
        return;
    }
    let available = usize::from(width).saturating_sub(2).max(1);
    let marker_style = Style::default()
        .fg(marker_color)
        .add_modifier(Modifier::BOLD);
    let text_style = Style::default().fg(text_color);
    let mut first = true;
    for source_line in text.split('\n') {
        if source_line.is_empty() {
            let prefix = if first { marker } else { "  " };
            first = false;
            lines.push(fill_transcript_background(
                Line::from(vec![
                    Span::styled(prefix, marker_style),
                    Span::styled(String::new(), text_style),
                ]),
                background,
                width,
            ));
            continue;
        }
        for wrapped in textwrap::wrap(source_line, available) {
            let prefix = if first { marker } else { "  " };
            first = false;
            lines.push(fill_transcript_background(
                Line::from(vec![
                    Span::styled(prefix, marker_style),
                    Span::styled(wrapped.into_owned(), text_style),
                ]),
                background,
                width,
            ));
        }
    }
}
fn push_transcript_gap(lines: &mut Vec<Line<'static>>) {
    if lines.last().is_some_and(|line| !line.spans.is_empty()) {
        lines.push(Line::default());
    }
}

fn push_human_message(
    lines: &mut Vec<Line<'static>>,
    content: &str,
    marker: &'static str,
    marker_color: Color,
    text_color: Color,
    background: Color,
    width: u16,
) {
    let mut marked = false;
    for part in transcript_content_parts(content) {
        match part {
            TranscriptContentPart::Text(text) => {
                push_compact_text(
                    lines,
                    &text,
                    if marked { "  " } else { marker },
                    marker_color,
                    text_color,
                    background,
                    width,
                );
                push_transcript_gap(lines);
                marked = true;
            }
            TranscriptContentPart::Code(code) => {
                let mut block = Vec::new();
                push_wrapped_line(
                    &mut block,
                    &code,
                    Style::default().fg(text_color),
                    transcript_block_content_width(width),
                );
                push_transcript_block(lines, block, background, marker_color, width);
            }
        }
    }
}

fn indented_wrapped_lines(text: &str, style: Style, width: u16) -> Vec<Line<'static>> {
    if width == 0 {
        return Vec::new();
    }
    let mut lines = Vec::new();
    for source_line in text.split('\n') {
        if source_line.is_empty() {
            lines.push(Line::from(Span::styled("  ", style)));
            continue;
        }
        let options = textwrap::Options::new(usize::from(width))
            .initial_indent("  ")
            .subsequent_indent("  ");
        for wrapped in textwrap::wrap(source_line, options) {
            lines.push(Line::from(Span::styled(wrapped.into_owned(), style)));
        }
    }
    lines
}

fn transcript_block_content_width(width: u16) -> u16 {
    if width >= 5 {
        width - 4
    } else if width > 2 {
        width - 2
    } else {
        width
    }
}

fn style_transcript_block(
    block: Vec<Line<'static>>,
    background: Color,
    border_color: Color,
    width: u16,
) -> Vec<Line<'static>> {
    if width < 5 {
        let horizontal_padding = usize::from(width > 2);
        let width = usize::from(width);
        return block
            .into_iter()
            .map(|mut line| {
                let content_width = line.width();
                if horizontal_padding != 0 {
                    line.spans.insert(0, Span::raw(" "));
                }
                let occupied = content_width.saturating_add(horizontal_padding);
                if occupied < width {
                    line.spans.push(Span::raw(" ".repeat(width - occupied)));
                }
                line.style(Style::default().bg(background))
            })
            .collect();
    }

    let width = usize::from(width);
    let inner_width = width - 2;
    let horizontal_padding = 1_usize;
    let block_style = Style::default().bg(background);
    let border_style = Style::default().fg(border_color).bg(background);
    let horizontal = "─".repeat(inner_width);
    let mut styled = Vec::with_capacity(block.len().saturating_add(2));
    styled.push(
        Line::from(vec![
            Span::styled("╭", border_style),
            Span::styled(horizontal.clone(), border_style),
            Span::styled("╮", border_style),
        ])
        .style(block_style),
    );
    for mut line in block {
        let content_width = line.width();
        line.spans.insert(0, Span::raw(" "));
        line.spans.insert(0, Span::styled("│", border_style));
        let occupied = content_width.saturating_add(horizontal_padding);
        if occupied < inner_width {
            line.spans
                .push(Span::raw(" ".repeat(inner_width - occupied)));
        }
        line.spans.push(Span::styled("│", border_style));
        styled.push(line.style(block_style));
    }
    styled.push(
        Line::from(vec![
            Span::styled("╰", border_style),
            Span::styled(horizontal, border_style),
            Span::styled("╯", border_style),
        ])
        .style(block_style),
    );
    styled
}

fn push_transcript_block(
    lines: &mut Vec<Line<'static>>,
    block: Vec<Line<'static>>,
    background: Color,
    border_color: Color,
    width: u16,
) {
    if block.is_empty() {
        return;
    }
    lines.extend(style_transcript_block(
        block,
        background,
        border_color,
        width,
    ));
    push_transcript_gap(lines);
}

struct LiveToolOutput {
    call_id: String,
    label: String,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl LiveToolOutput {
    fn new(call_id: String, label: String) -> Self {
        Self {
            call_id,
            label,
            stdout: Vec::new(),
            stderr: Vec::new(),
        }
    }

    fn append(&mut self, stream: ToolOutputStream, bytes: &[u8]) {
        let destination = match stream {
            ToolOutputStream::Stdout => &mut self.stdout,
            ToolOutputStream::Stderr => &mut self.stderr,
        };
        destination.extend_from_slice(bytes);
        if destination.len() > DEFAULT_MAX_TOOL_OUTPUT_BYTES {
            let excess = destination.len() - DEFAULT_MAX_TOOL_OUTPUT_BYTES;
            destination.drain(..excess);
        }
    }
}

fn tail_text(bytes: &[u8], lines: usize) -> String {
    let text = String::from_utf8_lossy(bytes);
    let all = text.lines().collect::<Vec<_>>();
    all[all.len().saturating_sub(lines)..].join("\n")
}

fn build_live_tool_lines(live: &LiveToolOutput, width: u16, expanded: bool) -> Vec<Line<'static>> {
    let content_width = transcript_block_content_width(width);
    let mut block = vec![Line::from(Span::styled(
        format!("└ Running · {}  #{}", live.label, live.call_id),
        Style::default().fg(COLOR_TOOL).add_modifier(Modifier::BOLD),
    ))];
    for (label, bytes, color) in [
        ("stdout", live.stdout.as_slice(), COLOR_TEXT),
        ("stderr", live.stderr.as_slice(), COLOR_ERROR),
    ] {
        if bytes.is_empty() {
            continue;
        }
        let content = if expanded {
            String::from_utf8_lossy(bytes).into_owned()
        } else {
            tail_text(bytes, COLLAPSED_LIVE_TAIL_LINES)
        };
        block.push(Line::from(Span::styled(
            format!("  {label}{}", if expanded { "" } else { " tail" }),
            Style::default().fg(COLOR_MUTED).add_modifier(Modifier::DIM),
        )));
        block.extend(indented_wrapped_lines(
            &content,
            Style::default().fg(color),
            content_width,
        ));
    }
    style_transcript_block(block, COLOR_TOOL_BACKGROUND, COLOR_TOOL, width)
}

fn build_grep_result_lines(content: &str, style: Style, width: u16) -> Vec<Line<'static>> {
    if width == 0 {
        return Vec::new();
    }
    let available = usize::from(width).saturating_sub(2).max(1);
    let mut lines = Vec::new();
    for result in content.lines().filter(|line| !line.trim().is_empty()) {
        let wrapped = textwrap::wrap(result, available);
        for (index, row) in wrapped.iter().enumerate() {
            let prefix = if index == 0 { "  • " } else { "    " };
            lines.push(Line::from(vec![
                Span::styled(prefix, style.add_modifier(Modifier::BOLD)),
                Span::styled(row.clone().into_owned(), style),
            ]));
        }
    }
    lines
}

fn build_completed_bash_tail(content: &str, width: u16) -> Vec<Line<'static>> {
    let (stdout, stderr) = content
        .split_once("\n[stderr]\n")
        .map_or((content, ""), |(stdout, stderr)| (stdout, stderr));
    let mut lines = Vec::new();
    for (label, stream, color) in [
        ("stdout tail", stdout, COLOR_TEXT),
        ("stderr tail", stderr, COLOR_ERROR),
    ] {
        if stream.is_empty() {
            continue;
        }
        lines.push(Line::from(Span::styled(
            format!("  {label}"),
            Style::default().fg(COLOR_MUTED).add_modifier(Modifier::DIM),
        )));
        lines.extend(indented_wrapped_lines(
            &tail_text(stream.as_bytes(), COLLAPSED_LIVE_TAIL_LINES),
            Style::default().fg(color),
            width,
        ));
    }
    lines
}

fn push_folded_lines(
    lines: &mut Vec<Line<'static>>,
    mut block: Vec<Line<'static>>,
    expanded: bool,
    visible_lines: usize,
    kind: &str,
    show_tail: bool,
) {
    if !expanded && block.len() > visible_lines {
        let hidden = block.len() - visible_lines;
        lines.push(Line::from(Span::styled(
            format!("  … {hidden} {kind} lines hidden · Ctrl+O expand"),
            Style::default().fg(COLOR_MUTED).add_modifier(Modifier::DIM),
        )));
        if show_tail {
            lines.extend(block.drain(hidden..));
        } else {
            lines.extend(block.drain(..visible_lines));
        }
    } else {
        lines.append(&mut block);
    }
}

fn append_tool_call_lines(
    lines: &mut Vec<Line<'static>>,
    call: &CoreToolCall,
    width: u16,
    expanded: bool,
) {
    if width == 0 {
        return;
    }
    let prefix = "  └─ ";
    let continuation = "     ";
    let suffix = format!("  #{}", call.call_id);
    let available = usize::from(width)
        .saturating_sub(textwrap::core::display_width(prefix))
        .max(1);
    let summary = call.transcript_summary();
    let wrapped = textwrap::wrap(&summary, available);
    let visible = if expanded {
        wrapped.len()
    } else {
        wrapped.len().min(COLLAPSED_COMMAND_LINES)
    };
    for (index, row) in wrapped.iter().take(visible).enumerate() {
        let indent = if index == 0 { prefix } else { continuation };
        let is_last = index + 1 == visible;
        let suffix_fits = expanded
            && is_last
            && textwrap::core::display_width(indent)
                + textwrap::core::display_width(row)
                + textwrap::core::display_width(&suffix)
                <= usize::from(width);
        let mut spans = vec![
            Span::styled(indent, Style::default().fg(COLOR_MUTED)),
            Span::styled(row.clone().into_owned(), Style::default().fg(COLOR_ACCENT)),
        ];
        if suffix_fits {
            spans.push(Span::styled(
                suffix.clone(),
                Style::default().fg(COLOR_MUTED),
            ));
        }
        lines.push(Line::from(spans));
    }
    if !expanded && wrapped.len() > visible {
        lines.push(Line::from(Span::styled(
            format!(
                "     … {} command lines hidden · Ctrl+O expand  #{}",
                wrapped.len() - visible,
                call.call_id
            ),
            Style::default().fg(COLOR_MUTED).add_modifier(Modifier::DIM),
        )));
    } else if expanded && !suffix.is_empty() && !wrapped.is_empty() {
        let last = wrapped.last().expect("non-empty tool command summary");
        let last_indent = if wrapped.len() == 1 {
            prefix
        } else {
            continuation
        };
        if textwrap::core::display_width(last_indent)
            + textwrap::core::display_width(last)
            + textwrap::core::display_width(&suffix)
            > usize::from(width)
        {
            lines.push(Line::from(vec![
                Span::styled(continuation, Style::default().fg(COLOR_MUTED)),
                Span::styled(suffix, Style::default().fg(COLOR_MUTED)),
            ]));
        }
    }
}

fn build_tool_transcript_block(
    calls: &[&CoreToolCall],
    result: Option<&CoreMessage>,
    width: u16,
    details_expanded: bool,
) -> (Vec<Line<'static>>, Color, Color) {
    let content_width = transcript_block_content_width(width);
    let is_file_result = result
        .is_some_and(|message| calls.iter().any(|call| call.name == "read") && !message.is_error);
    let is_grep_result = result
        .is_some_and(|message| calls.iter().any(|call| call.name == "grep") && !message.is_error);
    let is_todo_result = result
        .is_some_and(|message| calls.iter().any(|call| call.name == "todo") && !message.is_error);
    let is_error = result.is_some_and(|message| message.is_error);
    let (border_color, content_color, background) = if is_error {
        (COLOR_ERROR, COLOR_ERROR, COLOR_ERROR_BACKGROUND)
    } else if is_file_result {
        (COLOR_TOOL, Color::Rgb(161, 161, 170), COLOR_FILE_BACKGROUND)
    } else if is_grep_result {
        (
            Color::Rgb(56, 189, 248),
            Color::Rgb(186, 230, 253),
            COLOR_GREP_BACKGROUND,
        )
    } else {
        (COLOR_TOOL, Color::Rgb(161, 161, 170), COLOR_TOOL_BACKGROUND)
    };
    let mut block = Vec::new();
    for call in calls {
        append_tool_call_lines(&mut block, call, content_width, details_expanded);
    }
    if let Some(message) = result {
        let label = if is_error {
            "! Tool error"
        } else if is_file_result {
            "└ File result"
        } else if is_grep_result {
            "└ Search results"
        } else {
            "└ Tool result"
        };
        push_wrapped_line(
            &mut block,
            label,
            Style::default()
                .fg(border_color)
                .add_modifier(Modifier::BOLD),
            content_width,
        );
        if !message.content.is_empty() {
            if is_todo_result {
                block.push(Line::from(Span::styled(
                    "  ✓ task list updated",
                    Style::default().fg(content_color),
                )));
            } else if calls.iter().any(|call| call.name == "bash") && !details_expanded {
                block.extend(build_completed_bash_tail(&message.content, content_width));
            } else if is_grep_result {
                let result_lines = build_grep_result_lines(
                    &message.content,
                    Style::default().fg(content_color),
                    content_width,
                );
                push_folded_lines(
                    &mut block,
                    result_lines,
                    details_expanded,
                    COLLAPSED_FILE_OUTPUT_LINES,
                    "search result",
                    true,
                );
            } else {
                let content_lines = indented_wrapped_lines(
                    &message.content,
                    Style::default().fg(content_color),
                    content_width,
                );
                push_folded_lines(
                    &mut block,
                    content_lines,
                    details_expanded,
                    COLLAPSED_FILE_OUTPUT_LINES,
                    if is_file_result {
                        "file output"
                    } else if is_grep_result {
                        "search result"
                    } else {
                        "tool output"
                    },
                    !is_file_result && !is_todo_result,
                );
            }
        }
    }
    (block, background, border_color)
}

fn build_transcript_lines(
    messages: &[CoreMessage],
    local_entries: &[LocalTranscriptEntry],
    width: u16,
    details_expanded: bool,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut tool_calls: BTreeMap<&str, &CoreToolCall> = BTreeMap::new();
    let content_width = transcript_block_content_width(width);
    for message in messages {
        let source_call = message
            .tool_call_id
            .as_deref()
            .and_then(|call_id| tool_calls.get(call_id));
        if message.role == "tool" {
            let calls: Vec<&CoreToolCall> = match source_call {
                Some(call) => vec![call],
                None => Vec::new(),
            };
            let (block, background, border_color) =
                build_tool_transcript_block(&calls, Some(message), content_width, details_expanded);
            push_transcript_block(&mut lines, block, background, border_color, width);
            continue;
        }
        match message.role.as_str() {
            "user" => push_human_message(
                &mut lines,
                &message.content,
                "> ",
                COLOR_USER,
                COLOR_TEXT,
                COLOR_USER_BACKGROUND,
                width,
            ),
            "assistant" => {
                if !message.content.is_empty() {
                    push_human_message(
                        &mut lines,
                        &message.content,
                        "· ",
                        COLOR_ASSISTANT,
                        COLOR_TEXT,
                        COLOR_ASSISTANT_BACKGROUND,
                        width,
                    );
                }
                if !message.tool_calls.is_empty() {
                    let calls = message
                        .tool_calls
                        .iter()
                        .filter(|call| {
                            !(call.name == "grep"
                                && messages.iter().any(|result| {
                                    result.role == "tool"
                                        && result.tool_call_id.as_deref()
                                            == Some(call.call_id.as_str())
                                }))
                        })
                        .collect::<Vec<_>>();
                    if !calls.is_empty() {
                        let (block, background, border_color) = build_tool_transcript_block(
                            &calls,
                            None,
                            content_width,
                            details_expanded,
                        );
                        push_transcript_block(&mut lines, block, background, border_color, width);
                    }
                }
            }
            _ => push_human_message(
                &mut lines,
                &message.content,
                "· ",
                COLOR_MUTED,
                COLOR_TEXT,
                COLOR_MESSAGE_BACKGROUND,
                width,
            ),
        }
        for call in &message.tool_calls {
            tool_calls.insert(call.call_id.as_str(), call);
        }
    }
    for entry in local_entries {
        push_human_message(
            &mut lines,
            &format!("BTW · {}", entry.question),
            "> ",
            COLOR_USER,
            COLOR_TEXT,
            COLOR_USER_BACKGROUND,
            width,
        );
        push_human_message(
            &mut lines,
            &entry.answer,
            "· ",
            COLOR_ASSISTANT,
            COLOR_TEXT,
            COLOR_ASSISTANT_BACKGROUND,
            width,
        );
    }
    lines
}
const MAX_SELECTION_BYTES: usize = 1024 * 1024;
const MULTI_CLICK_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SelectionPoint {
    row: u16,
    column: u16,
}

impl SelectionPoint {
    fn from_mouse(event: MouseEvent) -> Self {
        Self {
            row: event.row,
            column: event.column,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SelectionRegion {
    left: u16,
    top: u16,
    right: u16,
    bottom: u16,
}

impl SelectionRegion {
    fn from_transcript_rect(rect: Rect) -> Option<Self> {
        let region = Self {
            left: rect.x.saturating_add(1),
            top: rect.y,
            right: rect.x.saturating_add(rect.width.saturating_sub(1)),
            bottom: rect.y.saturating_add(rect.height),
        };
        (region.left < region.right && region.top < region.bottom).then_some(region)
    }

    fn contains(self, point: SelectionPoint) -> bool {
        point.column >= self.left
            && point.column < self.right
            && point.row >= self.top
            && point.row < self.bottom
    }

    fn clamp(self, point: SelectionPoint) -> SelectionPoint {
        SelectionPoint {
            row: point.row.clamp(self.top, self.bottom - 1),
            column: point.column.clamp(self.left, self.right - 1),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TextSelection {
    anchor: SelectionPoint,
    focus: SelectionPoint,
    dragging: bool,
}

impl TextSelection {
    fn ordered(self) -> (SelectionPoint, SelectionPoint) {
        if self.anchor <= self.focus {
            (self.anchor, self.focus)
        } else {
            (self.focus, self.anchor)
        }
    }

    fn contains(self, point: SelectionPoint) -> bool {
        let (start, end) = self.ordered();
        if point.row < start.row || point.row > end.row {
            return false;
        }
        if start.row == end.row {
            return point.column >= start.column && point.column <= end.column;
        }
        if point.row == start.row {
            return point.column >= start.column;
        }
        if point.row == end.row {
            return point.column <= end.column;
        }
        true
    }
}
#[derive(Clone, Copy, Debug)]
struct ClickTracker {
    point: SelectionPoint,
    at: Instant,
    count: u8,
}

#[derive(Debug, Eq, PartialEq)]
enum MouseAction {
    None,
    Redraw,
    Scroll(i32),
    Copy(String),
    SelectionTooLarge,
}

struct SelectionOverlay {
    selection: TextSelection,
    region: SelectionRegion,
}

impl Widget for SelectionOverlay {
    fn render(self, _area: Rect, buffer: &mut Buffer) {
        let style = Style::default().fg(Color::Black).bg(COLOR_ACCENT);
        for row in self.region.top..self.region.bottom {
            for column in self.region.left..self.region.right {
                if self.selection.contains(SelectionPoint { row, column })
                    && let Some(cell) = buffer.cell_mut((column, row))
                {
                    cell.set_style(style);
                }
            }
        }
    }
}

fn text_between_columns(line: &str, start: usize, end: usize) -> String {
    let mut result = String::new();
    let mut column = 0_usize;
    let mut include_combining = false;
    for character in line.chars() {
        let width = UnicodeWidthChar::width(character).unwrap_or(0);
        if width == 0 {
            if include_combining {
                result.push(character);
            }
            continue;
        }
        let character_end = column.saturating_add(width);
        include_combining = character_end > start && column < end;
        if include_combining {
            result.push(character);
        }
        column = character_end;
    }
    result
}
fn is_pure_box_border(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty()
        && trimmed.chars().all(|character| {
            character.is_whitespace()
                || matches!(
                    character,
                    '│' | '┃'
                        | '║'
                        | '╭'
                        | '╮'
                        | '╰'
                        | '╯'
                        | '─'
                        | '┌'
                        | '┐'
                        | '└'
                        | '┘'
                )
        })
}

fn selection_content_span(line: &str) -> Option<(usize, usize)> {
    if line.trim().is_empty() || is_pure_box_border(line) {
        return None;
    }
    let mut start = 0_usize;
    if let Some(first) = line.chars().next()
        && matches!(first, '│' | '┃' | '║')
    {
        start = UnicodeWidthChar::width(first).unwrap_or(0);
        let after_border = &line[first.len_utf8()..];
        if after_border.starts_with(' ') {
            start = start.saturating_add(1);
        }
    }

    let trimmed = line.trim_end();
    let without_trailing_border = trimmed
        .char_indices()
        .next_back()
        .filter(|(_, character)| matches!(character, '│' | '┃' | '║'))
        .map_or(trimmed, |(index, _)| &trimmed[..index]);
    let end = UnicodeWidthStr::width(without_trailing_border.trim_end());
    (start < end).then_some((start, end))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InputClass {
    Whitespace,
    Identifier,
    Punctuation,
}

fn input_class(grapheme: &str) -> InputClass {
    let first = grapheme.chars().next();
    if first.is_some_and(char::is_whitespace) {
        InputClass::Whitespace
    } else if first.is_some_and(|character| character.is_alphanumeric() || character == '_') {
        InputClass::Identifier
    } else {
        InputClass::Punctuation
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct InputBuffer {
    text: String,
    cursor: usize,
}

impl InputBuffer {
    fn as_str(&self) -> &str {
        &self.text
    }

    fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    fn set_text(&mut self, text: String) {
        self.cursor = text.len();
        self.text = text;
    }

    fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    fn insert(&mut self, character: char) {
        self.text.insert(self.cursor, character);
        self.cursor += character.len_utf8();
    }

    fn backspace(&mut self) -> bool {
        let previous = self.previous_grapheme_boundary();
        if previous == self.cursor {
            return false;
        }
        self.text.drain(previous..self.cursor);
        self.cursor = previous;
        true
    }

    fn delete(&mut self) -> bool {
        let next = self.next_grapheme_boundary();
        if next == self.cursor {
            return false;
        }
        self.text.drain(self.cursor..next);
        true
    }

    fn move_left(&mut self) {
        self.cursor = self.previous_grapheme_boundary();
    }

    fn move_right(&mut self) {
        self.cursor = self.next_grapheme_boundary();
    }

    fn move_start(&mut self) {
        self.cursor = 0;
    }

    fn move_end(&mut self) {
        self.cursor = self.text.len();
    }

    fn move_line_start(&mut self) {
        self.cursor = self.text[..self.cursor]
            .rfind('\n')
            .map_or(0, |index| index + 1);
    }

    fn move_line_end(&mut self) {
        self.cursor = self.text[self.cursor..]
            .find('\n')
            .map_or(self.text.len(), |index| self.cursor + index);
    }

    fn move_word_left(&mut self) {
        self.cursor = self.previous_word_boundary();
    }

    fn move_word_right(&mut self) {
        self.cursor = self.next_word_boundary();
    }

    fn delete_word_left(&mut self) -> bool {
        let previous = self.previous_word_boundary();
        if previous == self.cursor {
            return false;
        }
        self.text.drain(previous..self.cursor);
        self.cursor = previous;
        true
    }

    fn delete_word_right(&mut self) -> bool {
        let next = self.next_word_boundary();
        if next == self.cursor {
            return false;
        }
        self.text.drain(self.cursor..next);
        true
    }

    fn previous_grapheme_boundary(&self) -> usize {
        self.text[..self.cursor]
            .grapheme_indices(true)
            .next_back()
            .map_or(self.cursor, |(index, _)| index)
    }

    fn next_grapheme_boundary(&self) -> usize {
        self.text[self.cursor..]
            .graphemes(true)
            .next()
            .map_or(self.cursor, |grapheme| self.cursor + grapheme.len())
    }

    fn previous_word_boundary(&self) -> usize {
        let mut position = self.cursor;
        while let Some((start, grapheme)) = self.grapheme_before(position) {
            if input_class(grapheme) != InputClass::Whitespace {
                break;
            }
            position = start;
        }
        let Some((start, grapheme)) = self.grapheme_before(position) else {
            return position;
        };
        let class = input_class(grapheme);
        position = start;
        while let Some((start, grapheme)) = self.grapheme_before(position) {
            if input_class(grapheme) != class {
                break;
            }
            position = start;
        }
        position
    }

    fn next_word_boundary(&self) -> usize {
        let mut position = self.cursor;
        while let Some(grapheme) = self.grapheme_at(position) {
            if input_class(grapheme) != InputClass::Whitespace {
                break;
            }
            position += grapheme.len();
        }
        let Some(grapheme) = self.grapheme_at(position) else {
            return position;
        };
        let class = input_class(grapheme);
        while let Some(grapheme) = self.grapheme_at(position) {
            if input_class(grapheme) != class {
                break;
            }
            position += grapheme.len();
        }
        position
    }

    fn grapheme_before(&self, position: usize) -> Option<(usize, &str)> {
        self.text[..position].grapheme_indices(true).next_back()
    }

    fn grapheme_at(&self, position: usize) -> Option<&str> {
        self.text[position..].graphemes(true).next()
    }
}

#[derive(Debug, Eq, PartialEq)]
struct InputLayout {
    lines: Vec<String>,
    cursor_row: usize,
    cursor_column: usize,
}

impl InputLayout {
    fn new(input: &InputBuffer, width: u16) -> Self {
        let width = usize::from(width.max(1));
        let mut lines = Vec::new();
        let mut line = String::new();
        let mut row = 0_usize;
        let mut column = 0_usize;
        let mut cursor = None;

        for (index, grapheme) in input.text.grapheme_indices(true) {
            if grapheme == "\n" {
                if column >= width {
                    lines.push(std::mem::take(&mut line));
                    row += 1;
                    column = 0;
                }
                if input.cursor == index {
                    cursor = Some((row, column));
                }
                lines.push(std::mem::take(&mut line));
                row += 1;
                column = 0;
                continue;
            }

            let grapheme_width = UnicodeWidthStr::width(grapheme);
            if grapheme_width > 0 && column > 0 && column.saturating_add(grapheme_width) > width {
                lines.push(std::mem::take(&mut line));
                row += 1;
                column = 0;
            }
            if input.cursor == index {
                cursor = Some((row, column));
            }
            line.push_str(grapheme);
            column = column.saturating_add(grapheme_width);
        }

        if input.cursor == input.text.len() {
            if column >= width && !line.is_empty() {
                lines.push(std::mem::take(&mut line));
                row += 1;
                column = 0;
            }
            cursor = Some((row, column));
        }
        lines.push(line);
        let (cursor_row, cursor_column) = cursor.unwrap_or((0, 0));
        Self {
            lines,
            cursor_row,
            cursor_column,
        }
    }

    fn render_height(&self, available_height: u16) -> u16 {
        let desired = u16::try_from(self.lines.len().saturating_add(2)).unwrap_or(u16::MAX);
        desired.min(available_height.max(3)).max(3)
    }

    fn visible_start(&self, rows: usize) -> usize {
        self.cursor_row
            .saturating_add(1)
            .saturating_sub(rows.max(1))
    }

    fn visible_text(&self, start: usize, rows: usize) -> Text<'static> {
        Text::from(
            self.lines
                .iter()
                .skip(start)
                .take(rows)
                .cloned()
                .map(Line::from)
                .collect::<Vec<_>>(),
        )
    }
}

struct App {
    input: InputBuffer,
    slash_selection: usize,
    slash_menu_dismissed: bool,
    status: String,
    snapshot: CoreSnapshot,
    approval: Option<CoreToolCall>,
    plan_review: bool,
    plan_feedback: bool,
    busy: bool,
    scroll_offset: usize,
    max_scroll: usize,
    page_rows: usize,
    follow_tail: bool,
    selection_region: Option<SelectionRegion>,
    selection_lines: Vec<String>,
    text_selection: Option<TextSelection>,
    last_click: Option<ClickTracker>,
    pending_click_selection: bool,
    transcript_revision: u64,
    transcript_details_expanded: bool,
    transcript_layout: Option<TranscriptLayoutCache>,
    local_entries: Vec<LocalTranscriptEntry>,
    live_tool: Option<LiveToolOutput>,
}

impl App {
    fn new(snapshot: CoreSnapshot) -> Self {
        Self {
            input: InputBuffer::default(),
            slash_selection: 0,
            slash_menu_dismissed: false,
            status: "Ready".to_owned(),
            snapshot,
            approval: None,
            plan_review: false,
            plan_feedback: false,
            busy: false,
            scroll_offset: 0,
            max_scroll: 0,
            page_rows: 1,
            follow_tail: true,
            selection_region: None,
            selection_lines: Vec::new(),
            text_selection: None,
            last_click: None,
            pending_click_selection: false,
            transcript_revision: 0,
            transcript_details_expanded: false,
            transcript_layout: None,
            local_entries: Vec::new(),
            live_tool: None,
        }
    }
    fn slash_candidate_count(&self) -> usize {
        slash_command_candidates(self.input.as_str()).count()
    }

    fn slash_menu_visible(&self) -> bool {
        !self.plan_review
            && !self.plan_feedback
            && !self.slash_menu_dismissed
            && self.slash_candidate_count() > 0
    }

    fn input_changed(&mut self) {
        self.slash_selection = 0;
        self.slash_menu_dismissed = false;
    }
    fn set_input(&mut self, text: String) {
        self.input.set_text(text);
        self.input_changed();
    }

    fn push_input_character(&mut self, character: char) {
        self.input.insert(character);
        self.input_changed();
    }

    fn pop_input_character(&mut self) {
        if self.input.backspace() {
            self.input_changed();
        }
    }

    fn delete_input_character(&mut self) {
        if self.input.delete() {
            self.input_changed();
        }
    }

    fn delete_input_word_left(&mut self) {
        if self.input.delete_word_left() {
            self.input_changed();
        }
    }

    fn delete_input_word_right(&mut self) {
        if self.input.delete_word_right() {
            self.input_changed();
        }
    }

    fn clear_input(&mut self) {
        self.input.clear();
        self.input_changed();
    }

    fn move_slash_selection(&mut self, previous: bool) -> bool {
        if !self.slash_menu_visible() {
            return false;
        }
        let count = self.slash_candidate_count();
        self.slash_selection = if previous {
            self.slash_selection.checked_sub(1).unwrap_or(count - 1)
        } else {
            (self.slash_selection + 1) % count
        };
        true
    }

    fn complete_selected_slash_command(&mut self) -> bool {
        let Some(spec) = slash_command_candidates(self.input.as_str())
            .nth(self.slash_selection)
            .copied()
        else {
            return false;
        };
        let mut completed = format!("/{}", spec.name);
        if spec.takes_arguments {
            completed.push(' ');
        }
        self.input.set_text(completed);
        self.slash_selection = 0;
        self.slash_menu_dismissed = true;
        true
    }

    fn dismiss_slash_menu(&mut self) -> bool {
        if !self.slash_menu_visible() {
            return false;
        }
        self.slash_selection = 0;
        self.slash_menu_dismissed = true;
        true
    }
    fn clear_text_selection(&mut self) {
        self.text_selection = None;
        self.pending_click_selection = false;
    }

    fn update_selection_content(&mut self, region: Option<SelectionRegion>, text: &Text<'_>) {
        if self.selection_region != region {
            self.clear_text_selection();
        }
        self.selection_region = region;
        self.selection_lines.clear();
        self.selection_lines
            .extend(text.lines.iter().map(Line::to_string));
    }
    fn register_click(&mut self, point: SelectionPoint, now: Instant) -> u8 {
        let count = self
            .last_click
            .filter(|last| {
                last.point == point
                    && now
                        .checked_duration_since(last.at)
                        .is_some_and(|elapsed| elapsed <= MULTI_CLICK_INTERVAL)
            })
            .map_or(1, |last| if last.count >= 3 { 1 } else { last.count + 1 });
        self.last_click = Some(ClickTracker {
            point,
            at: now,
            count,
        });
        count
    }

    fn word_selection_at(&self, point: SelectionPoint) -> Option<TextSelection> {
        let region = self.selection_region?;
        let line_index = usize::from(point.row.checked_sub(region.top)?);
        let line = self.selection_lines.get(line_index)?;
        let clicked_column = point.column.checked_sub(region.left)?;
        let (content_start, content_end) = selection_content_span(line)?;
        if usize::from(clicked_column) < content_start || usize::from(clicked_column) >= content_end
        {
            return None;
        }
        let mut column = 0_u16;
        let mut spans = Vec::new();
        for grapheme in line.graphemes(true) {
            let width = u16::try_from(UnicodeWidthStr::width(grapheme)).unwrap_or(u16::MAX);
            if width == 0 {
                continue;
            }
            let end = column.saturating_add(width);
            spans.push((column, end, input_class(grapheme)));
            column = end;
        }
        let clicked = spans
            .iter()
            .position(|(start, end, _)| clicked_column >= *start && clicked_column < *end)?;
        let class = spans[clicked].2;
        let mut first = clicked;
        while first > 0 && spans[first - 1].2 == class {
            first -= 1;
        }
        let mut last = clicked;
        while last + 1 < spans.len() && spans[last + 1].2 == class {
            last += 1;
        }
        let start = region.left.saturating_add(
            u16::try_from(usize::from(spans[first].0).max(content_start)).unwrap_or(u16::MAX),
        );
        let end = region
            .left
            .saturating_add(
                u16::try_from(usize::from(spans[last].1).min(content_end)).unwrap_or(u16::MAX),
            )
            .min(region.right)
            .saturating_sub(1);
        Some(TextSelection {
            anchor: SelectionPoint {
                row: point.row,
                column: start,
            },
            focus: SelectionPoint {
                row: point.row,
                column: end,
            },
            dragging: false,
        })
    }

    fn line_selection_at(&self, point: SelectionPoint) -> Option<TextSelection> {
        let region = self.selection_region?;
        let line_index = usize::from(point.row.checked_sub(region.top)?);
        let (start, end) = selection_content_span(self.selection_lines.get(line_index)?)?;
        Some(TextSelection {
            anchor: SelectionPoint {
                row: point.row,
                column: region
                    .left
                    .saturating_add(u16::try_from(start).unwrap_or(u16::MAX)),
            },
            focus: SelectionPoint {
                row: point.row,
                column: region
                    .left
                    .saturating_add(u16::try_from(end).unwrap_or(u16::MAX))
                    .min(region.right)
                    .saturating_sub(1),
            },
            dragging: false,
        })
    }
    fn paragraph_selection_at(&self, point: SelectionPoint) -> Option<TextSelection> {
        let region = self.selection_region?;
        let clicked = usize::from(point.row.checked_sub(region.top)?);
        selection_content_span(self.selection_lines.get(clicked)?)?;
        let mut first = clicked;
        while first > 0
            && self
                .selection_lines
                .get(first - 1)
                .and_then(|line| selection_content_span(line))
                .is_some()
        {
            first -= 1;
        }
        let mut last = clicked;
        while self
            .selection_lines
            .get(last + 1)
            .and_then(|line| selection_content_span(line))
            .is_some()
        {
            last += 1;
        }
        let (first_start, _) = selection_content_span(self.selection_lines.get(first)?)?;
        let (_, last_end) = selection_content_span(self.selection_lines.get(last)?)?;
        Some(TextSelection {
            anchor: SelectionPoint {
                row: region
                    .top
                    .saturating_add(u16::try_from(first).unwrap_or(u16::MAX)),
                column: region
                    .left
                    .saturating_add(u16::try_from(first_start).unwrap_or(u16::MAX)),
            },
            focus: SelectionPoint {
                row: region
                    .top
                    .saturating_add(u16::try_from(last).unwrap_or(u16::MAX)),
                column: region
                    .left
                    .saturating_add(u16::try_from(last_end).unwrap_or(u16::MAX))
                    .min(region.right)
                    .saturating_sub(1),
            },
            dragging: false,
        })
    }

    fn completed_selection_action(&mut self) -> MouseAction {
        let (Some(region), Some(selection)) = (self.selection_region, self.text_selection) else {
            return MouseAction::Redraw;
        };
        let (start, end) = selection.ordered();
        let mut text = String::new();
        let mut selected_rows = 0_usize;
        for row in start.row..=end.row {
            let line_index = usize::from(row.saturating_sub(region.top));
            let line = self
                .selection_lines
                .get(line_index)
                .map_or("", String::as_str);
            if is_pure_box_border(line) {
                continue;
            }
            let content_span = selection_content_span(line);
            let start_column = if row == start.row {
                usize::from(start.column.saturating_sub(region.left))
            } else {
                content_span.map_or(0, |(content_start, _)| content_start)
            };
            let end_column = if row == end.row {
                usize::from(end.column.saturating_sub(region.left).saturating_add(1))
            } else {
                content_span.map_or(0, |(_, content_end)| content_end)
            };
            let selected = content_span.map_or("".to_owned(), |(content_start, content_end)| {
                text_between_columns(
                    line,
                    start_column.max(content_start),
                    end_column.min(content_end),
                )
            });
            let selected = selected.trim_end();
            let separator_bytes = usize::from(selected_rows > 0);
            if text
                .len()
                .saturating_add(separator_bytes)
                .saturating_add(selected.len())
                > MAX_SELECTION_BYTES
            {
                return MouseAction::SelectionTooLarge;
            }
            if selected_rows > 0 {
                text.push('\n');
            }
            text.push_str(selected);
            selected_rows += 1;
        }
        if text.trim().is_empty() {
            self.clear_text_selection();
            MouseAction::Redraw
        } else {
            MouseAction::Copy(text)
        }
    }

    fn handle_mouse(&mut self, event: MouseEvent) -> MouseAction {
        self.handle_mouse_at(event, Instant::now())
    }

    fn handle_mouse_at(&mut self, event: MouseEvent, now: Instant) -> MouseAction {
        let point = SelectionPoint::from_mouse(event);
        let over_transcript = self
            .selection_region
            .is_some_and(|region| point.row >= region.top && point.row < region.bottom);
        match event.kind {
            MouseEventKind::ScrollUp if over_transcript => {
                self.last_click = None;
                self.clear_text_selection();
                MouseAction::Scroll(-1)
            }
            MouseEventKind::ScrollDown if over_transcript => {
                self.last_click = None;
                self.clear_text_selection();
                MouseAction::Scroll(1)
            }
            MouseEventKind::Down(MouseButton::Left) => {
                let Some(region) = self.selection_region else {
                    return MouseAction::None;
                };
                if !region.contains(point) {
                    self.last_click = None;
                    let changed = self.text_selection.take().is_some();
                    self.pending_click_selection = false;
                    return if changed {
                        MouseAction::Redraw
                    } else {
                        MouseAction::None
                    };
                }
                let click_count = self.register_click(point, now);
                let multi_click_selection = if click_count >= 3 {
                    self.paragraph_selection_at(point)
                } else if click_count == 2 && event.modifiers.contains(KeyModifiers::CONTROL) {
                    self.line_selection_at(point)
                } else if click_count == 2 {
                    self.word_selection_at(point)
                } else {
                    None
                };
                if let Some(selection) = multi_click_selection {
                    self.text_selection = Some(selection);
                    self.pending_click_selection = true;
                    return MouseAction::Redraw;
                }
                if event.modifiers.contains(KeyModifiers::CONTROL)
                    && let Some(selection) = &mut self.text_selection
                {
                    selection.focus = point;
                    selection.dragging = true;
                    return MouseAction::Redraw;
                }
                self.pending_click_selection = false;
                self.text_selection = Some(TextSelection {
                    anchor: point,
                    focus: point,
                    dragging: true,
                });
                MouseAction::Redraw
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                let (Some(region), Some(selection)) =
                    (self.selection_region, &mut self.text_selection)
                else {
                    return MouseAction::None;
                };
                if !selection.dragging {
                    return MouseAction::None;
                }
                self.last_click = None;
                self.pending_click_selection = false;
                selection.focus = region.clamp(point);
                MouseAction::Redraw
            }
            MouseEventKind::Up(MouseButton::Left) => {
                if self.pending_click_selection {
                    self.pending_click_selection = false;
                    return self.completed_selection_action();
                }
                let (Some(region), Some(selection)) =
                    (self.selection_region, &mut self.text_selection)
                else {
                    return MouseAction::None;
                };
                if !selection.dragging {
                    return MouseAction::None;
                }
                selection.focus = region.clamp(point);
                let is_click = selection.anchor == selection.focus;
                selection.dragging = false;
                if is_click {
                    self.clear_text_selection();
                    return MouseAction::Redraw;
                }
                self.completed_selection_action()
            }
            _ => MouseAction::None,
        }
    }

    fn set_snapshot(&mut self, snapshot: CoreSnapshot) {
        let transcript_changed = self.snapshot.messages.len() != snapshot.messages.len()
            || self.snapshot.messages.last() != snapshot.messages.last();
        self.snapshot = snapshot;
        if transcript_changed {
            self.transcript_revision = self.transcript_revision.wrapping_add(1);
            self.transcript_layout = None;
            self.clear_text_selection();
        }
    }

    fn toggle_transcript_details(&mut self) {
        self.transcript_details_expanded = !self.transcript_details_expanded;
        self.transcript_revision = self.transcript_revision.wrapping_add(1);
        self.transcript_layout = None;
        self.clear_text_selection();
    }

    fn push_local_entry(&mut self, entry: LocalTranscriptEntry) {
        self.local_entries.push(entry);
        self.transcript_revision = self.transcript_revision.wrapping_add(1);
        self.transcript_layout = None;
        self.clear_text_selection();
    }

    fn set_local_entries(&mut self, entries: Vec<LocalTranscriptEntry>) {
        self.local_entries = entries;
        self.transcript_revision = self.transcript_revision.wrapping_add(1);
        self.transcript_layout = None;
        self.clear_text_selection();
    }

    fn start_live_tool(&mut self, call_id: String, label: String) {
        self.live_tool = Some(LiveToolOutput::new(call_id, label));
        self.clear_text_selection();
    }

    fn append_live_progress(
        &mut self,
        call_id: &str,
        progress: ToolProgress,
    ) -> Result<(), AppError> {
        let bytes = STANDARD.decode(progress.data)?;
        if self
            .live_tool
            .as_ref()
            .is_none_or(|live| live.call_id != call_id)
        {
            self.live_tool = Some(LiveToolOutput::new(call_id.to_owned(), "tool".to_owned()));
        }
        if let Some(live) = &mut self.live_tool {
            live.append(progress.stream, &bytes);
        }
        self.clear_text_selection();
        Ok(())
    }

    fn finish_live_tool(&mut self, call_id: &str) {
        if self
            .live_tool
            .as_ref()
            .is_some_and(|live| live.call_id == call_id)
        {
            self.live_tool = None;
            self.clear_text_selection();
        }
    }

    fn clear_live_tool(&mut self) {
        self.live_tool = None;
        self.clear_text_selection();
    }

    fn visible_transcript(&mut self, width: u16, height: u16) -> Text<'static> {
        let rebuild = self
            .transcript_layout
            .as_ref()
            .is_none_or(|cache| cache.revision != self.transcript_revision || cache.width != width);
        if rebuild {
            self.transcript_layout = Some(TranscriptLayoutCache {
                revision: self.transcript_revision,
                width,
                lines: build_transcript_lines(
                    &self.snapshot.messages,
                    &self.local_entries,
                    width,
                    self.transcript_details_expanded,
                ),
            });
        }
        let live_lines = self
            .live_tool
            .as_ref()
            .map(|live| build_live_tool_lines(live, width, self.transcript_details_expanded))
            .unwrap_or_default();
        let lines = &self
            .transcript_layout
            .as_ref()
            .expect("transcript layout was initialized")
            .lines;
        let total_lines = lines.len().saturating_add(live_lines.len());
        self.max_scroll = total_lines.saturating_sub(usize::from(height));
        self.page_rows = usize::from(height.max(1));
        if self.follow_tail {
            self.scroll_offset = self.max_scroll;
        } else {
            self.scroll_offset = self.scroll_offset.min(self.max_scroll);
        }
        let start = self.scroll_offset.min(total_lines);
        let end = start.saturating_add(usize::from(height)).min(total_lines);
        let mut visible = Vec::with_capacity(end.saturating_sub(start));
        if start < lines.len() {
            visible.extend_from_slice(&lines[start..end.min(lines.len())]);
        }
        if end > lines.len() {
            let live_start = start.saturating_sub(lines.len());
            let live_end = end - lines.len();
            visible.extend_from_slice(&live_lines[live_start..live_end]);
        }
        Text::from(visible)
    }

    fn visible_plan(&mut self, width: u16, height: u16) -> Text<'static> {
        let mut lines = Vec::new();
        for source in self.snapshot.plan.content.lines() {
            let style = if source.starts_with('#') {
                Style::default()
                    .fg(COLOR_ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(COLOR_TEXT)
            };
            push_wrapped_line(&mut lines, source, style, width.max(1));
        }
        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "No plan content.",
                Style::default().fg(COLOR_MUTED),
            )));
        }
        self.max_scroll = lines.len().saturating_sub(usize::from(height));
        self.page_rows = usize::from(height.max(1));
        self.follow_tail = false;
        self.scroll_offset = self.scroll_offset.min(self.max_scroll);
        let start = self.scroll_offset.min(lines.len());
        let end = start.saturating_add(usize::from(height)).min(lines.len());
        Text::from(lines[start..end].to_vec())
    }

    fn scroll_up(&mut self, rows: usize) {
        if self.max_scroll == 0 {
            return;
        }
        let current = if self.follow_tail {
            self.max_scroll
        } else {
            self.scroll_offset
        };
        self.scroll_offset = current.saturating_sub(rows.max(1));
        self.follow_tail = false;
        self.clear_text_selection();
    }

    fn scroll_down(&mut self, rows: usize) {
        let current = if self.follow_tail {
            self.max_scroll
        } else {
            self.scroll_offset
        };
        self.scroll_offset = current.saturating_add(rows.max(1)).min(self.max_scroll);
        self.follow_tail = self.scroll_offset == self.max_scroll;
        self.clear_text_selection();
    }

    fn page_up(&mut self) {
        self.scroll_up(self.page_rows.saturating_sub(1).max(1));
    }

    fn page_down(&mut self) {
        self.scroll_down(self.page_rows.saturating_sub(1).max(1));
    }

    fn scroll_home(&mut self) {
        self.scroll_offset = 0;
        self.follow_tail = self.max_scroll == 0;
        self.clear_text_selection();
    }

    fn scroll_end(&mut self) {
        self.scroll_offset = self.max_scroll;
        self.follow_tail = true;
        self.clear_text_selection();
    }

    fn scroll_delta(&mut self, delta: i32) {
        if delta < 0 {
            self.scroll_up(delta.unsigned_abs() as usize);
        } else if delta > 0 {
            self.scroll_down(delta.unsigned_abs() as usize);
        }
    }
}

enum PlanReviewAction {
    Approve,
    Refine(String),
    Edit(String),
    Cancel,
}

enum RuntimeAction {
    Submit(String),
    Compact(Option<String>),
    EnterPlan(String),
    ReplaceTodos(Vec<CoreTodoPhase>),
    PlanReview(PlanReviewAction),
    Btw {
        question: String,
        snapshot: CoreSnapshot,
    },
    Approval {
        call_id: String,
        approved: bool,
    },
    Drive(CoreResponse),
    Steer(String),
}

enum RuntimeUpdate {
    Progress {
        snapshot: CoreSnapshot,
        status: String,
    },
    ToolStarted {
        call_id: String,
        label: String,
    },
    ToolProgress {
        call_id: String,
        progress: ToolProgress,
    },
    ToolFinished {
        call_id: String,
    },
    Settled(RuntimeFinal),
    Failed(String),
}

struct RuntimeFinal {
    snapshot: CoreSnapshot,
    status: String,
    approval: Option<CoreToolCall>,
    plan_review: bool,
    local_entry: Option<LocalTranscriptEntry>,
}

enum KeyAction {
    None,
    Quit,
    Cancel,
    Submit(String),
    Compact(Option<String>),
    EnterPlan(String),
    EditTodos,
    PlanReview(PlanReviewAction),
    EditPlan,
    Steer(String),
    Btw(String),
    Approval { call_id: String, approved: bool },
}

struct ModelToolCatalogs {
    normal: Vec<ToolSpec>,
    plan: Vec<ToolSpec>,
    declared: Vec<ToolSpec>,
}

struct RuntimeResources {
    core: CoreClient,
    plugins: PluginManager,
    provider: ModelClient,
    model_tools: ModelToolCatalogs,
    desired_safe_tools: Vec<String>,
    desired_permission_mode: PermissionMode,
    auto_compact_enabled: bool,
    auto_compact_threshold: u64,
    sidechain: BtwSidechainStore,
}

type SharedRuntime = Arc<Mutex<RuntimeResources>>;

struct ActiveRuntime {
    handle: JoinHandle<()>,
    cancel: CancellationToken,
    steer: mpsc::UnboundedSender<String>,
}

fn restore_terminal() {
    let _ = disable_raw_mode();
    let mut stdout = std::io::stdout();
    let _ = execute!(stdout, Show, DisableMouseCapture, LeaveAlternateScreen);
}

fn install_terminal_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        restore_terminal();
        previous(panic_info);
    }));
}

fn automatically_safe_tool_names(tools: &[ToolSpec]) -> Vec<String> {
    tools
        .iter()
        .filter(|tool| matches!(tool.name.as_str(), "read" | "grep"))
        .map(|tool| tool.name.clone())
        .collect()
}

fn todo_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "todo".to_owned(),
        description: "Maintain the session task list. Use it for work with three or more distinct steps and update status as work changes.".to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["init", "append", "start", "done", "drop", "block", "unblock", "rm", "view"]},
                "phases": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "minLength": 1, "maxLength": 200},
                            "tasks": {
                                "type": "array",
                                "maxItems": 64,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string", "minLength": 1, "maxLength": 200},
                                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "abandoned", "blocked"]},
                                        "blocker": {"type": "string", "minLength": 1, "maxLength": 200}
                                    },
                                    "required": ["content", "status"],
                                    "additionalProperties": false
                                }
                            }
                        },
                        "required": ["name", "tasks"],
                        "additionalProperties": false
                    }
                },
                "task": {"type": "string", "minLength": 1, "maxLength": 200},
                "phase": {"type": "string", "minLength": 1, "maxLength": 200},
                "items": {"type": "array", "maxItems": 64, "items": {"type": "string", "minLength": 1, "maxLength": 200}},
                "reason": {"type": "string", "minLength": 1, "maxLength": 200}
            },
            "required": ["op"],
            "additionalProperties": false
        }),
    }
}

fn plan_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "plan".to_owned(),
        description: "Update the Markdown implementation plan during plan mode. Use op=update while drafting and op=propose when it is ready for human review.".to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["update", "propose"]},
                "content": {"type": "string", "minLength": 1, "maxLength": 65536}
            },
            "required": ["op", "content"],
            "additionalProperties": false
        }),
    }
}

fn model_tool_catalogs(plugin_tools: &[ToolSpec]) -> ModelToolCatalogs {
    let todo = todo_tool_spec();
    let plan = plan_tool_spec();

    let mut normal = plugin_tools.to_vec();
    normal.push(todo.clone());

    let mut plan_mode = plugin_tools
        .iter()
        .filter(|tool| matches!(tool.name.as_str(), "read" | "grep" | "bash"))
        .cloned()
        .collect::<Vec<_>>();
    plan_mode.push(todo.clone());
    plan_mode.push(plan.clone());

    let mut declared = plugin_tools.to_vec();
    declared.push(todo);
    declared.push(plan);

    ModelToolCatalogs {
        normal,
        plan: plan_mode,
        declared,
    }
}

#[tokio::main]
async fn main() -> Result<(), AppError> {
    install_terminal_panic_hook();
    let args = Args::parse();
    let auto_compact_threshold = args
        .context_window
        .saturating_mul(u64::from(args.auto_compact_threshold_percent))
        / 100;
    let session = args.session.clone();
    let restored_session = match session.as_ref() {
        Some(path) => path.try_exists()?,
        None => false,
    };
    if let Some(path) = &session {
        let parent = path.parent().ok_or(AppError::SessionPathWithoutParent)?;
        tokio::fs::create_dir_all(parent).await?;
    }
    let sidechain_path = btw_sidechain_path(args.btw_sidechain.as_deref(), session.as_deref())?;
    let sidechain = BtwSidechainStore::load(sidechain_path).await?;
    let sidechain_entries = sidechain.entries.clone();
    let core_path = args.core.clone().unwrap_or_else(default_core_path);
    let git_plugin_path = args
        .git_plugin
        .clone()
        .unwrap_or_else(default_git_plugin_path);
    let mut core = CoreClient::spawn(&core_path, session.as_deref()).await?;
    let plugins = PluginManager::spawn(&args.plugin, &git_plugin_path).await?;
    let model_tools = model_tool_catalogs(plugins.model_tools());
    let mut safe_tools = automatically_safe_tool_names(plugins.model_tools());
    if plugins.has_tool("git_read") {
        safe_tools.push("git_read".to_owned());
    }
    let provider = ModelClient::from_args(&args).await?;
    let initial = core.snapshot().await?;
    let desired_permission_mode =
        startup_permission_mode(restored_session, args.permission_mode, &initial.snapshot);
    let mut app = App::new(initial.snapshot);
    app.set_local_entries(sidechain_entries);
    let initial_action = match app.snapshot.phase.as_str() {
        "idle" => {
            let configured = core
                .event(&CoreEvent::configure_tools(
                    safe_tools.clone(),
                    desired_permission_mode,
                ))
                .await?;
            app.set_snapshot(configured.snapshot);
            args.prompt.clone().map(RuntimeAction::Submit)
        }
        "waiting_model" => {
            if args.prompt.is_some() {
                return Err(AppError::PromptWhileResuming {
                    phase: app.snapshot.phase.clone(),
                });
            }
            Some(RuntimeAction::Drive(CoreResponse {
                version: CORE_PROTOCOL_VERSION,
                request_id: "resume_model".to_owned(),
                ok: true,
                snapshot: app.snapshot.clone(),
                effects: vec![CoreEffect {
                    kind: "request_model".to_owned(),
                    call: None,
                }],
                error: None,
            }))
        }
        "waiting_approval" => {
            if args.prompt.is_some() {
                return Err(AppError::PromptWhileResuming {
                    phase: app.snapshot.phase.clone(),
                });
            }
            let call = active_pending_call(&app.snapshot)?;
            app.status = format!("Approve {}? [y]es / [n]o", call.display_label());
            app.approval = Some(call);
            None
        }
        "waiting_tool" => {
            if args.prompt.is_some() {
                return Err(AppError::PromptWhileResuming {
                    phase: app.snapshot.phase.clone(),
                });
            }
            let call = active_pending_call(&app.snapshot)?;
            let interrupted = core
                .event(&CoreEvent::tool_completed(
                    call.call_id,
                    "Tool execution was interrupted by a process restart and was not replayed."
                        .to_owned(),
                    true,
                ))
                .await?;
            Some(RuntimeAction::Drive(interrupted))
        }
        "waiting_plan_review" => {
            if args.prompt.is_some() {
                return Err(AppError::PromptWhileResuming {
                    phase: app.snapshot.phase.clone(),
                });
            }
            app.plan_review = true;
            app.status = "Plan ready · y approve · r refine · e edit · n cancel".to_owned();
            None
        }
        "waiting_compaction" => {
            if args.prompt.is_some() {
                return Err(AppError::PromptWhileResuming {
                    phase: app.snapshot.phase.clone(),
                });
            }
            let recovered = core.event(&CoreEvent::compaction_failed()).await?;
            Some(RuntimeAction::Drive(recovered))
        }
        phase => {
            return Err(AppError::UnsupportedSessionPhase {
                phase: phase.to_owned(),
            });
        }
    };
    let runtime = Arc::new(Mutex::new(RuntimeResources {
        core,
        plugins,
        provider,
        model_tools,
        desired_safe_tools: safe_tools,
        desired_permission_mode,
        auto_compact_enabled: !args.no_auto_compact,
        auto_compact_threshold,
        sidechain,
    }));
    run_tui(&mut app, Arc::clone(&runtime), initial_action).await?;
    let mut runtime = runtime.lock().await;
    let _ = runtime.core.shutdown().await;
    runtime.plugins.shutdown().await;
    Ok(())
}

fn active_pending_call(snapshot: &CoreSnapshot) -> Result<CoreToolCall, AppError> {
    let phase = snapshot.phase.clone();
    let index =
        usize::try_from(snapshot.current_call).map_err(|_| AppError::MissingPendingTool {
            phase: phase.clone(),
        })?;
    snapshot
        .pending_calls
        .get(index)
        .cloned()
        .ok_or(AppError::MissingPendingTool { phase })
}

fn todos_to_markdown(phases: &[CoreTodoPhase]) -> String {
    let mut output = String::new();
    for phase in phases {
        output.push_str("## ");
        output.push_str(&phase.name);
        output.push('\n');
        for task in &phase.tasks {
            let marker = match task.status.as_str() {
                "in_progress" => ">",
                "completed" => "x",
                "abandoned" => "-",
                "blocked" => "!",
                _ => " ",
            };
            output.push_str(&format!("- [{marker}] {}", task.content));
            if let Some(blocker) = &task.blocker {
                output.push_str(&format!(" <!-- blocker: {blocker} -->"));
            }
            output.push('\n');
        }
        output.push('\n');
    }
    output
}

fn parse_todos_markdown(document: &str) -> Result<Vec<CoreTodoPhase>, AppError> {
    let mut phases: Vec<CoreTodoPhase> = Vec::new();
    let mut phase_names = BTreeMap::new();
    let mut task_names = BTreeMap::new();
    let mut task_count = 0_usize;

    for (index, raw_line) in document.lines().enumerate() {
        let line_number = index + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line.strip_prefix("## ") {
            let name = name.trim();
            if name.is_empty() || name.chars().count() > 200 {
                return Err(AppError::TodoDocument {
                    line: line_number,
                    message: "phase name must contain 1-200 characters".to_owned(),
                });
            }
            if phases.len() >= 16 {
                return Err(AppError::TodoDocument {
                    line: line_number,
                    message: "todo list may contain at most 16 phases".to_owned(),
                });
            }
            if phase_names.insert(name.to_owned(), ()).is_some() {
                return Err(AppError::TodoDocument {
                    line: line_number,
                    message: format!("duplicate phase `{name}`"),
                });
            }
            phases.push(CoreTodoPhase {
                name: name.to_owned(),
                tasks: Vec::new(),
            });
            continue;
        }

        let Some(phase) = phases.last_mut() else {
            return Err(AppError::TodoDocument {
                line: line_number,
                message: "task appears before a `## Phase` heading".to_owned(),
            });
        };
        let (status, remainder) = if let Some(content) = line.strip_prefix("- [ ] ") {
            ("pending", content)
        } else if let Some(content) = line.strip_prefix("- [>] ") {
            ("in_progress", content)
        } else if let Some(content) = line.strip_prefix("- [x] ") {
            ("completed", content)
        } else if let Some(content) = line.strip_prefix("- [-] ") {
            ("abandoned", content)
        } else if let Some(content) = line.strip_prefix("- [!] ") {
            ("blocked", content)
        } else {
            return Err(AppError::TodoDocument {
                line: line_number,
                message: "expected `- [ ]`, `- [>]`, `- [x]`, `- [-]`, or `- [!]`".to_owned(),
            });
        };
        let (content, blocker) = match remainder
            .strip_suffix(" -->")
            .and_then(|text| text.rsplit_once(" <!-- blocker: "))
        {
            Some((content, blocker)) => (content.trim(), Some(blocker.trim().to_owned())),
            None => (remainder.trim(), None),
        };
        if content.is_empty() || content.chars().count() > 200 {
            return Err(AppError::TodoDocument {
                line: line_number,
                message: "task content must contain 1-200 characters".to_owned(),
            });
        }
        if blocker
            .as_ref()
            .is_some_and(|value| value.is_empty() || value.chars().count() > 200)
        {
            return Err(AppError::TodoDocument {
                line: line_number,
                message: "blocker must contain 1-200 characters".to_owned(),
            });
        }
        if task_names.insert(content.to_owned(), ()).is_some() {
            return Err(AppError::TodoDocument {
                line: line_number,
                message: format!("duplicate task `{content}`"),
            });
        }
        task_count += 1;
        if task_count > 64 {
            return Err(AppError::TodoDocument {
                line: line_number,
                message: "todo list may contain at most 64 tasks".to_owned(),
            });
        }
        phase.tasks.push(CoreTodoItem {
            content: content.to_owned(),
            status: status.to_owned(),
            blocker,
        });
    }
    Ok(phases)
}

async fn edit_text_in_external_editor(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    initial: &str,
    suffix: &str,
) -> Result<String, AppError> {
    let path = env::temp_dir().join(format!("mycode-{}-{suffix}.md", Uuid::new_v4().simple()));
    tokio::fs::write(&path, initial.as_bytes()).await?;

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        Show,
        DisableMouseCapture,
        LeaveAlternateScreen
    )?;

    let editor = env::var_os("VISUAL")
        .or_else(|| env::var_os("EDITOR"))
        .unwrap_or_else(|| "vi".into());
    let status_result = Command::new(editor).arg(&path).status().await;
    let content_result = async {
        let file = tokio::fs::File::open(&path).await?;
        let mut bytes = Vec::new();
        file.take((MAX_EDITOR_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .await?;
        if bytes.len() > MAX_EDITOR_BYTES {
            return Err(AppError::EditedContentTooLarge {
                limit: MAX_EDITOR_BYTES,
            });
        }
        Ok(String::from_utf8(bytes)?)
    }
    .await;
    let _ = tokio::fs::remove_file(&path).await;

    execute!(
        terminal.backend_mut(),
        EnterAlternateScreen,
        EnableMouseCapture
    )?;
    enable_raw_mode()?;
    terminal.clear()?;

    let status = status_result?;
    if !status.success() {
        return Err(AppError::EditorFailed { status });
    }
    content_result
}

struct TerminalRestore;

impl Drop for TerminalRestore {
    fn drop(&mut self) {
        restore_terminal();
    }
}

async fn run_tui(
    app: &mut App,
    runtime: SharedRuntime,
    initial_action: Option<RuntimeAction>,
) -> Result<(), AppError> {
    enable_raw_mode()?;
    let restore = TerminalRestore;
    let mut terminal_stdout = std::io::stdout();
    execute!(terminal_stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(terminal_stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = tui_loop(&mut terminal, app, runtime, initial_action).await;
    let cursor_result = terminal.show_cursor();
    drop(terminal);
    drop(restore);
    match result {
        Err(error) => Err(error),
        Ok(()) => {
            cursor_result?;
            Ok(())
        }
    }
}

async fn tui_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
    runtime: SharedRuntime,
    initial_action: Option<RuntimeAction>,
) -> Result<(), AppError> {
    let mut events = EventStream::new();
    let (updates, mut update_rx) = mpsc::unbounded_channel();
    let mut active = initial_action.map(|action| {
        app.busy = true;
        app.status = "Starting…".to_owned();
        spawn_runtime_action(Arc::clone(&runtime), action, updates.clone())
    });
    let mut frame_tick = interval(FRAME_INTERVAL);
    frame_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut dirty = true;
    let mut pending_scroll = 0_i32;

    loop {
        tokio::select! {
            event = events.next() => {
                let Some(event) = event else {
                    cancel_and_wait(&mut active, app).await;
                    return Ok(());
                };
                match event? {
                    CrosstermEvent::Key(key)
                        if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) => {
                        if pending_scroll != 0 {
                            app.scroll_delta(pending_scroll);
                            pending_scroll = 0;
                        }
                        dirty = true;
                        match handle_key(key, app) {
                            KeyAction::None => {}
                            KeyAction::Quit => {
                                cancel_and_wait(&mut active, app).await;
                                return Ok(());
                            }
                            KeyAction::Cancel => request_cancel(&mut active, app),
                            KeyAction::Steer(text) => {
                                if let Some(active_runtime) = active.as_ref() {
                                    match active_runtime.steer.send(text) {
                                        Ok(()) => app.status = "Steering…".to_owned(),
                                        Err(error) => {
                                            app.set_input(error.0);
                                            app.status =
                                                "Task settled before steer arrived · press Enter to send"
                                                    .to_owned();
                                        }
                                    }
                                } else {
                                    app.busy = true;
                                    app.approval = None;
                                    app.status = "Steering…".to_owned();
                                    active = Some(spawn_runtime_action(
                                        Arc::clone(&runtime),
                                        RuntimeAction::Steer(text),
                                        updates.clone(),
                                    ));
                                }
                            }
                            KeyAction::Submit(text) if active.is_none() => {
                                app.busy = true;
                                app.status = "Submitting…".to_owned();
                                active = Some(spawn_runtime_action(
                                    Arc::clone(&runtime),
                                    RuntimeAction::Submit(text),
                                    updates.clone(),
                                ));
                            }
                            KeyAction::Compact(instructions) if active.is_none() => {
                                app.busy = true;
                                app.status = "Compacting context…".to_owned();
                                active = Some(spawn_runtime_action(
                                    Arc::clone(&runtime),
                                    RuntimeAction::Compact(instructions),
                                    updates.clone(),
                                ));
                            }
                            KeyAction::EnterPlan(text) if active.is_none() => {
                                app.busy = true;
                                app.status = "Entering plan mode…".to_owned();
                                active = Some(spawn_runtime_action(
                                    Arc::clone(&runtime),
                                    RuntimeAction::EnterPlan(text),
                                    updates.clone(),
                                ));
                            }
                            KeyAction::EditTodos if active.is_none() => {
                                let initial = todos_to_markdown(&app.snapshot.todos);
                                match edit_text_in_external_editor(terminal, &initial, "todos").await {
                                    Ok(document) => match parse_todos_markdown(&document) {
                                        Ok(todos) => {
                                            app.busy = true;
                                            app.status = "Saving todos…".to_owned();
                                            active = Some(spawn_runtime_action(
                                                Arc::clone(&runtime),
                                                RuntimeAction::ReplaceTodos(todos),
                                                updates.clone(),
                                            ));
                                        }
                                        Err(error) => app.status = format!("Error: {error}"),
                                    },
                                    Err(error) => app.status = format!("Error: {error}"),
                                }
                            }
                            KeyAction::EditPlan if active.is_none() => {
                                let initial = app.snapshot.plan.content.clone();
                                match edit_text_in_external_editor(terminal, &initial, "plan").await {
                                    Ok(content) if content != initial => {
                                        app.busy = true;
                                        app.plan_review = false;
                                        app.status = "Saving plan revision…".to_owned();
                                        active = Some(spawn_runtime_action(
                                            Arc::clone(&runtime),
                                            RuntimeAction::PlanReview(PlanReviewAction::Edit(content)),
                                            updates.clone(),
                                        ));
                                    }
                                    Ok(_) => app.status =
                                        "Plan unchanged · y approve · r refine · e edit · n cancel".to_owned(),
                                    Err(error) => app.status = format!("Error: {error}"),
                                }
                            }
                            KeyAction::PlanReview(action) if active.is_none() => {
                                app.busy = true;
                                app.plan_review = false;
                                app.plan_feedback = false;
                                app.follow_tail = true;
                                app.status = "Applying plan decision…".to_owned();
                                active = Some(spawn_runtime_action(
                                    Arc::clone(&runtime),
                                    RuntimeAction::PlanReview(action),
                                    updates.clone(),
                                ));
                            }
                            KeyAction::Btw(question) if active.is_none() => {
                                app.busy = true;
                                app.status = "Asking BTW…".to_owned();
                                active = Some(spawn_runtime_action(
                                    Arc::clone(&runtime),
                                    RuntimeAction::Btw {
                                        question,
                                        snapshot: app.snapshot.clone(),
                                    },
                                    updates.clone(),
                                ));
                            }
                            KeyAction::Approval { call_id, approved } if active.is_none() => {
                                app.busy = true;
                                app.approval = None;
                                app.status = "Applying approval…".to_owned();
                                active = Some(spawn_runtime_action(
                                    Arc::clone(&runtime),
                                    RuntimeAction::Approval { call_id, approved },
                                    updates.clone(),
                                ));
                            }
                            KeyAction::Submit(_)
                            | KeyAction::Compact(_)
                            | KeyAction::EnterPlan(_)
                            | KeyAction::EditTodos
                            | KeyAction::PlanReview(_)
                            | KeyAction::EditPlan
                            | KeyAction::Btw(_)
                            | KeyAction::Approval { .. } => {}
                        }
                    }
                    CrosstermEvent::Mouse(mouse) => match app.handle_mouse(mouse) {
                        MouseAction::None => {}
                        MouseAction::Redraw => dirty = true,
                        MouseAction::Scroll(delta) => {
                            pending_scroll = pending_scroll.saturating_add(delta);
                            dirty = true;
                        }
                        MouseAction::Copy(text) => {
                            let character_count = text.chars().count();
                            execute!(
                                std::io::stdout(),
                                CopyToClipboard::to_clipboard_from(text)
                            )?;
                            app.status = format!("Copied {character_count} characters");
                            dirty = true;
                        }
                        MouseAction::SelectionTooLarge => {
                            app.status = format!(
                                "Selection exceeds the {} byte clipboard limit",
                                MAX_SELECTION_BYTES
                            );
                            dirty = true;
                        }
                    },
                    CrosstermEvent::Resize(_, _) => dirty = true,
                    _ => {}
                }
            }
            update = update_rx.recv() => {
                match update {
                    Some(RuntimeUpdate::Progress { snapshot, status }) => {
                        app.set_snapshot(snapshot);
                        app.status = status;
                        app.busy = true;
                        dirty = true;
                    }
                    Some(RuntimeUpdate::ToolStarted { call_id, label }) => {
                        app.start_live_tool(call_id, label.clone());
                        app.status = format!("Running {label}…");
                        app.busy = true;
                        dirty = true;
                    }
                    Some(RuntimeUpdate::ToolProgress { call_id, progress }) => {
                        app.append_live_progress(&call_id, progress)?;
                        app.busy = true;
                        dirty = true;
                    }
                    Some(RuntimeUpdate::ToolFinished { call_id }) => {
                        app.finish_live_tool(&call_id);
                        dirty = true;
                    }
                    Some(RuntimeUpdate::Settled(final_state)) => {
                        if let Some(active) = active.take() {
                            let _ = active.handle.await;
                        }
                        let RuntimeFinal {
                            snapshot,
                            status,
                            approval,
                            plan_review,
                            local_entry,
                        } = final_state;
                        if plan_review && !app.plan_review {
                            app.scroll_offset = 0;
                            app.follow_tail = false;
                        }
                        app.set_snapshot(snapshot);
                        if let Some(entry) = local_entry {
                            app.push_local_entry(entry);
                        }
                        app.status = status;
                        app.approval = approval;
                        app.plan_review = plan_review;
                        app.plan_feedback = false;
                        app.busy = false;
                        app.clear_live_tool();
                        dirty = true;
                    }
                    Some(RuntimeUpdate::Failed(error)) => {
                        if let Some(active) = active.take() {
                            let _ = active.handle.await;
                        }
                        app.status = format!("Error: {error}");
                        app.busy = false;
                        app.plan_review = app.snapshot.phase == "waiting_plan_review";
                        app.plan_feedback = false;
                        app.clear_live_tool();
                        dirty = true;
                    }
                    None => return Ok(()),
                }
            }
            _ = frame_tick.tick(), if dirty || pending_scroll != 0 => {
                if pending_scroll != 0 {
                    app.scroll_delta(pending_scroll);
                    pending_scroll = 0;
                }
                terminal.draw(|frame| draw(frame, app))?;
                dirty = false;
            }
        }
    }
}

fn handle_input_navigation(key: KeyEvent, app: &mut App) -> bool {
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    match key.code {
        KeyCode::Left if alt || control => app.input.move_word_left(),
        KeyCode::Right if alt || control => app.input.move_word_right(),
        KeyCode::Left => app.input.move_left(),
        KeyCode::Right => app.input.move_right(),
        KeyCode::Home if control => app.input.move_start(),
        KeyCode::End if control => app.input.move_end(),
        KeyCode::Home if !app.input.is_empty() => app.input.move_line_start(),
        KeyCode::End if !app.input.is_empty() => app.input.move_line_end(),
        KeyCode::Char('b') if control => app.input.move_left(),
        KeyCode::Char('f') if control => app.input.move_right(),
        KeyCode::Char('b') if alt => app.input.move_word_left(),
        KeyCode::Char('f') if alt => app.input.move_word_right(),
        KeyCode::Char('a') if control => app.input.move_line_start(),
        KeyCode::Char('e') if control => app.input.move_line_end(),
        KeyCode::Char('w') if control => app.delete_input_word_left(),
        KeyCode::Char('d') if alt => app.delete_input_word_right(),
        KeyCode::Backspace if alt || control => app.delete_input_word_left(),
        KeyCode::Delete => app.delete_input_character(),
        _ => return false,
    }
    true
}

fn handle_key(key: KeyEvent, app: &mut App) -> KeyAction {
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    if control && matches!(key.code, KeyCode::Char('c')) {
        return KeyAction::Quit;
    }
    if control && matches!(key.code, KeyCode::Char('o')) {
        app.toggle_transcript_details();
        return KeyAction::None;
    }
    if app.plan_review {
        if app.plan_feedback {
            if handle_input_navigation(key, app) {
                return KeyAction::None;
            }
            return match key.code {
                KeyCode::Enter if !app.input.as_str().trim().is_empty() => {
                    let feedback = app.input.as_str().trim().to_owned();
                    app.clear_input();
                    app.plan_feedback = false;
                    KeyAction::PlanReview(PlanReviewAction::Refine(feedback))
                }
                KeyCode::Backspace => {
                    app.pop_input_character();
                    KeyAction::None
                }
                KeyCode::Esc => {
                    app.clear_input();
                    app.plan_feedback = false;
                    app.status = "Plan ready · y approve · r refine · e edit · n cancel".to_owned();
                    KeyAction::None
                }
                KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    app.push_input_character(character);
                    KeyAction::None
                }
                _ => KeyAction::None,
            };
        }
        return match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                KeyAction::PlanReview(PlanReviewAction::Approve)
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                app.plan_feedback = true;
                app.clear_input();
                app.status = "Describe the plan changes, then press Enter".to_owned();
                KeyAction::None
            }
            KeyCode::Char('e') | KeyCode::Char('E') => KeyAction::EditPlan,
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                KeyAction::PlanReview(PlanReviewAction::Cancel)
            }
            _ => KeyAction::None,
        };
    }
    if app.approval.is_none() {
        match key.code {
            KeyCode::Up if app.move_slash_selection(true) => return KeyAction::None,
            KeyCode::Down if app.move_slash_selection(false) => return KeyAction::None,
            _ => {}
        }
    }

    let input_active = app.approval.is_none() || !app.input.is_empty();
    if input_active && handle_input_navigation(key, app) {
        return KeyAction::None;
    }

    match key.code {
        KeyCode::Up => {
            app.scroll_up(1);
            return KeyAction::None;
        }
        KeyCode::Down => {
            app.scroll_down(1);
            return KeyAction::None;
        }
        KeyCode::PageUp => {
            app.page_up();
            return KeyAction::None;
        }
        KeyCode::PageDown => {
            app.page_down();
            return KeyAction::None;
        }
        KeyCode::Home => {
            app.scroll_home();
            return KeyAction::None;
        }
        KeyCode::End => {
            app.scroll_end();
            return KeyAction::None;
        }
        _ => {}
    }
    if app.busy && matches!(key.code, KeyCode::Esc) {
        return KeyAction::Cancel;
    }
    if app.approval.is_some() && app.input.is_empty() {
        return match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                let call = app.approval.take().expect("approval checked");
                KeyAction::Approval {
                    call_id: call.call_id,
                    approved: true,
                }
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                let call = app.approval.take().expect("approval checked");
                KeyAction::Approval {
                    call_id: call.call_id,
                    approved: false,
                }
            }
            _ => KeyAction::None,
        };
    }
    if matches!(key.code, KeyCode::Esc) && app.dismiss_slash_menu() {
        return KeyAction::None;
    }
    match key.code {
        KeyCode::Enter if !app.input.as_str().trim().is_empty() => {
            let parsed = parse_submission(app.input.as_str().to_owned());
            match parsed {
                Ok(UserSubmission::Command(SlashCommand::Exit)) => {
                    app.clear_input();
                    KeyAction::Quit
                }
                Ok(UserSubmission::Command(SlashCommand::Compact(instructions))) => {
                    app.clear_input();
                    if app.snapshot.phase == "idle" && !app.busy {
                        KeyAction::Compact(instructions)
                    } else {
                        app.status =
                            "Error: context can be compacted manually only while idle".to_owned();
                        KeyAction::None
                    }
                }
                Ok(UserSubmission::Command(SlashCommand::Steer(instruction))) => {
                    app.clear_input();
                    if snapshot_accepts_steer(&app.snapshot) {
                        KeyAction::Steer(instruction)
                    } else {
                        app.status = "Error: no active main task to steer".to_owned();
                        KeyAction::None
                    }
                }
                Ok(UserSubmission::Command(SlashCommand::Plan(goal))) => {
                    app.clear_input();
                    if app.snapshot.phase == "idle" && !app.busy {
                        KeyAction::EnterPlan(goal)
                    } else {
                        app.status = "Error: plan mode can start only while idle".to_owned();
                        KeyAction::None
                    }
                }
                Ok(UserSubmission::Command(SlashCommand::Todo)) => {
                    app.clear_input();
                    if app.snapshot.phase == "idle" && !app.busy {
                        KeyAction::EditTodos
                    } else {
                        app.status =
                            "Error: todos can be edited directly only while idle".to_owned();
                        KeyAction::None
                    }
                }
                Ok(UserSubmission::Prompt(_))
                    if app.busy && !snapshot_accepts_steer(&app.snapshot) =>
                {
                    KeyAction::None
                }
                Ok(UserSubmission::Prompt(prompt)) if snapshot_accepts_steer(&app.snapshot) => {
                    app.clear_input();
                    KeyAction::Steer(prompt)
                }
                Ok(UserSubmission::Prompt(prompt)) => {
                    app.clear_input();
                    KeyAction::Submit(prompt)
                }
                Ok(UserSubmission::Command(SlashCommand::Btw(_)))
                    if app.busy || snapshot_accepts_steer(&app.snapshot) =>
                {
                    KeyAction::None
                }
                Ok(UserSubmission::Command(SlashCommand::Btw(question))) => {
                    app.clear_input();
                    KeyAction::Btw(question)
                }
                Err(error) => {
                    app.clear_input();
                    app.status = format!("Error: {error}");
                    KeyAction::None
                }
            }
        }
        KeyCode::Tab => {
            app.complete_selected_slash_command();
            KeyAction::None
        }
        KeyCode::Backspace => {
            app.pop_input_character();
            KeyAction::None
        }
        KeyCode::Esc => {
            app.clear_input();
            KeyAction::None
        }
        KeyCode::Char(character) if !control => {
            app.push_input_character(character);
            KeyAction::None
        }
        _ => KeyAction::None,
    }
}

fn spawn_runtime_action(
    runtime: SharedRuntime,
    action: RuntimeAction,
    updates: mpsc::UnboundedSender<RuntimeUpdate>,
) -> ActiveRuntime {
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let (steer, mut steer_rx) = mpsc::unbounded_channel();
    let handle = tokio::spawn(async move {
        match run_runtime_action(&runtime, action, &updates, &task_cancel, &mut steer_rx).await {
            Ok(final_state) => {
                let _ = updates.send(RuntimeUpdate::Settled(final_state));
            }
            Err(error) => {
                let _ = updates.send(RuntimeUpdate::Failed(error.to_string()));
            }
        }
    });
    ActiveRuntime {
        handle,
        cancel,
        steer,
    }
}

async fn run_runtime_action(
    runtime: &SharedRuntime,
    action: RuntimeAction,
    updates: &mpsc::UnboundedSender<RuntimeUpdate>,
    cancel: &CancellationToken,
    steer: &mut mpsc::UnboundedReceiver<String>,
) -> Result<RuntimeFinal, AppError> {
    let mut runtime = runtime.lock().await;
    match action {
        RuntimeAction::Btw { question, snapshot } => {
            let answer = tokio::select! {
                () = cancel.cancelled() => {
                    return Ok(RuntimeFinal {
                        snapshot,
                        status: "BTW cancelled".to_owned(),
                        approval: None,
                        plan_review: false,
                        local_entry: None,
                    });
                }
                result = runtime.provider.complete_text_only(
                    &snapshot,
                    &runtime.sidechain.entries,
                    &question,
                ) => result?,
            };
            let entry = LocalTranscriptEntry { question, answer };
            runtime.sidechain.append(entry.clone()).await?;
            Ok(RuntimeFinal {
                snapshot,
                status: "Ready".to_owned(),
                approval: None,
                plan_review: false,
                local_entry: Some(entry),
            })
        }
        RuntimeAction::Submit(text) => {
            let response = runtime.core.event(&CoreEvent::submit(text)).await?;
            drive_response(response, &mut runtime, updates, cancel, steer).await
        }
        RuntimeAction::Compact(instructions) => {
            let response = runtime
                .core
                .event(&CoreEvent::start_compaction(instructions, 0, false))
                .await?;
            drive_response(response, &mut runtime, updates, cancel, steer).await
        }
        RuntimeAction::EnterPlan(text) => {
            let response = runtime.core.event(&CoreEvent::enter_plan(text)).await?;
            drive_response(response, &mut runtime, updates, cancel, steer).await
        }
        RuntimeAction::ReplaceTodos(todos) => {
            let response = runtime.core.event(&CoreEvent::replace_todos(todos)).await?;
            drive_response(response, &mut runtime, updates, cancel, steer).await
        }
        RuntimeAction::PlanReview(action) => {
            let event = match action {
                PlanReviewAction::Approve => CoreEvent::approve_plan(),
                PlanReviewAction::Refine(feedback) => CoreEvent::refine_plan(feedback),
                PlanReviewAction::Edit(content) => CoreEvent::edit_plan(content),
                PlanReviewAction::Cancel => CoreEvent::cancel_plan_review(),
            };
            let response = runtime.core.event(&event).await?;
            drive_response(response, &mut runtime, updates, cancel, steer).await
        }
        RuntimeAction::Steer(text) => {
            let response = runtime.core.event(&CoreEvent::steer(text)).await?;
            drive_response(response, &mut runtime, updates, cancel, steer).await
        }
        RuntimeAction::Approval { call_id, approved } => {
            let response = runtime
                .core
                .event(&CoreEvent::approval(call_id, approved))
                .await?;
            drive_response(response, &mut runtime, updates, cancel, steer).await
        }
        RuntimeAction::Drive(response) => {
            drive_response(response, &mut runtime, updates, cancel, steer).await
        }
    }
}

fn compaction_cut(snapshot: &CoreSnapshot) -> Option<usize> {
    let mut users = snapshot
        .messages
        .iter()
        .enumerate()
        .rev()
        .filter_map(|(index, message)| (message.role == "user").then_some(index));
    let _latest = users.next()?;
    let second_latest = users.next()?;
    (u64::try_from(second_latest).ok()? > snapshot.compaction.first_kept_message)
        .then_some(second_latest)
}

fn should_auto_compact(runtime: &RuntimeResources, snapshot: &CoreSnapshot) -> bool {
    runtime.auto_compact_enabled
        && runtime.auto_compact_threshold > 0
        && snapshot.compaction.last_input_tokens >= runtime.auto_compact_threshold
        && compaction_cut(snapshot).is_some()
}

async fn drive_response(
    response: CoreResponse,
    runtime: &mut RuntimeResources,
    updates: &mpsc::UnboundedSender<RuntimeUpdate>,
    cancel: &CancellationToken,
    steer: &mut mpsc::UnboundedReceiver<String>,
) -> Result<RuntimeFinal, AppError> {
    let mut snapshot = response.snapshot;
    let mut effects = response.effects;
    let mut final_status = "Ready".to_owned();
    send_runtime_progress(updates, &snapshot, "Processing…");
    loop {
        while let Some(effect) = effects.pop() {
            if cancel.is_cancelled() {
                return cancel_runtime(runtime).await;
            }
            match effect.kind.as_str() {
                "request_model" => {
                    if should_auto_compact(runtime, &snapshot) {
                        let next = runtime
                            .core
                            .event(&CoreEvent::start_compaction(
                                None,
                                snapshot.compaction.last_input_tokens,
                                true,
                            ))
                            .await?;
                        snapshot = next.snapshot;
                        effects.clear();
                        effects.extend(next.effects);
                        send_runtime_progress(updates, &snapshot, "Auto-compacting context…");
                        continue;
                    }

                    send_runtime_progress(updates, &snapshot, "Waiting for model…");
                    let completion_result = tokio::select! {
                        biased;
                        () = cancel.cancelled() => return cancel_runtime(runtime).await,
                        Some(text) = steer.recv() => {
                            let next = apply_steer_event(runtime, &snapshot, text).await?;
                            snapshot = next.snapshot;
                            effects.clear();
                            effects.extend(next.effects);
                            send_runtime_progress(updates, &snapshot, "Steering…");
                            continue;
                        }
                        result = runtime.provider.complete(
                            &snapshot,
                            if snapshot.plan.enabled {
                                &runtime.model_tools.plan
                            } else {
                                &runtime.model_tools.normal
                            },
                            &runtime.model_tools.declared,
                        ) => result,
                    };
                    let completion = match completion_result {
                        Ok(completion) => completion,
                        Err(error)
                            if error.is_context_overflow()
                                && compaction_cut(&snapshot).is_some() =>
                        {
                            let next = runtime
                                .core
                                .event(&CoreEvent::start_compaction(
                                    None,
                                    snapshot.compaction.last_input_tokens,
                                    true,
                                ))
                                .await?;
                            snapshot = next.snapshot;
                            effects.clear();
                            effects.extend(next.effects);
                            send_runtime_progress(
                                updates,
                                &snapshot,
                                "Recovering context overflow…",
                            );
                            continue;
                        }
                        Err(error) => return Err(error.into()),
                    };
                    let next = runtime
                        .core
                        .event(&CoreEvent::model_completed(
                            completion.content,
                            completion.tool_calls,
                            completion.input_tokens,
                        ))
                        .await?;
                    snapshot = next.snapshot;
                    effects.extend(next.effects);
                }
                "request_compaction" => {
                    send_runtime_progress(updates, &snapshot, "Summarizing old context…");
                    let summary_result = tokio::select! {
                        () = cancel.cancelled() => return cancel_runtime(runtime).await,
                        result = runtime.provider.summarize_compaction(&snapshot) => result,
                    };
                    let next = match summary_result {
                        Ok(summary) => {
                            runtime
                                .core
                                .event(&CoreEvent::compaction_completed(summary))
                                .await?
                        }
                        Err(error) => {
                            final_status = format!("Compaction failed: {error}");
                            runtime.core.event(&CoreEvent::compaction_failed()).await?
                        }
                    };
                    snapshot = next.snapshot;
                    effects.extend(next.effects);
                }
                "request_approval" => {
                    if let Ok(text) = steer.try_recv() {
                        let next = apply_steer_event(runtime, &snapshot, text).await?;
                        snapshot = next.snapshot;
                        effects.clear();
                        effects.extend(next.effects);
                        send_runtime_progress(updates, &snapshot, "Steering…");
                        continue;
                    }
                    let call = effect.call.ok_or_else(|| AppError::MissingCoreEffectCall {
                        kind: "request_approval".to_owned(),
                    })?;
                    return Ok(RuntimeFinal {
                        snapshot,
                        status: format!("Approve {}? [y]es / [n]o", call.display_label()),
                        approval: Some(call),
                        plan_review: false,
                        local_entry: None,
                    });
                }
                "request_plan_review" => {
                    return Ok(RuntimeFinal {
                        snapshot,
                        status: "Plan ready · y approve · r refine · e edit · n cancel".to_owned(),
                        approval: None,
                        plan_review: true,
                        local_entry: None,
                    });
                }
                "invoke_tool" => {
                    let call = effect.call.ok_or_else(|| AppError::MissingCoreEffectCall {
                        kind: "invoke_tool".to_owned(),
                    })?;
                    let _ = updates.send(RuntimeUpdate::ToolStarted {
                        call_id: call.call_id.clone(),
                        label: call.display_label().to_owned(),
                    });
                    send_runtime_progress(
                        updates,
                        &snapshot,
                        &format!("Running {}…", call.display_label()),
                    );
                    let progress_call_id = call.call_id.clone();
                    let tool_result = {
                        let plugin_call = runtime.plugins.call(&call, |progress| {
                            let _ = updates.send(RuntimeUpdate::ToolProgress {
                                call_id: progress_call_id.clone(),
                                progress,
                            });
                        });
                        tokio::pin!(plugin_call);
                        loop {
                            tokio::select! {
                                biased;
                                () = cancel.cancelled() => break None,
                                Some(text) = steer.recv() => {
                                    let queued = runtime.core.event(&CoreEvent::steer(text)).await?;
                                    snapshot = queued.snapshot;
                                    send_runtime_progress(
                                        updates,
                                        &snapshot,
                                        "Steer queued · finishing current tool…",
                                    );
                                }
                                result = &mut plugin_call => break Some(result),
                            }
                        }
                    };
                    let Some(tool_result) = tool_result else {
                        return cancel_runtime(runtime).await;
                    };
                    let (content, is_error) = match tool_result {
                        Ok(result) => (result.output, false),
                        Err(error) => (format!("Plugin execution failed: {error}"), true),
                    };
                    let _ = updates.send(RuntimeUpdate::ToolFinished {
                        call_id: call.call_id.clone(),
                    });
                    let next = runtime
                        .core
                        .event(&CoreEvent::tool_completed(call.call_id, content, is_error))
                        .await?;
                    snapshot = next.snapshot;
                    effects.extend(next.effects);
                }
                _ => return Err(AppError::UnknownCoreEffect { kind: effect.kind }),
            }
        }
        if snapshot.phase == "idle" && should_auto_compact(runtime, &snapshot) {
            let next = runtime
                .core
                .event(&CoreEvent::start_compaction(
                    None,
                    snapshot.compaction.last_input_tokens,
                    true,
                ))
                .await?;
            snapshot = next.snapshot;
            effects.extend(next.effects);
            send_runtime_progress(updates, &snapshot, "Auto-compacting context…");
            continue;
        }
        let Ok(text) = steer.try_recv() else {
            break;
        };
        let next = apply_steer_event(runtime, &snapshot, text).await?;
        snapshot = next.snapshot;
        effects.extend(next.effects);
        send_runtime_progress(updates, &snapshot, "Steering…");
    }
    apply_desired_permissions(runtime, &mut snapshot).await?;
    let plan_review = snapshot.phase == "waiting_plan_review";
    Ok(RuntimeFinal {
        snapshot,
        status: final_status,
        approval: None,
        plan_review,
        local_entry: None,
    })
}

async fn apply_steer_event(
    runtime: &mut RuntimeResources,
    snapshot: &CoreSnapshot,
    text: String,
) -> Result<CoreResponse, AppError> {
    let event = if snapshot_accepts_steer(snapshot) {
        CoreEvent::steer(text)
    } else {
        CoreEvent::submit(text)
    };
    Ok(runtime.core.event(&event).await?)
}

async fn cancel_runtime(runtime: &mut RuntimeResources) -> Result<RuntimeFinal, AppError> {
    runtime.plugins.retire().await;
    let response = runtime.core.event(&CoreEvent::abort()).await?;
    let mut snapshot = response.snapshot;
    apply_desired_permissions(runtime, &mut snapshot).await?;
    Ok(RuntimeFinal {
        snapshot,
        status: "Cancelled".to_owned(),
        approval: None,
        plan_review: false,
        local_entry: None,
    })
}

async fn apply_desired_permissions(
    runtime: &mut RuntimeResources,
    snapshot: &mut CoreSnapshot,
) -> Result<(), AppError> {
    let desired = runtime.desired_permission_mode.wire_value();
    if snapshot.phase == "idle" && snapshot.permission_mode != desired {
        let configured = runtime
            .core
            .event(&CoreEvent::configure_tools(
                runtime.desired_safe_tools.clone(),
                runtime.desired_permission_mode,
            ))
            .await?;
        *snapshot = configured.snapshot;
    }
    Ok(())
}

fn send_runtime_progress(
    updates: &mpsc::UnboundedSender<RuntimeUpdate>,
    snapshot: &CoreSnapshot,
    status: &str,
) {
    let _ = updates.send(RuntimeUpdate::Progress {
        snapshot: snapshot.clone(),
        status: status.to_owned(),
    });
}

fn request_cancel(active: &mut Option<ActiveRuntime>, app: &mut App) {
    if let Some(active) = active.as_ref() {
        active.cancel.cancel();
        app.status = "Cancelling…".to_owned();
        app.busy = true;
    }
}

async fn cancel_and_wait(active: &mut Option<ActiveRuntime>, app: &mut App) {
    let Some(active) = active.take() else {
        return;
    };
    active.cancel.cancel();
    app.status = "Cancelling…".to_owned();
    let _ = active.handle.await;
}
fn slash_candidate_lines(app: &App, max_rows: usize) -> Vec<Line<'static>> {
    let count = app.slash_candidate_count();
    let rows = count.min(max_rows);
    let start = app.slash_selection.saturating_add(1).saturating_sub(rows);
    slash_command_candidates(app.input.as_str())
        .enumerate()
        .skip(start)
        .take(rows)
        .map(|(index, spec)| {
            let selected = index == app.slash_selection;
            Line::from(vec![
                Span::styled(
                    if selected { " › " } else { "   " },
                    Style::default()
                        .fg(if selected { COLOR_ACCENT } else { COLOR_MUTED })
                        .add_modifier(if selected {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::styled(
                    spec.usage,
                    Style::default().fg(COLOR_TEXT).add_modifier(if selected {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
                ),
                Span::raw("  "),
                Span::styled(spec.description, Style::default().fg(COLOR_MUTED)),
            ])
        })
        .collect()
}

fn todo_panel_lines(snapshot: &CoreSnapshot, width: u16) -> Vec<Line<'static>> {
    let content_width = width.saturating_sub(2).max(1);
    let mut lines = Vec::new();
    if snapshot.plan.status != "none" {
        lines.push(Line::from(Span::styled(
            format!(
                "Plan r{} · {}",
                snapshot.plan.revision, snapshot.plan.status
            ),
            Style::default()
                .fg(if snapshot.plan.enabled {
                    COLOR_ACCENT
                } else {
                    COLOR_MUTED
                })
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::default());
    }
    for phase in &snapshot.todos {
        lines.push(Line::from(Span::styled(
            phase.name.clone(),
            Style::default().fg(COLOR_TEXT).add_modifier(Modifier::BOLD),
        )));
        for task in &phase.tasks {
            let (marker, color) = match task.status.as_str() {
                "in_progress" => (">", COLOR_ACCENT),
                "completed" => ("x", COLOR_ASSISTANT),
                "abandoned" => ("-", COLOR_MUTED),
                "blocked" => ("!", COLOR_ERROR),
                _ => (" ", COLOR_MUTED),
            };
            push_wrapped_line(
                &mut lines,
                &format!("[{marker}] {}", task.content),
                Style::default().fg(color),
                content_width,
            );
            if let Some(blocker) = &task.blocker {
                push_wrapped_line(
                    &mut lines,
                    &format!("    blocked: {blocker}"),
                    Style::default().fg(COLOR_ERROR),
                    content_width,
                );
            }
        }
        lines.push(Line::default());
    }
    if snapshot.todos.is_empty() {
        lines.push(Line::from(Span::styled(
            "No todos. Run /todo or let the agent create them.",
            Style::default().fg(COLOR_MUTED),
        )));
    }
    lines
}

fn draw(frame: &mut ratatui::Frame<'_>, app: &mut App) {
    let area = frame.area();
    let candidate_count = if app.approval.is_none() && app.slash_menu_visible() {
        app.slash_candidate_count()
    } else {
        0
    };
    let max_candidate_height = area.height.saturating_sub(1 + 4 + 3 + 1);
    let desired_candidate_height = u16::try_from(candidate_count)
        .unwrap_or(u16::MAX)
        .saturating_add(2);
    let candidate_height = if candidate_count > 0 && max_candidate_height >= 3 {
        desired_candidate_height.min(max_candidate_height)
    } else {
        0
    };
    let input_layout = InputLayout::new(&app.input, area.width.saturating_sub(2));
    let input_height =
        input_layout.render_height(area.height.saturating_sub(1 + 4 + candidate_height + 1));
    let [header, transcript, candidates, input, status] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(4),
        Constraint::Length(candidate_height),
        Constraint::Length(input_height),
        Constraint::Length(1),
    ])
    .areas(area);

    let phase_color = match app.snapshot.phase.as_str() {
        "idle" => COLOR_ASSISTANT,
        "waiting_approval" | "waiting_plan_review" => COLOR_TOOL,
        "waiting_model" | "waiting_tool" | "waiting_compaction" => COLOR_ACCENT,
        _ => COLOR_MUTED,
    };
    let mut header_spans = vec![
        Span::styled(
            " mycode ",
            Style::default()
                .fg(COLOR_TEXT)
                .bg(COLOR_ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            app.snapshot.phase.clone(),
            Style::default()
                .fg(phase_color)
                .add_modifier(Modifier::BOLD),
        ),
    ];
    if app.snapshot.plan.status != "none" {
        header_spans.push(Span::styled(
            format!(
                "  plan r{} · {}",
                app.snapshot.plan.revision, app.snapshot.plan.status
            ),
            Style::default().fg(if app.snapshot.plan.enabled {
                COLOR_ACCENT
            } else {
                COLOR_MUTED
            }),
        ));
    }
    if app.snapshot.compaction.revision > 0 {
        header_spans.push(Span::styled(
            format!("  compact r{}", app.snapshot.compaction.revision),
            Style::default().fg(COLOR_MUTED),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(header_spans)), header);

    let show_todos = !app.snapshot.todos.is_empty() && transcript.width >= 72;
    let mut main_area = transcript;
    let mut todo_area = None;
    if show_todos {
        let sidebar_width = (transcript.width / 3).clamp(26, 40);
        let areas = Layout::horizontal([Constraint::Min(40), Constraint::Length(sidebar_width)])
            .split(transcript);
        main_area = areas[0];
        todo_area = Some(areas[1]);
    }

    if app.plan_review {
        app.update_selection_content(None, &Text::default());
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(COLOR_ACCENT))
            .title(Span::styled(
                " Plan review · y approve · r refine · e edit · n cancel ",
                Style::default()
                    .fg(COLOR_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(main_area);
        let visible_plan = app.visible_plan(inner.width, inner.height);
        frame.render_widget(
            Paragraph::new(visible_plan)
                .wrap(Wrap { trim: false })
                .block(block),
            main_area,
        );
    } else {
        let content_width = main_area.width.saturating_sub(2);
        let content_height = main_area.height;
        let visible_transcript = app.visible_transcript(content_width, content_height);
        let selection_region = SelectionRegion::from_transcript_rect(main_area);
        app.update_selection_content(selection_region, &visible_transcript);
        frame.render_widget(
            Paragraph::new(visible_transcript)
                .block(Block::default().padding(Padding::horizontal(1))),
            main_area,
        );
        if let (Some(selection), Some(region)) = (app.text_selection, app.selection_region) {
            frame.render_widget(SelectionOverlay { selection, region }, main_area);
        }
    }
    if let Some(todo_area) = todo_area {
        frame.render_widget(
            Paragraph::new(todo_panel_lines(&app.snapshot, todo_area.width))
                .wrap(Wrap { trim: false })
                .block(
                    Block::default()
                        .borders(Borders::LEFT)
                        .border_style(Style::default().fg(COLOR_MUTED))
                        .title(Span::styled(
                            " Todos · /todo edit ",
                            Style::default().fg(COLOR_MUTED),
                        )),
                ),
            todo_area,
        );
    }
    if candidate_height > 0 {
        let candidate_rows = usize::from(candidates.height.saturating_sub(2));
        frame.render_widget(
            Paragraph::new(slash_candidate_lines(app, candidate_rows)).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(COLOR_ACCENT))
                    .title(Span::styled(
                        " Slash commands · ↑↓ select · Tab complete · Esc close ",
                        Style::default()
                            .fg(COLOR_ACCENT)
                            .add_modifier(Modifier::BOLD),
                    )),
            ),
            candidates,
        );
    }

    let (input_title, input_color) = match app.approval.as_ref() {
        Some(call) => (
            format!(
                " Approval · {} · y allow · n deny · /steer redirect ",
                call.display_label()
            ),
            COLOR_TOOL,
        ),
        None if app.plan_feedback => (
            " Plan feedback · Enter submit · Esc return to review ".to_owned(),
            COLOR_ACCENT,
        ),
        None if app.plan_review => (" Plan review · use y / r / e / n ".to_owned(), COLOR_ACCENT),
        None if app.busy && snapshot_accepts_steer(&app.snapshot) => (
            " Message · Enter steer · Esc cancel ".to_owned(),
            COLOR_ACCENT,
        ),
        None if app.busy => (" Message · Esc cancel ".to_owned(), COLOR_ACCENT),
        None => (" Message · Enter send ".to_owned(), COLOR_MUTED),
    };
    let visible_input_rows = usize::from(input.height.saturating_sub(2)).max(1);
    let input_start = input_layout.visible_start(visible_input_rows);
    frame.render_widget(
        Paragraph::new(input_layout.visible_text(input_start, visible_input_rows))
            .style(Style::default().fg(COLOR_TEXT))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(input_color))
                    .title(Span::styled(
                        input_title,
                        Style::default()
                            .fg(input_color)
                            .add_modifier(Modifier::BOLD),
                    )),
            ),
        input,
    );
    if app.approval.is_none() || !app.input.is_empty() {
        let cursor_column = u16::try_from(input_layout.cursor_column).unwrap_or(u16::MAX);
        let cursor_row =
            u16::try_from(input_layout.cursor_row.saturating_sub(input_start)).unwrap_or(u16::MAX);
        frame.set_cursor_position((
            input.x.saturating_add(1).saturating_add(cursor_column),
            input.y.saturating_add(1).saturating_add(cursor_row),
        ));
    }

    let status_color = if app.status.contains("failed") || app.status.contains("error") {
        COLOR_ERROR
    } else if app.status == "Ready" {
        COLOR_ASSISTANT
    } else if app.busy {
        COLOR_TOOL
    } else {
        COLOR_MUTED
    };
    let scroll = if app.max_scroll == 0 {
        String::new()
    } else {
        format!(" · view {}/{}", app.scroll_offset, app.max_scroll)
    };
    let details = if app.transcript_details_expanded {
        " · Ctrl+O collapse details"
    } else {
        " · Ctrl+O expand details"
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                format!(" {}", app.status),
                Style::default()
                    .fg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    " · pending {} · todos {} · context {} · compact r{} · steer {} · safe {} · permission {}{scroll}{details}",
                    app.snapshot.pending_calls.len(),
                    app.snapshot.todos.iter().map(|phase| phase.tasks.len()).sum::<usize>(),
                    app.snapshot.compaction.last_input_tokens,
                    app.snapshot.compaction.revision,
                    app.snapshot.pending_steers.len(),
                    app.snapshot.safe_tools.len(),
                    app.snapshot.permission_mode
                ),
                Style::default().fg(COLOR_MUTED),
            ),
        ]))
        .alignment(Alignment::Left),
        status,
    );
}

fn default_core_path() -> PathBuf {
    if let Some(path) = env::var_os("MYCODE_CORE_BIN") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../mycode/.lake/build/bin/mycode")
}

fn default_git_plugin_path() -> PathBuf {
    if let Some(path) = env::var_os("MYCODE_GIT_PLUGIN_BIN") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../mycode/.lake/build/bin/mycode_git_plugin")
}

#[cfg(test)]
mod tests {
    use std::{
        env, fs,
        path::Path,
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    use clap::Parser;
    use crossterm::event::{
        KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    };
    use ratatui::{Terminal, backend::TestBackend, text::Line};

    use super::{
        App, Args, BtwSidechainStore, COLOR_ACCENT, COLOR_ASSISTANT_BACKGROUND, COLOR_MUTED,
        COLOR_USER_BACKGROUND, CoreCompactionState, CoreEvent, CoreMessage, CorePlanState,
        CoreSnapshot, CoreTodoItem, CoreTodoPhase, CoreToolCall, InputBuffer, InputLayout,
        KeyAction, LiveToolOutput, LocalTranscriptEntry, MouseAction, PermissionMode,
        PlanReviewAction, ProviderError, SelectionPoint, SelectionRegion, SlashCommand,
        SlashCommandError, TextSelection, ToolOutputStream, ToolSpec, TranscriptLayoutCache,
        UserSubmission, active_model_messages, anthropic_messages, automatically_safe_tool_names,
        btw_sidechain_path, build_live_tool_lines, build_transcript_lines, compaction_cut, draw,
        handle_key, model_tool_catalogs, parse_openai_tool_calls, parse_submission,
        parse_todos_markdown, provider_input_tokens, startup_permission_mode, text_between_columns,
        todos_to_markdown,
    };

    #[test]
    fn tool_summaries_hide_todo_json_and_show_file_paths() {
        let todo = CoreToolCall {
            call_id: "todo".to_owned(),
            name: "todo".to_owned(),
            arguments: serde_json::json!({"op": "done", "task": "secret details"}),
        };
        assert_eq!(todo.transcript_summary(), "todo  update task list");
        assert!(!todo.transcript_summary().contains("secret details"));

        let edit = CoreToolCall {
            call_id: "edit".to_owned(),
            name: "edit".to_owned(),
            arguments: serde_json::json!({"path": "src/main.rs", "oldText": "secret"}),
        };
        assert!(edit.transcript_summary().contains("✎ src/main.rs"));

        let grep = CoreToolCall {
            call_id: "grep".to_owned(),
            name: "grep".to_owned(),
            arguments: serde_json::json!({
                "pattern": "fn main",
                "path": "src",
                "caseSensitive": false,
                "maxResults": 10
            }),
        };
        assert_eq!(
            grep.transcript_summary(),
            "grep  🔎 /fn main/  src · ignore case"
        );
        assert!(!grep.transcript_summary().contains("maxResults"));
    }

    #[test]
    fn read_results_use_black_code_blocks_and_todo_results_are_summarized() {
        let messages = [
            CoreMessage {
                role: "assistant".to_owned(),
                tool_calls: vec![
                    CoreToolCall {
                        call_id: "read".to_owned(),
                        name: "read".to_owned(),
                        arguments: serde_json::json!({"path": "src/main.rs"}),
                    },
                    CoreToolCall {
                        call_id: "todo".to_owned(),
                        name: "todo".to_owned(),
                        arguments: serde_json::json!({"op": "done", "task": "hidden"}),
                    },
                ],
                ..CoreMessage::default()
            },
            CoreMessage {
                role: "tool".to_owned(),
                tool_call_id: Some("read".to_owned()),
                content: "fn main() {}".to_owned(),
                ..CoreMessage::default()
            },
            CoreMessage {
                role: "tool".to_owned(),
                tool_call_id: Some("todo".to_owned()),
                content: "{\"todos\": [\"hidden\"]}".to_owned(),
                ..CoreMessage::default()
            },
        ];
        let lines = build_transcript_lines(&messages, &[], 80, true);
        let read_line = lines
            .iter()
            .find(|line| line.to_string().contains("fn main"))
            .expect("read output must render");
        assert_eq!(read_line.style.bg, Some(super::COLOR_FILE_BACKGROUND));
        let rendered = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("📄 src/main.rs"));
        assert!(rendered.contains("✓ task list updated"));
        assert!(!rendered.contains("hidden"));
    }
    #[test]
    fn grep_results_use_search_blocks_and_hide_raw_arguments() {
        let messages = [
            CoreMessage {
                role: "assistant".to_owned(),
                tool_calls: vec![CoreToolCall {
                    call_id: "grep".to_owned(),
                    name: "grep".to_owned(),
                    arguments: serde_json::json!({
                        "pattern": "fn main",
                        "path": "src",
                        "caseSensitive": false,
                        "maxResults": 10
                    }),
                }],
                ..CoreMessage::default()
            },
            CoreMessage {
                role: "tool".to_owned(),
                tool_call_id: Some("grep".to_owned()),
                content: "src/main.rs:1:fn main() {}".to_owned(),
                ..CoreMessage::default()
            },
        ];
        let lines = build_transcript_lines(&messages, &[], 80, true);
        let rendered = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        let match_line = lines
            .iter()
            .find(|line| line.to_string().contains("src/main.rs:1"))
            .expect("grep result must render");
        assert_eq!(match_line.style.bg, Some(super::COLOR_GREP_BACKGROUND));
        assert!(rendered.contains("🔎 /fn main/  src · ignore case"));
        assert!(rendered.contains("└ Search results"));
        assert!(rendered.contains("  • src/main.rs:1:fn main() {}"));
        assert!(!rendered.contains("maxResults"));
        assert!(!rendered.contains("caseSensitive"));
    }
    #[test]
    fn parses_complete_openai_function_calls() {
        let calls = serde_json::json!([{
            "id": "call_1",
            "function": {"name": "read", "arguments": "{\"path\":\"Main.lean\"}"}
        }]);
        let parsed = parse_openai_tool_calls(calls.as_array().expect("array must exist"))
            .expect("function call must parse");
        assert_eq!(parsed[0].name, "read");
        assert_eq!(parsed[0].arguments["path"], "Main.lean");
    }

    #[test]
    fn parses_provider_input_usage() {
        assert_eq!(
            provider_input_tokens(
                &serde_json::json!({"usage": {"prompt_tokens": 1234}}),
                "prompt_tokens"
            ),
            1234
        );
        assert_eq!(
            provider_input_tokens(
                &serde_json::json!({"usage": {"input_tokens": 5678}}),
                "input_tokens"
            ),
            5678
        );
        assert_eq!(
            provider_input_tokens(&serde_json::json!({}), "input_tokens"),
            0
        );
    }

    #[test]
    fn compacted_model_context_keeps_summary_and_recent_turns() {
        let snapshot = CoreSnapshot {
            messages: vec![
                CoreMessage {
                    role: "user".to_owned(),
                    content: "old request".to_owned(),
                    ..CoreMessage::default()
                },
                CoreMessage {
                    role: "assistant".to_owned(),
                    content: "old answer".to_owned(),
                    ..CoreMessage::default()
                },
                CoreMessage {
                    role: "user".to_owned(),
                    content: "recent request".to_owned(),
                    ..CoreMessage::default()
                },
                CoreMessage {
                    role: "assistant".to_owned(),
                    content: "recent answer".to_owned(),
                    ..CoreMessage::default()
                },
            ],
            compaction: CoreCompactionState {
                revision: 1,
                summary: "old decisions".to_owned(),
                first_kept_message: 2,
                ..CoreCompactionState::default()
            },
            ..CoreSnapshot::default()
        };
        let messages = active_model_messages(&snapshot);
        assert_eq!(messages.len(), 3);
        assert!(messages[0].content.contains("old decisions"));
        assert_eq!(messages[1].content, "recent request");
        assert_eq!(messages[2].content, "recent answer");
    }

    #[test]
    fn compaction_cut_keeps_two_latest_user_turns() {
        let snapshot = CoreSnapshot {
            messages: ["one", "two", "three"]
                .into_iter()
                .flat_map(|content| {
                    [
                        CoreMessage {
                            role: "user".to_owned(),
                            content: content.to_owned(),
                            ..CoreMessage::default()
                        },
                        CoreMessage {
                            role: "assistant".to_owned(),
                            content: "answer".to_owned(),
                            ..CoreMessage::default()
                        },
                    ]
                })
                .collect(),
            ..CoreSnapshot::default()
        };
        assert_eq!(compaction_cut(&snapshot), Some(2));
    }

    #[test]
    fn classifies_provider_context_overflow() {
        let error = ProviderError::Status {
            status: reqwest::StatusCode::BAD_REQUEST,
            body: "maximum context length exceeded".to_owned(),
        };
        assert!(error.is_context_overflow());
    }

    #[test]
    fn marks_read_and_grep_as_automatically_safe() {
        let tools = ["read", "grep", "write"]
            .into_iter()
            .map(|name| ToolSpec {
                name: name.to_owned(),
                description: String::new(),
                input_schema: serde_json::json!({}),
            })
            .collect::<Vec<_>>();

        assert_eq!(
            automatically_safe_tool_names(&tools),
            vec!["read".to_owned(), "grep".to_owned()]
        );
    }

    #[test]
    fn built_in_tools_have_separate_normal_and_plan_catalogues() {
        let plugin_tools = ["read", "write", "bash"]
            .into_iter()
            .map(|name| ToolSpec {
                name: name.to_owned(),
                description: String::new(),
                input_schema: serde_json::json!({}),
            })
            .collect::<Vec<_>>();
        let catalogues = model_tool_catalogs(&plugin_tools);
        fn names(tools: &[ToolSpec]) -> Vec<&str> {
            tools.iter().map(|tool| tool.name.as_str()).collect()
        }

        assert_eq!(
            names(&catalogues.normal),
            vec!["read", "write", "bash", "todo"]
        );
        assert_eq!(
            names(&catalogues.plan),
            vec!["read", "bash", "todo", "plan"]
        );
        assert_eq!(
            names(&catalogues.declared),
            vec!["read", "write", "bash", "todo", "plan"]
        );
    }

    #[test]
    fn startup_permission_defaults_to_auto_and_preserves_restored_sessions() {
        let implicit = Args::try_parse_from([
            "mycode-tui",
            "--provider",
            "openai",
            "--model",
            "smoke",
            "--plugin",
            "plugin",
        ])
        .expect("base arguments must parse");
        assert_eq!(implicit.context_window, 128_000);
        assert_eq!(implicit.auto_compact_threshold_percent, 80);
        assert!(!implicit.no_auto_compact);
        let explicit = Args::try_parse_from([
            "mycode-tui",
            "--provider",
            "openai",
            "--model",
            "smoke",
            "--plugin",
            "plugin",
            "--permission-mode",
            "auto",
        ])
        .expect("explicit permission arguments must parse");
        let snapshot = CoreSnapshot {
            phase: "idle".to_owned(),
            permission_mode: "ask".to_owned(),
            ..CoreSnapshot::default()
        };
        assert_eq!(CoreSnapshot::default().permission_mode, "auto");
        assert_eq!(CoreEvent::new("abort").permission_mode, "auto");

        assert_eq!(
            startup_permission_mode(true, implicit.permission_mode, &snapshot),
            PermissionMode::Ask
        );
        assert_eq!(
            startup_permission_mode(false, implicit.permission_mode, &snapshot),
            PermissionMode::Auto
        );
        assert_eq!(
            startup_permission_mode(true, explicit.permission_mode, &snapshot),
            PermissionMode::Auto
        );
    }

    #[test]
    fn input_buffer_edits_at_unicode_grapheme_boundaries() {
        let mut input = InputBuffer::default();
        input.set_text("a界e\u{301}".to_owned());
        assert_eq!(input.cursor, input.as_str().len());

        input.move_left();
        assert_eq!(input.cursor, "a界".len());
        input.insert('X');
        assert_eq!(input.as_str(), "a界Xe\u{301}");
        assert!(input.backspace());
        assert_eq!(input.as_str(), "a界e\u{301}");
        input.move_left();
        assert!(input.delete());
        assert_eq!(input.as_str(), "ae\u{301}");
    }

    #[test]
    fn input_buffer_moves_and_deletes_by_code_word() {
        let mut input = InputBuffer::default();
        input.set_text("alpha_beta  中文 :: gamma".to_owned());

        input.move_word_left();
        assert_eq!(
            input.cursor,
            input.as_str().rfind("gamma").expect("gamma must exist")
        );
        input.move_word_left();
        assert_eq!(
            input.cursor,
            input.as_str().find("::").expect("separator must exist")
        );
        input.move_word_left();
        assert_eq!(
            input.cursor,
            input.as_str().find("中文").expect("word must exist")
        );
        input.move_start();
        input.move_word_right();
        assert_eq!(input.cursor, "alpha_beta".len());
        input.move_end();
        assert!(input.delete_word_left());
        assert_eq!(input.as_str(), "alpha_beta  中文 :: ");
    }

    #[test]
    fn cursor_keybindings_insert_and_delete_away_from_the_tail() {
        let mut app = App::new(CoreSnapshot::default());
        app.set_input("one two".to_owned());
        app.max_scroll = 20;
        app.scroll_offset = 20;

        let _ = handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::ALT), &mut app);
        assert_eq!(app.input.cursor, "one ".len());
        let _ = handle_key(
            KeyEvent::new(KeyCode::Char('X'), KeyModifiers::NONE),
            &mut app,
        );
        assert_eq!(app.input.as_str(), "one Xtwo");
        let _ = handle_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE), &mut app);
        assert_eq!(app.input.cursor, 0);
        assert_eq!(app.scroll_offset, 20);
        let _ = handle_key(KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE), &mut app);
        assert_eq!(app.input.as_str(), "ne Xtwo");
        let _ = handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE), &mut app);
        let _ = handle_key(
            KeyEvent::new(KeyCode::Char('w'), KeyModifiers::CONTROL),
            &mut app,
        );
        assert_eq!(app.input.as_str(), "ne ");
    }
    #[test]
    fn alt_and_control_navigation_shortcuts_move_the_cursor() {
        let mut app = App::new(CoreSnapshot::default());
        app.set_input("one two".to_owned());

        let _ = handle_key(
            KeyEvent::new(KeyCode::Char('b'), KeyModifiers::ALT),
            &mut app,
        );
        assert_eq!(app.input.cursor, "one ".len());
        let _ = handle_key(
            KeyEvent::new(KeyCode::Char('f'), KeyModifiers::ALT),
            &mut app,
        );
        assert_eq!(app.input.cursor, app.input.as_str().len());
        let _ = handle_key(
            KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL),
            &mut app,
        );
        assert_eq!(app.input.cursor, "one tw".len());
        let _ = handle_key(
            KeyEvent::new(KeyCode::Char('f'), KeyModifiers::CONTROL),
            &mut app,
        );
        assert_eq!(app.input.cursor, app.input.as_str().len());
        let _ = handle_key(
            KeyEvent::new(KeyCode::Left, KeyModifiers::CONTROL),
            &mut app,
        );
        assert_eq!(app.input.cursor, "one ".len());
        let _ = handle_key(
            KeyEvent::new(KeyCode::Right, KeyModifiers::CONTROL),
            &mut app,
        );
        assert_eq!(app.input.cursor, app.input.as_str().len());
        let _ = handle_key(
            KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL),
            &mut app,
        );
        assert_eq!(app.input.cursor, 0);
        let _ = handle_key(
            KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL),
            &mut app,
        );
        assert_eq!(app.input.cursor, app.input.as_str().len());
    }

    #[test]
    fn input_layout_wraps_text_and_places_full_line_cursor_on_next_row() {
        let mut input = InputBuffer::default();
        input.set_text("ab界cd".to_owned());
        let layout = InputLayout::new(&input, 4);
        assert_eq!(layout.lines, vec!["ab界".to_owned(), "cd".to_owned()]);
        assert_eq!((layout.cursor_row, layout.cursor_column), (1, 2));

        input.set_text("ab界".to_owned());
        let layout = InputLayout::new(&input, 4);
        assert_eq!(layout.lines, vec!["ab界".to_owned(), String::new()]);
        assert_eq!((layout.cursor_row, layout.cursor_column), (1, 0));
        assert_eq!(layout.visible_start(1), 1);
    }

    #[test]
    fn draw_places_terminal_cursor_at_input_buffer_cursor() {
        let mut app = App::new(CoreSnapshot::default());
        app.set_input("abcdef".to_owned());
        app.input.move_left();
        app.input.move_left();
        let backend = TestBackend::new(40, 12);
        let mut terminal = Terminal::new(backend).expect("test terminal must initialize");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("input cursor frame must draw");

        let cursor = terminal.backend().cursor_position();
        assert_eq!(cursor.x, 5);
        assert_eq!(cursor.y, 9);
    }

    #[test]
    fn parses_supported_slash_commands_and_literal_escape() {
        assert_eq!(
            parse_submission("/btw why this design?".to_owned()),
            Ok(UserSubmission::Command(SlashCommand::Btw(
                "why this design?".to_owned()
            )))
        );
        assert_eq!(
            parse_submission("/steer use the safer plan".to_owned()),
            Ok(UserSubmission::Command(SlashCommand::Steer(
                "use the safer plan".to_owned()
            )))
        );
        assert_eq!(
            parse_submission("/plan migrate storage safely".to_owned()),
            Ok(UserSubmission::Command(SlashCommand::Plan(
                "migrate storage safely".to_owned()
            )))
        );
        assert_eq!(
            parse_submission("/todo".to_owned()),
            Ok(UserSubmission::Command(SlashCommand::Todo))
        );
        assert_eq!(
            parse_submission("/compact".to_owned()),
            Ok(UserSubmission::Command(SlashCommand::Compact(None)))
        );
        assert_eq!(
            parse_submission("/compact preserve API decisions".to_owned()),
            Ok(UserSubmission::Command(SlashCommand::Compact(Some(
                "preserve API decisions".to_owned()
            ))))
        );
        assert_eq!(
            parse_submission("/exit".to_owned()),
            Ok(UserSubmission::Command(SlashCommand::Exit))
        );
        assert_eq!(
            parse_submission("//btw literal".to_owned()),
            Ok(UserSubmission::Prompt("/btw literal".to_owned()))
        );
        assert_eq!(
            parse_submission("/btw".to_owned()),
            Err(SlashCommandError::InvalidUsage("/btw <question>")),
        );
        assert_eq!(
            parse_submission("/unknown".to_owned()),
            Err(SlashCommandError::Unknown("unknown".to_owned()))
        );
    }
    #[test]
    fn slash_candidates_filter_and_tab_complete() {
        let mut app = App::new(CoreSnapshot::default());
        app.set_input("/".to_owned());
        assert!(app.slash_menu_visible());
        assert_eq!(app.slash_candidate_count(), 6);

        let action = handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE), &mut app);
        assert!(matches!(action, KeyAction::None));
        assert_eq!(app.slash_selection, 1);
        let action = handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE), &mut app);
        assert!(matches!(action, KeyAction::None));
        assert_eq!(app.input.as_str(), "/compact ");
        assert!(!app.slash_menu_visible());

        app.set_input("/b".to_owned());
        app.input_changed();
        assert_eq!(app.slash_candidate_count(), 1);
        let _ = handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE), &mut app);
        assert_eq!(app.input.as_str(), "/btw ");
        assert_eq!(app.slash_candidate_count(), 0);

        app.set_input("//".to_owned());
        app.input_changed();
        assert!(!app.slash_menu_visible());
        app.set_input("/btw question".to_owned());
        app.input_changed();
        assert!(!app.slash_menu_visible());
    }

    #[test]
    fn slash_navigation_precedes_scrolling() {
        let mut app = App::new(CoreSnapshot::default());
        app.set_input("/".to_owned());
        app.max_scroll = 20;
        app.scroll_offset = 20;
        app.follow_tail = true;

        let _ = handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE), &mut app);
        assert_eq!(app.slash_selection, 5);
        assert_eq!(app.scroll_offset, 20);
        assert!(app.follow_tail);

        let _ = handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE), &mut app);
        assert_eq!(app.slash_selection, 0);
        assert_eq!(app.scroll_offset, 20);
    }

    #[test]
    fn busy_escape_cancels_before_dismissing_slash_menu() {
        let mut app = App::new(CoreSnapshot::default());
        app.set_input("/".to_owned());
        app.busy = true;

        assert!(matches!(
            handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE), &mut app),
            KeyAction::Cancel
        ));
        assert!(app.slash_menu_visible());
        assert_eq!(app.input.as_str(), "/");

        app.busy = false;
        assert!(matches!(
            handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE), &mut app),
            KeyAction::None
        ));
        assert!(!app.slash_menu_visible());
        assert_eq!(app.input.as_str(), "/");
    }

    #[test]
    fn todo_markdown_round_trips_human_edits() {
        let phases = vec![CoreTodoPhase {
            name: "Implementation".to_owned(),
            tasks: vec![
                CoreTodoItem {
                    content: "Add protocol".to_owned(),
                    status: "in_progress".to_owned(),
                    blocker: None,
                },
                CoreTodoItem {
                    content: "Run smoke test".to_owned(),
                    status: "blocked".to_owned(),
                    blocker: Some("mock server unavailable".to_owned()),
                },
            ],
        }];
        let document = todos_to_markdown(&phases);
        assert_eq!(
            parse_todos_markdown(&document).expect("todo document must parse"),
            phases
        );
    }

    #[test]
    fn plan_review_accepts_feedback_and_approval_keys() {
        let snapshot = CoreSnapshot {
            phase: "waiting_plan_review".to_owned(),
            plan: CorePlanState {
                enabled: true,
                revision: 2,
                status: "review".to_owned(),
                content: "# Plan".to_owned(),
            },
            ..CoreSnapshot::default()
        };
        let mut app = App::new(snapshot.clone());
        app.plan_review = true;
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE),
                &mut app
            ),
            KeyAction::None
        ));
        assert!(app.plan_feedback);
        app.set_input("Use bounded batches".to_owned());
        let _ = handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::ALT), &mut app);
        let _ = handle_key(
            KeyEvent::new(KeyCode::Char('X'), KeyModifiers::NONE),
            &mut app,
        );
        match handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), &mut app) {
            KeyAction::PlanReview(PlanReviewAction::Refine(feedback)) => {
                assert_eq!(feedback, "Use bounded Xbatches")
            }
            _ => panic!("plan feedback must dispatch a refine action"),
        }

        let mut approval_app = App::new(snapshot);
        approval_app.plan_review = true;
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE),
                &mut approval_app
            ),
            KeyAction::PlanReview(PlanReviewAction::Approve)
        ));
    }

    #[test]
    fn slash_candidate_menu_renders_command_usage_and_help() {
        let mut app = App::new(CoreSnapshot::default());
        app.set_input("/".to_owned());
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).expect("test terminal must initialize");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("slash candidate frame must draw");
        let screen: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();

        for expected in [
            "Slash commands",
            "Tab complete",
            "/btw <question>",
            "Ask outside the main conversation",
            "/exit",
            "Exit MyCode cleanly",
            "/steer <instruction>",
            "Redirect the active main task",
        ] {
            assert!(
                screen.contains(expected),
                "candidate menu must contain {expected:?}"
            );
        }
    }

    #[test]
    fn enter_dispatches_btw_and_exit_locally() {
        let mut app = App::new(CoreSnapshot::default());
        app.set_input("/btw side question".to_owned());
        match handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), &mut app) {
            KeyAction::Btw(question) => assert_eq!(question, "side question"),
            _ => panic!("/btw must dispatch as a local BTW action"),
        }
        assert!(app.input.is_empty());

        app.busy = true;
        app.set_input("/exit".to_owned());
        assert!(matches!(
            handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), &mut app),
            KeyAction::Quit
        ));
    }

    #[test]
    fn enter_steers_active_model_and_waiting_approval() {
        let mut model_app = App::new(CoreSnapshot {
            phase: "waiting_model".to_owned(),
            ..CoreSnapshot::default()
        });
        model_app.busy = true;
        model_app.set_input("Use the safer plan".to_owned());
        match handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut model_app,
        ) {
            KeyAction::Steer(text) => assert_eq!(text, "Use the safer plan"),
            _ => panic!("busy main task must accept steer input"),
        }
        assert!(model_app.input.is_empty());

        let mut approval_app = App::new(CoreSnapshot {
            phase: "waiting_approval".to_owned(),
            ..CoreSnapshot::default()
        });
        approval_app.approval = Some(CoreToolCall {
            call_id: "call-steer-approval".to_owned(),
            name: "write".to_owned(),
            arguments: serde_json::json!({}),
        });
        approval_app.set_input("/steer do not write".to_owned());
        match handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut approval_app,
        ) {
            KeyAction::Steer(text) => assert_eq!(text, "do not write"),
            _ => panic!("approval prompt must accept explicit steer input"),
        }
    }

    #[test]
    fn main_prompt_waits_while_btw_sidechain_is_busy() {
        let mut app = App::new(CoreSnapshot {
            phase: "idle".to_owned(),
            ..CoreSnapshot::default()
        });
        app.busy = true;
        app.set_input("Continue the main task".to_owned());
        assert!(matches!(
            handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), &mut app),
            KeyAction::None
        ));
        assert_eq!(app.input.as_str(), "Continue the main task");
    }

    #[test]
    fn local_btw_entry_invalidates_layout_without_mutating_snapshot() {
        let snapshot = CoreSnapshot {
            messages: vec![CoreMessage {
                role: "user".to_owned(),
                content: "main task".to_owned(),
                ..CoreMessage::default()
            }],
            ..CoreSnapshot::default()
        };
        let mut app = App::new(snapshot.clone());
        app.transcript_layout = Some(TranscriptLayoutCache {
            revision: app.transcript_revision,
            width: 80,
            lines: vec![Line::default()],
        });
        let revision = app.transcript_revision;
        app.push_local_entry(LocalTranscriptEntry {
            question: "side question".to_owned(),
            answer: "side answer".to_owned(),
        });
        assert_eq!(app.snapshot.messages, snapshot.messages);
        assert_eq!(app.snapshot.pending_calls, snapshot.pending_calls);
        assert_eq!(app.snapshot.phase, snapshot.phase);
        assert_eq!(app.transcript_revision, revision.wrapping_add(1));
        assert!(app.transcript_layout.is_none());
        let rendered = app
            .visible_transcript(80, 20)
            .lines
            .iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("> BTW · side question"));
        assert!(rendered.contains("· side answer"));
        assert!(!rendered.contains("◇ BTW sidechain"));
    }

    #[tokio::test]
    async fn btw_sidechain_persists_and_recovers_independently() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock must be after the Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!("mycode-btw-sidechain-{nonce}"));
        fs::create_dir(&root).expect("sidechain fixture directory must be created");
        let path = root.join("session.json.btw.json");
        let mut store = BtwSidechainStore::load(path.clone())
            .await
            .expect("missing sidechain must start empty");
        let first = LocalTranscriptEntry {
            question: "first question".to_owned(),
            answer: "first answer".to_owned(),
        };
        let second = LocalTranscriptEntry {
            question: "follow-up".to_owned(),
            answer: "follow-up answer".to_owned(),
        };
        store
            .append(first.clone())
            .await
            .expect("first sidechain entry must persist");
        store
            .append(second.clone())
            .await
            .expect("follow-up sidechain entry must persist");

        let recovered = BtwSidechainStore::load(path)
            .await
            .expect("persisted sidechain must reload");
        assert_eq!(recovered.entries, vec![first, second]);
        fs::remove_dir_all(root).expect("sidechain fixture must be removed");
    }

    #[test]
    fn session_sidechain_uses_a_separate_sidecar_path() {
        assert_eq!(
            btw_sidechain_path(None, Some(Path::new("sessions/main.json")))
                .expect("session sidechain path must resolve"),
            Path::new("sessions/main.json.btw.json")
        );
    }

    #[test]
    fn encodes_auto_permission_for_the_lean_core() {
        let event = CoreEvent::configure_tools(vec!["read".to_owned()], PermissionMode::Auto);
        let encoded = serde_json::to_value(event).expect("event must serialize");
        assert_eq!(encoded["permissionMode"], "auto");
    }

    #[test]
    fn groups_consecutive_anthropic_tool_results() {
        let messages = vec![
            CoreMessage {
                role: "tool".to_owned(),
                content: "first".to_owned(),
                tool_call_id: Some("call_1".to_owned()),
                ..CoreMessage::default()
            },
            CoreMessage {
                role: "tool".to_owned(),
                content: "second".to_owned(),
                tool_call_id: Some("call_2".to_owned()),
                ..CoreMessage::default()
            },
        ];
        let mapped = anthropic_messages(&messages);
        assert_eq!(mapped.len(), 1);
        assert_eq!(
            mapped[0]["content"]
                .as_array()
                .expect("content array")
                .len(),
            2
        );
    }

    #[test]
    fn merges_anthropic_tool_results_before_steer_text() {
        let mapped = anthropic_messages(&[
            CoreMessage {
                role: "tool".to_owned(),
                content: "first contents".to_owned(),
                tool_call_id: Some("call_1".to_owned()),
                ..CoreMessage::default()
            },
            CoreMessage {
                role: "tool".to_owned(),
                content: "skipped".to_owned(),
                tool_call_id: Some("call_2".to_owned()),
                is_error: true,
                ..CoreMessage::default()
            },
            CoreMessage {
                role: "user".to_owned(),
                content: "Use the new plan".to_owned(),
                ..CoreMessage::default()
            },
        ]);
        assert_eq!(mapped.len(), 1);
        let content = mapped[0]["content"]
            .as_array()
            .expect("Anthropic user content must use blocks");
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[1]["type"], "tool_result");
        assert_eq!(content[2]["type"], "text");
        assert_eq!(content[2]["text"], "Use the new plan");
    }

    #[test]
    fn merges_consecutive_anthropic_user_instructions_as_text_blocks() {
        let mapped = anthropic_messages(&[
            CoreMessage {
                role: "user".to_owned(),
                content: "Original plan".to_owned(),
                ..CoreMessage::default()
            },
            CoreMessage {
                role: "user".to_owned(),
                content: "Steered plan".to_owned(),
                ..CoreMessage::default()
            },
        ]);
        assert_eq!(mapped.len(), 1);
        let content = mapped[0]["content"]
            .as_array()
            .expect("Anthropic user content must use blocks");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["text"], "Original plan");
        assert_eq!(content[1]["text"], "Steered plan");
    }

    #[test]
    fn cached_layout_preserves_multiline_indent_and_tool_styles() {
        let lines = build_transcript_lines(
            &[
                CoreMessage {
                    role: "tool".to_owned(),
                    content: "first\nsecond".to_owned(),
                    ..CoreMessage::default()
                },
                CoreMessage {
                    role: "assistant".to_owned(),
                    tool_calls: vec![CoreToolCall {
                        call_id: "call_1".to_owned(),
                        name: "bash".to_owned(),
                        arguments: serde_json::json!({"command": "git status --short"}),
                    }],
                    ..CoreMessage::default()
                },
            ],
            &[],
            80,
            true,
        );
        let first_line = lines
            .iter()
            .find(|line| line.to_string().contains("first"))
            .expect("first content line must be present")
            .to_string();
        let second_line = lines
            .iter()
            .find(|line| line.to_string().contains("second"))
            .expect("second content line must be present")
            .to_string();
        assert!(first_line.starts_with("│   first"));
        assert!(first_line.ends_with('│'));
        assert!(second_line.starts_with("│   second"));
        assert!(second_line.ends_with('│'));
        let tool_line = lines
            .iter()
            .find(|line| line.to_string().contains("git status --short"))
            .expect("tool line must be present");
        assert_eq!(tool_line.spans[2].style.fg, Some(COLOR_MUTED));
        assert_eq!(tool_line.spans[3].style.fg, Some(COLOR_ACCENT));
        assert_eq!(tool_line.spans[4].style.fg, Some(COLOR_MUTED));
    }

    #[test]
    fn human_messages_are_compact_and_only_code_and_tools_have_borders() {
        let lines = build_transcript_lines(
            &[
                CoreMessage {
                    role: "user".to_owned(),
                    content: "request".to_owned(),
                    ..CoreMessage::default()
                },
                CoreMessage {
                    role: "assistant".to_owned(),
                    content: "answer\n```rust\nfn main() {}\n```\ndone".to_owned(),
                    tool_calls: vec![CoreToolCall {
                        call_id: "call_1".to_owned(),
                        name: "bash".to_owned(),
                        arguments: serde_json::json!({"command": "git status --short"}),
                    }],
                    ..CoreMessage::default()
                },
                CoreMessage {
                    role: "tool".to_owned(),
                    content: "result".to_owned(),
                    tool_call_id: Some("call_1".to_owned()),
                    ..CoreMessage::default()
                },
            ],
            &[],
            48,
            true,
        );
        let rendered = lines
            .iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let user = lines
            .iter()
            .find(|line| line.to_string().starts_with("> request"))
            .expect("compact user message must be present");
        let assistant = lines
            .iter()
            .find(|line| line.to_string().starts_with("· answer"))
            .expect("compact assistant message must be present");

        assert_eq!(user.style.bg, Some(COLOR_USER_BACKGROUND));
        assert_eq!(assistant.style.bg, Some(COLOR_ASSISTANT_BACKGROUND));
        assert!(!rendered.iter().any(|line| line.contains("You")));
        assert!(!rendered.iter().any(|line| line.contains("Assistant")));
        assert!(
            rendered
                .iter()
                .any(|line| line.starts_with("│ fn main() {}"))
        );
        assert!(
            rendered
                .iter()
                .any(|line| line.contains("git status --short"))
        );
        assert!(rendered.iter().any(|line| line.contains("└ Tool result")));
        assert_eq!(
            rendered.iter().filter(|line| line.starts_with('╭')).count(),
            3
        );
    }

    #[test]
    fn transcript_blocks_have_one_blank_line_between_them() {
        let lines = build_transcript_lines(
            &[
                CoreMessage {
                    role: "user".to_owned(),
                    content: "request".to_owned(),
                    ..CoreMessage::default()
                },
                CoreMessage {
                    role: "assistant".to_owned(),
                    content: "answer".to_owned(),
                    ..CoreMessage::default()
                },
            ],
            &[],
            32,
            true,
        );

        assert_eq!(lines.len(), 4);
        assert!(lines[0].to_string().starts_with("> request"));
        assert!(lines[1].spans.is_empty());
        assert!(lines[2].to_string().starts_with("· answer"));
        assert!(lines[3].spans.is_empty());
    }

    #[test]
    fn folds_long_commands_until_details_are_expanded() {
        let messages = [CoreMessage {
            role: "assistant".to_owned(),
            tool_calls: vec![CoreToolCall {
                call_id: "call_long".to_owned(),
                name: "bash".to_owned(),
                arguments: serde_json::json!({"command": "x".repeat(160)}),
            }],
            ..CoreMessage::default()
        }];
        let collapsed = build_transcript_lines(&messages, &[], 32, false);
        let expanded = build_transcript_lines(&messages, &[], 32, true);
        assert!(
            collapsed
                .iter()
                .any(|line| line.to_string().contains("command lines hidden"))
        );
        assert!(
            !expanded
                .iter()
                .any(|line| line.to_string().contains("command lines hidden"))
        );
        assert!(expanded.len() > collapsed.len());
    }

    #[test]
    fn folds_long_file_results_until_details_are_expanded() {
        let messages = [
            CoreMessage {
                role: "assistant".to_owned(),
                tool_calls: vec![CoreToolCall {
                    call_id: "call_read".to_owned(),
                    name: "read".to_owned(),
                    arguments: serde_json::json!({"path": "large.txt"}),
                }],
                ..CoreMessage::default()
            },
            CoreMessage {
                role: "tool".to_owned(),
                tool_call_id: Some("call_read".to_owned()),
                content: (0..16)
                    .map(|index| format!("line {index}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                ..CoreMessage::default()
            },
        ];
        let collapsed = build_transcript_lines(&messages, &[], 80, false);
        let expanded = build_transcript_lines(&messages, &[], 80, true);
        assert!(
            collapsed
                .iter()
                .any(|line| line.to_string().contains("└ File result"))
        );
        assert!(
            collapsed
                .iter()
                .any(|line| line.to_string().contains("file output lines hidden"))
        );
        assert!(
            !expanded
                .iter()
                .any(|line| line.to_string().contains("file output lines hidden"))
        );
        assert!(expanded.len() > collapsed.len());
    }

    #[test]
    fn folded_command_results_show_latest_tail() {
        let messages = [
            CoreMessage {
                role: "assistant".to_owned(),
                tool_calls: vec![CoreToolCall {
                    call_id: "call_bash".to_owned(),
                    name: "bash".to_owned(),
                    arguments: serde_json::json!({"command": "long command"}),
                }],
                ..CoreMessage::default()
            },
            CoreMessage {
                role: "tool".to_owned(),
                tool_call_id: Some("call_bash".to_owned()),
                content: format!(
                    "{}\n[stderr]\n{}",
                    (0..8)
                        .map(|index| format!("stdout line {index}"))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    (0..8)
                        .map(|index| format!("stderr line {index}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                ),
                ..CoreMessage::default()
            },
        ];
        let collapsed = build_transcript_lines(&messages, &[], 80, false)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(collapsed.contains("stdout tail"));
        assert!(collapsed.contains("stderr tail"));
        assert!(collapsed.contains("stdout line 7"));
        assert!(collapsed.contains("stderr line 7"));
        assert!(!collapsed.contains("stdout line 0"));
        assert!(!collapsed.contains("stderr line 0"));
    }

    #[test]
    fn live_command_collapsed_view_shows_latest_stdout_and_stderr() {
        let mut live = LiveToolOutput::new("call_live".to_owned(), "bash".to_owned());
        live.append(
            ToolOutputStream::Stdout,
            b"stdout one\nstdout two\nstdout three\nstdout four\nstdout five\n",
        );
        live.append(
            ToolOutputStream::Stderr,
            b"stderr one\nstderr two\nstderr three\nstderr four\nstderr five\n",
        );
        let collapsed = build_live_tool_lines(&live, 80, false)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(collapsed.contains("stdout tail"));
        assert!(collapsed.contains("stderr tail"));
        assert!(!collapsed.contains("stdout one"));
        assert!(collapsed.contains("stdout five"));
        assert!(!collapsed.contains("stderr one"));
        assert!(collapsed.contains("stderr five"));

        let expanded = build_live_tool_lines(&live, 80, true)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(expanded.contains("stdout one"));
        assert!(expanded.contains("stderr one"));
    }

    #[test]
    fn control_o_toggles_transcript_details_and_invalidates_layout() {
        let mut app = App::new(CoreSnapshot::default());
        app.transcript_layout = Some(TranscriptLayoutCache {
            revision: app.transcript_revision,
            width: 80,
            lines: vec![Line::default()],
        });
        let _ = handle_key(
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            &mut app,
        );
        assert!(app.transcript_details_expanded);
        assert!(app.transcript_layout.is_none());
    }

    #[test]
    fn displays_original_command_for_lowered_git_calls() {
        let call = CoreToolCall {
            call_id: "call_git".to_owned(),
            name: "git_write".to_owned(),
            arguments: serde_json::json!({
                "operation": "add",
                "arguments": ["Main.lean"],
                "command": "git add -- Main.lean"
            }),
        };
        assert_eq!(call.display_label(), "git add -- Main.lean");
    }

    #[test]
    fn refreshed_layout_renders_semantic_message_hierarchy() {
        let snapshot = CoreSnapshot {
            phase: "idle".to_owned(),
            messages: vec![
                CoreMessage {
                    role: "user".to_owned(),
                    content: "Inspect the repository".to_owned(),
                    ..CoreMessage::default()
                },
                CoreMessage {
                    role: "assistant".to_owned(),
                    tool_calls: vec![CoreToolCall {
                        call_id: "call_1".to_owned(),
                        name: "bash".to_owned(),
                        arguments: serde_json::json!({"command": "git status --short"}),
                    }],
                    ..CoreMessage::default()
                },
                CoreMessage {
                    role: "tool".to_owned(),
                    content: "M Cargo.toml".to_owned(),
                    tool_call_id: Some("call_1".to_owned()),
                    ..CoreMessage::default()
                },
            ],
            safe_tools: vec!["read".to_owned(), "git_read".to_owned()],
            ..CoreSnapshot::default()
        };
        let mut app = App::new(snapshot);
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).expect("test terminal must initialize");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("refreshed frame must draw");
        let screen: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        for expected in [
            "mycode",
            "> Inspect the repository",
            "└─ bash  git status --short",
            "└ Tool result",
            "Message · Enter send",
            "Ready · pending 0 · todos 0 · context 0 · compact r0 · steer 0 · safe 2",
        ] {
            assert!(
                screen.contains(expected),
                "screen must contain {expected:?}"
            );
        }
        assert!(!screen.contains("You"));
        assert!(!screen.contains("Assistant"));
    }
    #[test]
    fn mouse_selection_overlay_highlights_transcript_cells() {
        let snapshot = CoreSnapshot {
            messages: vec![CoreMessage {
                role: "assistant".to_owned(),
                content: "selectable transcript".to_owned(),
                ..CoreMessage::default()
            }],
            ..CoreSnapshot::default()
        };
        let mut app = App::new(snapshot);
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).expect("test terminal must initialize");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("initial transcript frame must draw");
        let region = app.selection_region.expect("transcript must be selectable");
        app.text_selection = Some(TextSelection {
            anchor: SelectionPoint {
                row: region.top,
                column: region.left,
            },
            focus: SelectionPoint {
                row: region.top,
                column: region.left + 2,
            },
            dragging: false,
        });

        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("selected transcript frame must draw");
        for column in region.left..=region.left + 2 {
            let cell = terminal
                .backend()
                .buffer()
                .cell((column, region.top))
                .expect("selected cell must exist");
            assert_eq!(cell.bg, COLOR_ACCENT);
        }
    }

    #[test]
    fn mouse_drag_copies_selected_transcript_text() {
        let mut app = App::new(CoreSnapshot::default());
        app.selection_region = Some(SelectionRegion {
            left: 2,
            top: 4,
            right: 14,
            bottom: 7,
        });
        app.selection_lines = vec![
            "│ alpha beta  │".to_owned(),
            "│ gamma delta │".to_owned(),
            "│ third       │".to_owned(),
        ];

        let down = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 4,
            row: 4,
            modifiers: KeyModifiers::NONE,
        };
        assert_eq!(app.handle_mouse(down), MouseAction::Redraw);
        let drag = MouseEvent {
            kind: MouseEventKind::Drag(MouseButton::Left),
            column: 8,
            row: 5,
            modifiers: KeyModifiers::NONE,
        };
        assert_eq!(app.handle_mouse(drag), MouseAction::Redraw);
        let up = MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            column: 8,
            row: 5,
            modifiers: KeyModifiers::NONE,
        };
        assert_eq!(
            app.handle_mouse(up),
            MouseAction::Copy("alpha beta\ngamma".to_owned())
        );
        assert!(!app.text_selection.expect("selection must remain").dragging);
    }
    #[test]
    fn click_inside_transcript_does_not_copy_text() {
        let mut app = App::new(CoreSnapshot::default());
        app.selection_region = Some(SelectionRegion {
            left: 2,
            top: 4,
            right: 14,
            bottom: 7,
        });
        app.selection_lines = vec!["│ alpha beta │".to_owned()];
        let down = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 4,
            row: 4,
            modifiers: KeyModifiers::NONE,
        };
        let up = MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            ..down
        };

        assert_eq!(app.handle_mouse(down), MouseAction::Redraw);
        assert_eq!(app.handle_mouse(up), MouseAction::Redraw);
        assert!(app.text_selection.is_none());
    }
    #[test]
    fn double_click_selects_word_control_double_click_selects_line_and_triple_click_selects_paragraph()
     {
        let mut app = App::new(CoreSnapshot::default());
        app.selection_region = Some(SelectionRegion {
            left: 2,
            top: 4,
            right: 24,
            bottom: 7,
        });
        app.selection_lines = vec!["│ alpha beta gamma  │".to_owned()];
        let point = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 10,
            row: 4,
            modifiers: KeyModifiers::NONE,
        };
        let release = MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            ..point
        };
        let start = Instant::now();

        assert_eq!(app.handle_mouse_at(point, start), MouseAction::Redraw);
        assert_eq!(
            app.handle_mouse_at(release, start + Duration::from_millis(10)),
            MouseAction::Redraw
        );
        assert_eq!(
            app.handle_mouse_at(point, start + Duration::from_millis(100)),
            MouseAction::Redraw
        );
        assert_eq!(
            app.handle_mouse_at(release, start + Duration::from_millis(110)),
            MouseAction::Copy("beta".to_owned())
        );

        let mut line_app = App::new(CoreSnapshot::default());
        line_app.selection_region = app.selection_region;
        line_app.selection_lines = app.selection_lines.clone();
        assert_eq!(line_app.handle_mouse_at(point, start), MouseAction::Redraw);
        assert_eq!(
            line_app.handle_mouse_at(release, start + Duration::from_millis(10)),
            MouseAction::Redraw
        );
        let control_point = MouseEvent {
            modifiers: KeyModifiers::CONTROL,
            ..point
        };
        let control_release = MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            ..control_point
        };
        assert_eq!(
            line_app.handle_mouse_at(control_point, start + Duration::from_millis(100)),
            MouseAction::Redraw
        );
        assert_eq!(
            line_app.handle_mouse_at(control_release, start + Duration::from_millis(110)),
            MouseAction::Copy("alpha beta gamma".to_owned())
        );
        let mut paragraph_app = App::new(CoreSnapshot::default());
        paragraph_app.selection_region = Some(SelectionRegion {
            left: 2,
            top: 4,
            right: 24,
            bottom: 10,
        });
        paragraph_app.selection_lines = vec![
            "╭────────────────────╮".to_owned(),
            "│ alpha beta         │".to_owned(),
            "│ gamma delta        │".to_owned(),
            "╰────────────────────╯".to_owned(),
            String::new(),
            "│ other paragraph    │".to_owned(),
        ];
        let paragraph_point = MouseEvent {
            column: 4,
            row: 5,
            ..point
        };
        let paragraph_release = MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            ..paragraph_point
        };
        assert_eq!(
            paragraph_app.handle_mouse_at(paragraph_point, start),
            MouseAction::Redraw
        );
        assert_eq!(
            paragraph_app.handle_mouse_at(paragraph_release, start + Duration::from_millis(10)),
            MouseAction::Redraw
        );
        assert_eq!(
            paragraph_app.handle_mouse_at(paragraph_point, start + Duration::from_millis(100)),
            MouseAction::Redraw
        );
        assert_eq!(
            paragraph_app.handle_mouse_at(paragraph_release, start + Duration::from_millis(110)),
            MouseAction::Copy("alpha".to_owned())
        );
        assert_eq!(
            paragraph_app.handle_mouse_at(paragraph_point, start + Duration::from_millis(200)),
            MouseAction::Redraw
        );
        assert_eq!(
            paragraph_app.handle_mouse_at(paragraph_release, start + Duration::from_millis(210)),
            MouseAction::Copy("alpha beta\ngamma delta".to_owned())
        );
    }

    #[test]
    fn selection_extracts_wide_and_combining_characters() {
        assert_eq!(text_between_columns("a界b", 1, 3), "界");
        assert_eq!(text_between_columns("a界b", 2, 3), "界");
        assert_eq!(text_between_columns("e\u{301}x", 0, 1), "e\u{301}");
    }

    #[test]
    fn mouse_wheel_only_scrolls_over_transcript() {
        let mut app = App::new(CoreSnapshot::default());
        app.max_scroll = 20;
        app.scroll_offset = 20;
        app.selection_region = Some(SelectionRegion {
            left: 1,
            top: 2,
            right: 20,
            bottom: 12,
        });

        let inside = MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 3,
            row: 5,
            modifiers: KeyModifiers::NONE,
        };
        let MouseAction::Scroll(delta) = app.handle_mouse(inside) else {
            panic!("wheel over transcript must request scrolling");
        };
        app.scroll_delta(delta);
        assert_eq!(app.scroll_offset, 19);

        let outside = MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 3,
            row: 15,
            modifiers: KeyModifiers::NONE,
        };
        assert_eq!(app.handle_mouse(outside), MouseAction::None);
        assert_eq!(app.scroll_offset, 19);
    }

    #[test]
    fn plan_review_renders_plan_and_todo_state() {
        let snapshot = CoreSnapshot {
            phase: "waiting_plan_review".to_owned(),
            plan: CorePlanState {
                enabled: true,
                revision: 3,
                status: "review".to_owned(),
                content: "# Storage plan\n\nUse bounded migration batches.".to_owned(),
            },
            todos: vec![CoreTodoPhase {
                name: "Migration".to_owned(),
                tasks: vec![CoreTodoItem {
                    content: "Verify rollback".to_owned(),
                    status: "pending".to_owned(),
                    blocker: None,
                }],
            }],
            ..CoreSnapshot::default()
        };
        let mut app = App::new(snapshot);
        app.plan_review = true;
        let backend = TestBackend::new(110, 30);
        let mut terminal = Terminal::new(backend).expect("test terminal must initialize");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("plan review frame must draw");
        let screen: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        for expected in [
            "Plan review · y approve · r refine · e edit · n cancel",
            "Storage plan",
            "Use bounded migration batches.",
            "Migration",
            "Verify rollback",
        ] {
            assert!(
                screen.contains(expected),
                "screen must contain {expected:?}"
            );
        }
    }

    #[test]
    fn scroll_navigation_leaves_and_restores_tail_following() {
        let mut app = App::new(CoreSnapshot::default());
        app.max_scroll = 20;
        app.page_rows = 6;
        app.scroll_offset = 20;

        app.scroll_up(1);
        assert_eq!(app.scroll_offset, 19);
        assert!(!app.follow_tail);

        app.page_up();
        assert_eq!(app.scroll_offset, 14);

        app.page_down();
        app.scroll_down(1);
        assert_eq!(app.scroll_offset, 20);
        assert!(app.follow_tail);

        app.scroll_home();
        assert_eq!(app.scroll_offset, 0);
        assert!(!app.follow_tail);
        app.scroll_end();
        assert_eq!(app.scroll_offset, 20);
        assert!(app.follow_tail);
    }

    #[test]
    fn padded_transcript_counts_the_rendered_wrap_width() {
        let snapshot = CoreSnapshot {
            messages: vec![
                CoreMessage {
                    role: "assistant".to_owned(),
                    content: "12345678901234567".to_owned(),
                    ..CoreMessage::default()
                },
                CoreMessage {
                    role: "assistant".to_owned(),
                    content: "tail".to_owned(),
                    ..CoreMessage::default()
                },
            ],
            ..CoreSnapshot::default()
        };
        let mut app = App::new(snapshot);
        let backend = TestBackend::new(20, 10);
        let mut terminal = Terminal::new(backend).expect("test terminal must initialize");

        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("padded transcript must draw");

        let rendered_lines = 5_usize;
        assert_eq!(app.max_scroll, rendered_lines.saturating_sub(app.page_rows));
        assert_eq!(app.scroll_offset, app.max_scroll);
        assert!(app.follow_tail);
    }

    #[test]
    fn drawing_follows_tail_until_user_scrolls_up() {
        let snapshot = CoreSnapshot {
            messages: (0..24)
                .map(|index| CoreMessage {
                    role: if index % 2 == 0 { "user" } else { "assistant" }.to_owned(),
                    content: format!("message {index}: {}", "wrapped content ".repeat(5)),
                    ..CoreMessage::default()
                })
                .collect(),
            ..CoreSnapshot::default()
        };
        let mut app = App::new(snapshot);
        let backend = TestBackend::new(40, 12);
        let mut terminal = Terminal::new(backend).expect("test terminal must initialize");

        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("initial frame must draw");
        let initial_max = app.max_scroll;
        assert!(initial_max > 0);
        assert_eq!(app.scroll_offset, initial_max);
        let initial_cache = app
            .transcript_layout
            .as_ref()
            .expect("initial draw must build the transcript cache");
        let initial_revision = initial_cache.revision;
        let initial_lines = initial_cache.lines.as_ptr();

        app.scroll_up(1);
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("scroll-only frame must draw");
        let scrolled_cache = app
            .transcript_layout
            .as_ref()
            .expect("scrolling must retain the transcript cache");
        assert_eq!(scrolled_cache.revision, initial_revision);
        assert_eq!(scrolled_cache.lines.as_ptr(), initial_lines);

        app.scroll_up(3);
        let pinned_offset = app.scroll_offset;
        let mut updated = app.snapshot.clone();
        updated.messages.push(CoreMessage {
            role: "assistant".to_owned(),
            content: "new tail content".repeat(8),
            ..CoreMessage::default()
        });
        app.set_snapshot(updated);
        assert!(app.transcript_layout.is_none());
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("scrolled frame must draw");
        assert!(app.max_scroll > initial_max);
        assert_eq!(app.scroll_offset, pinned_offset);
        assert_ne!(
            app.transcript_layout
                .as_ref()
                .expect("message append must rebuild the transcript cache")
                .revision,
            initial_revision
        );

        app.scroll_end();
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("tail frame must draw");
        assert_eq!(app.scroll_offset, app.max_scroll);
        assert!(app.follow_tail);
    }

    #[test]
    fn tail_following_supports_more_than_u16_rows() {
        let mut app = App::new(CoreSnapshot::default());
        app.transcript_layout = Some(TranscriptLayoutCache {
            revision: app.transcript_revision,
            width: 80,
            lines: vec![Line::default(); 70_000],
        });
        let visible = app.visible_transcript(80, 20);
        assert_eq!(visible.height(), 20);
        assert_eq!(app.max_scroll, 69_980);
        assert_eq!(app.scroll_offset, 69_980);
    }
}
