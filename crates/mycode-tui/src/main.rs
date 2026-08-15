#![deny(warnings)]
#![deny(clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    env,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use clap::{Parser, ValueEnum};
use crossterm::cursor::Show;
use crossterm::{
    event::{
        DisableMouseCapture, EnableMouseCapture, Event as CrosstermEvent, EventStream, KeyCode,
        KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures_util::StreamExt;
use mycode_plugin_protocol::{
    CallToolParams, DEFAULT_MAX_FRAME_BYTES, InitializeParams, InitializeResult, Limits,
    RequestEnvelope as PluginRequest, RequestOperation as PluginOperation, ToolListResult,
    ToolResult, ToolSpec, read_response, write_request,
};
#[cfg(unix)]
use nix::{
    sys::signal::{Signal, killpg},
    unistd::Pid,
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Padding, Paragraph},
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{Mutex, mpsc},
    task::JoinHandle,
    time::{MissedTickBehavior, interval, timeout},
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

const CORE_PROTOCOL_VERSION: u64 = 1;
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a careful coding agent. Use the declared tools when needed. Explain changes briefly.";
const MAX_PROVIDER_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const FRAME_INTERVAL: Duration = Duration::from_millis(16);
const COLOR_ACCENT: Color = Color::Rgb(139, 92, 246);
const COLOR_USER: Color = Color::Rgb(56, 189, 248);
const COLOR_ASSISTANT: Color = Color::Rgb(74, 222, 128);
const COLOR_TOOL: Color = Color::Rgb(250, 204, 21);
const COLOR_TEXT: Color = Color::Rgb(228, 228, 231);
const COLOR_MUTED: Color = Color::Rgb(113, 113, 122);
const COLOR_ERROR: Color = Color::Rgb(248, 113, 113);
const COLLAPSED_COMMAND_LINES: usize = 2;
const COLLAPSED_FILE_OUTPUT_LINES: usize = 8;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Provider {
    Openai,
    Anthropic,
    Linewise,
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
    base_url: Option<String>,
    #[arg(long, default_value = DEFAULT_SYSTEM_PROMPT)]
    system_prompt: String,
    #[arg(long)]
    prompt: Option<String>,
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
        match self.arguments.get("command").and_then(Value::as_str) {
            Some(command) => format!("{}  {command}", self.name),
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
}

impl CoreEvent {
    fn configure_tools(safe_tools: Vec<String>) -> Self {
        Self {
            kind: "configure_tools".to_owned(),
            text: None,
            tool_calls: Vec::new(),
            call_id: None,
            approved: None,
            content: None,
            is_error: None,
            safe_tools,
        }
    }

    fn submit(text: String) -> Self {
        Self {
            kind: "submit".to_owned(),
            text: Some(text),
            tool_calls: Vec::new(),
            call_id: None,
            approved: None,
            content: None,
            is_error: None,
            safe_tools: Vec::new(),
        }
    }

    fn model_completed(content: String, tool_calls: Vec<CoreToolCall>) -> Self {
        Self {
            kind: "model_completed".to_owned(),
            text: None,
            tool_calls,
            call_id: None,
            approved: None,
            content: Some(content),
            is_error: None,
            safe_tools: Vec::new(),
        }
    }

    fn approval(call_id: String, approved: bool) -> Self {
        Self {
            kind: "approval_result".to_owned(),
            text: None,
            tool_calls: Vec::new(),
            call_id: Some(call_id),
            approved: Some(approved),
            content: None,
            is_error: None,
            safe_tools: Vec::new(),
        }
    }

    fn tool_completed(call_id: String, content: String, is_error: bool) -> Self {
        Self {
            kind: "tool_completed".to_owned(),
            text: None,
            tool_calls: Vec::new(),
            call_id: Some(call_id),
            approved: None,
            content: Some(content),
            is_error: Some(is_error),
            safe_tools: Vec::new(),
        }
    }

    fn abort() -> Self {
        Self {
            kind: "abort".to_owned(),
            text: None,
            tool_calls: Vec::new(),
            call_id: None,
            approved: None,
            content: None,
            is_error: None,
            safe_tools: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CoreEffect {
    kind: String,
    #[serde(default)]
    call: Option<CoreToolCall>,
}

#[derive(Clone, Debug, Default, Deserialize)]
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

    async fn call(&mut self, call: &CoreToolCall) -> Result<ToolResult, PluginClientError> {
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
        let response = match timeout(
            Duration::from_secs(300),
            read_response(&mut self.output, DEFAULT_MAX_FRAME_BYTES),
        )
        .await
        {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                self.retire().await;
                return Err(error.into());
            }
            Err(_) => {
                self.retire().await;
                return Err(PluginClientError::DeadlineExceeded);
            }
        };
        if response.id != request_id {
            self.retire().await;
            return Err(PluginClientError::CorrelationMismatch);
        }
        response.into_result().map_err(PluginClientError::from)
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

    async fn call(&mut self, call: &CoreToolCall) -> Result<ToolResult, PluginClientError> {
        let index = self
            .routes
            .get(&call.name)
            .copied()
            .ok_or_else(|| PluginClientError::UnknownTool(call.name.clone()))?;
        self.ensure_plugin(index).await?;
        self.plugins[index].client.call(call).await
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
    #[error("provider generated a tool name not declared by a plugin")]
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

    async fn complete(
        &self,
        snapshot: &CoreSnapshot,
        tools: &[ToolSpec],
    ) -> Result<ModelCompletion, ProviderError> {
        let completion = match self.provider {
            Provider::Openai | Provider::Linewise => self.complete_openai(snapshot, tools).await?,
            Provider::Anthropic => self.complete_anthropic(snapshot, tools).await?,
        };
        if completion
            .tool_calls
            .iter()
            .any(|call| !tools.iter().any(|tool| tool.name == call.name))
        {
            return Err(ProviderError::UnknownTool);
        }
        Ok(completion)
    }

    async fn complete_openai(
        &self,
        snapshot: &CoreSnapshot,
        tools: &[ToolSpec],
    ) -> Result<ModelCompletion, ProviderError> {
        let mut messages = vec![json!({"role": "system", "content": self.system_prompt})];
        for message in &snapshot.messages {
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
        let response = self
            .http
            .post(format!(
                "{}/chat/completions",
                self.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "stream": false
            }))
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
        Ok(ModelCompletion {
            content,
            tool_calls,
        })
    }

    async fn complete_anthropic(
        &self,
        snapshot: &CoreSnapshot,
        tools: &[ToolSpec],
    ) -> Result<ModelCompletion, ProviderError> {
        let messages = anthropic_messages(&snapshot.messages);
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
        let response = self
            .http
            .post(format!("{}/messages", self.base_url.trim_end_matches('/')))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&json!({
                "model": self.model,
                "max_tokens": 4096,
                "system": self.system_prompt,
                "messages": messages,
                "tools": tools
            }))
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
        Ok(ModelCompletion {
            content: text,
            tool_calls,
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
            "user" => json!({"role": "user", "content": message.content}),
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

fn push_folded_lines(
    lines: &mut Vec<Line<'static>>,
    mut block: Vec<Line<'static>>,
    expanded: bool,
    visible_lines: usize,
    kind: &str,
) {
    if !expanded && block.len() > visible_lines {
        let hidden = block.len() - visible_lines;
        lines.extend(block.drain(..visible_lines));
        lines.push(Line::from(Span::styled(
            format!("  … {hidden} {kind} lines hidden · Ctrl+E expand"),
            Style::default().fg(COLOR_MUTED).add_modifier(Modifier::DIM),
        )));
    } else {
        lines.append(&mut block);
    }
}

fn push_tool_call(lines: &mut Vec<Line<'static>>, call: &CoreToolCall, width: u16, expanded: bool) {
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
                "     … {} command lines hidden · Ctrl+E expand  #{}",
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

fn build_transcript_lines(
    messages: &[CoreMessage],
    width: u16,
    details_expanded: bool,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut tool_calls: BTreeMap<&str, &CoreToolCall> = BTreeMap::new();
    for message in messages {
        let source_call = message
            .tool_call_id
            .as_deref()
            .and_then(|call_id| tool_calls.get(call_id));
        let is_file_result =
            message.role == "tool" && source_call.is_some_and(|call| call.name == "read");
        let (label, color, content_color) = match message.role.as_str() {
            "user" => ("› You", COLOR_USER, COLOR_TEXT),
            "assistant" => ("• Assistant", COLOR_ASSISTANT, COLOR_TEXT),
            "tool" if message.is_error => ("! Tool error", COLOR_ERROR, COLOR_ERROR),
            "tool" if is_file_result => ("└ File result", COLOR_TOOL, Color::Rgb(161, 161, 170)),
            "tool" => ("└ Tool result", COLOR_TOOL, Color::Rgb(161, 161, 170)),
            _ => ("· Message", COLOR_MUTED, COLOR_TEXT),
        };
        push_wrapped_line(
            &mut lines,
            label,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
            width,
        );
        if !message.content.is_empty() {
            let content_lines =
                indented_wrapped_lines(&message.content, Style::default().fg(content_color), width);
            if message.role == "tool" {
                push_folded_lines(
                    &mut lines,
                    content_lines,
                    details_expanded,
                    COLLAPSED_FILE_OUTPUT_LINES,
                    if is_file_result {
                        "file output"
                    } else {
                        "tool output"
                    },
                );
            } else {
                lines.extend(content_lines);
            }
        }
        for call in &message.tool_calls {
            push_tool_call(&mut lines, call, width, details_expanded);
            tool_calls.insert(call.call_id.as_str(), call);
        }
        lines.push(Line::default());
    }
    lines
}

struct App {
    input: String,
    status: String,
    snapshot: CoreSnapshot,
    approval: Option<CoreToolCall>,
    busy: bool,
    scroll_offset: usize,
    max_scroll: usize,
    page_rows: usize,
    follow_tail: bool,
    transcript_top: u16,
    transcript_bottom: u16,
    transcript_revision: u64,
    transcript_details_expanded: bool,
    transcript_layout: Option<TranscriptLayoutCache>,
}

impl App {
    fn new(snapshot: CoreSnapshot) -> Self {
        Self {
            input: String::new(),
            status: "Ready".to_owned(),
            snapshot,
            approval: None,
            busy: false,
            scroll_offset: 0,
            max_scroll: 0,
            page_rows: 1,
            follow_tail: true,
            transcript_top: 0,
            transcript_bottom: 0,
            transcript_revision: 0,
            transcript_details_expanded: false,
            transcript_layout: None,
        }
    }

    fn set_snapshot(&mut self, snapshot: CoreSnapshot) {
        let transcript_changed = self.snapshot.messages.len() != snapshot.messages.len()
            || self.snapshot.messages.last() != snapshot.messages.last();
        self.snapshot = snapshot;
        if transcript_changed {
            self.transcript_revision = self.transcript_revision.wrapping_add(1);
            self.transcript_layout = None;
        }
    }

    fn toggle_transcript_details(&mut self) {
        self.transcript_details_expanded = !self.transcript_details_expanded;
        self.transcript_revision = self.transcript_revision.wrapping_add(1);
        self.transcript_layout = None;
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
                    width,
                    self.transcript_details_expanded,
                ),
            });
        }
        let lines = &self
            .transcript_layout
            .as_ref()
            .expect("transcript layout was initialized")
            .lines;
        self.max_scroll = lines.len().saturating_sub(usize::from(height));
        self.page_rows = usize::from(height.max(1));
        if self.follow_tail {
            self.scroll_offset = self.max_scroll;
        } else {
            self.scroll_offset = self.scroll_offset.min(self.max_scroll);
        }
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
    }

    fn scroll_down(&mut self, rows: usize) {
        let current = if self.follow_tail {
            self.max_scroll
        } else {
            self.scroll_offset
        };
        self.scroll_offset = current.saturating_add(rows.max(1)).min(self.max_scroll);
        self.follow_tail = self.scroll_offset == self.max_scroll;
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
    }

    fn scroll_end(&mut self) {
        self.scroll_offset = self.max_scroll;
        self.follow_tail = true;
    }

    fn scroll_delta(&mut self, delta: i32) {
        if delta < 0 {
            self.scroll_up(delta.unsigned_abs() as usize);
        } else if delta > 0 {
            self.scroll_down(delta.unsigned_abs() as usize);
        }
    }

    fn mouse_scroll_delta(&self, event: MouseEvent) -> i32 {
        if event.row < self.transcript_top || event.row >= self.transcript_bottom {
            return 0;
        }
        match event.kind {
            MouseEventKind::ScrollUp => -1,
            MouseEventKind::ScrollDown => 1,
            _ => 0,
        }
    }
}

enum RuntimeAction {
    Submit(String),
    Approval { call_id: String, approved: bool },
    Drive(CoreResponse),
}

enum RuntimeUpdate {
    Progress {
        snapshot: CoreSnapshot,
        status: String,
    },
    Settled(RuntimeFinal),
    Failed(String),
}

struct RuntimeFinal {
    snapshot: CoreSnapshot,
    status: String,
    approval: Option<CoreToolCall>,
}

enum KeyAction {
    None,
    Quit,
    Cancel,
    Submit(String),
    Approval { call_id: String, approved: bool },
}

struct RuntimeResources {
    core: CoreClient,
    plugins: PluginManager,
    provider: ModelClient,
}

type SharedRuntime = Arc<Mutex<RuntimeResources>>;

struct ActiveRuntime {
    handle: JoinHandle<()>,
    cancel: CancellationToken,
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

#[tokio::main]
async fn main() -> Result<(), AppError> {
    install_terminal_panic_hook();
    let args = Args::parse();
    let session = args.session.clone();
    if let Some(path) = &session {
        let parent = path.parent().ok_or(AppError::SessionPathWithoutParent)?;
        tokio::fs::create_dir_all(parent).await?;
    }
    let core_path = args.core.clone().unwrap_or_else(default_core_path);
    let git_plugin_path = args
        .git_plugin
        .clone()
        .unwrap_or_else(default_git_plugin_path);
    let mut core = CoreClient::spawn(&core_path, session.as_deref()).await?;
    let plugins = PluginManager::spawn(&args.plugin, &git_plugin_path).await?;
    let mut safe_tools: Vec<String> = plugins
        .model_tools()
        .iter()
        .filter(|tool| tool.name == "read")
        .map(|tool| tool.name.clone())
        .collect();
    if plugins.has_tool("git_read") {
        safe_tools.push("git_read".to_owned());
    }
    let provider = ModelClient::from_args(&args).await?;
    let initial = core.snapshot().await?;
    let mut app = App::new(initial.snapshot);
    let initial_action = match app.snapshot.phase.as_str() {
        "idle" => {
            let configured = core.event(&CoreEvent::configure_tools(safe_tools)).await?;
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
                    CrosstermEvent::Key(key) if key.kind == KeyEventKind::Press => {
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
                            KeyAction::Submit(text) if active.is_none() => {
                                app.busy = true;
                                app.status = "Submitting…".to_owned();
                                active = Some(spawn_runtime_action(
                                    Arc::clone(&runtime),
                                    RuntimeAction::Submit(text),
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
                            KeyAction::Submit(_) | KeyAction::Approval { .. } => {}
                        }
                    }
                    CrosstermEvent::Mouse(mouse) => {
                        let delta = app.mouse_scroll_delta(mouse);
                        if delta != 0 {
                            pending_scroll = pending_scroll.saturating_add(delta);
                            dirty = true;
                        }
                    }
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
                    Some(RuntimeUpdate::Settled(final_state)) => {
                        if let Some(active) = active.take() {
                            let _ = active.handle.await;
                        }
                        app.set_snapshot(final_state.snapshot);
                        app.status = final_state.status;
                        app.approval = final_state.approval;
                        app.busy = false;
                        dirty = true;
                    }
                    Some(RuntimeUpdate::Failed(error)) => {
                        if let Some(active) = active.take() {
                            let _ = active.handle.await;
                        }
                        app.status = format!("Error: {error}");
                        app.busy = false;
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

fn handle_key(key: KeyEvent, app: &mut App) -> KeyAction {
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
        return KeyAction::Quit;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('e')) {
        app.toggle_transcript_details();
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
    if app.approval.is_some() {
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
    match key.code {
        KeyCode::Enter if !app.busy && !app.input.trim().is_empty() => {
            KeyAction::Submit(std::mem::take(&mut app.input))
        }
        KeyCode::Backspace => {
            app.input.pop();
            KeyAction::None
        }
        KeyCode::Esc => {
            app.input.clear();
            KeyAction::None
        }
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.input.push(character);
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
    let handle = tokio::spawn(async move {
        match run_runtime_action(&runtime, action, &updates, &task_cancel).await {
            Ok(final_state) => {
                let _ = updates.send(RuntimeUpdate::Settled(final_state));
            }
            Err(error) => {
                let _ = updates.send(RuntimeUpdate::Failed(error.to_string()));
            }
        }
    });
    ActiveRuntime { handle, cancel }
}

async fn run_runtime_action(
    runtime: &SharedRuntime,
    action: RuntimeAction,
    updates: &mpsc::UnboundedSender<RuntimeUpdate>,
    cancel: &CancellationToken,
) -> Result<RuntimeFinal, AppError> {
    let mut runtime = runtime.lock().await;
    let response = match action {
        RuntimeAction::Submit(text) => runtime.core.event(&CoreEvent::submit(text)).await?,
        RuntimeAction::Approval { call_id, approved } => {
            runtime
                .core
                .event(&CoreEvent::approval(call_id, approved))
                .await?
        }
        RuntimeAction::Drive(response) => response,
    };
    drive_response(response, &mut runtime, updates, cancel).await
}

async fn drive_response(
    response: CoreResponse,
    runtime: &mut RuntimeResources,
    updates: &mpsc::UnboundedSender<RuntimeUpdate>,
    cancel: &CancellationToken,
) -> Result<RuntimeFinal, AppError> {
    let mut snapshot = response.snapshot;
    let mut effects = response.effects;
    send_runtime_progress(updates, &snapshot, "Processing…");
    while let Some(effect) = effects.pop() {
        if cancel.is_cancelled() {
            return cancel_runtime(runtime).await;
        }
        match effect.kind.as_str() {
            "request_model" => {
                send_runtime_progress(updates, &snapshot, "Waiting for model…");
                let completion = tokio::select! {
                    () = cancel.cancelled() => return cancel_runtime(runtime).await,
                    result = runtime.provider.complete(&snapshot, runtime.plugins.model_tools()) => result?,
                };
                let next = runtime
                    .core
                    .event(&CoreEvent::model_completed(
                        completion.content,
                        completion.tool_calls,
                    ))
                    .await?;
                snapshot = next.snapshot;
                effects.extend(next.effects);
            }
            "request_approval" => {
                let call = effect.call.ok_or_else(|| AppError::MissingCoreEffectCall {
                    kind: "request_approval".to_owned(),
                })?;
                return Ok(RuntimeFinal {
                    snapshot,
                    status: format!("Approve {}? [y]es / [n]o", call.display_label()),
                    approval: Some(call),
                });
            }
            "invoke_tool" => {
                let call = effect.call.ok_or_else(|| AppError::MissingCoreEffectCall {
                    kind: "invoke_tool".to_owned(),
                })?;
                send_runtime_progress(
                    updates,
                    &snapshot,
                    &format!("Running {}…", call.display_label()),
                );
                let tool_result = tokio::select! {
                    () = cancel.cancelled() => None,
                    result = runtime.plugins.call(&call) => Some(result),
                };
                let Some(tool_result) = tool_result else {
                    return cancel_runtime(runtime).await;
                };
                let (content, is_error) = match tool_result {
                    Ok(result) => (result.output, false),
                    Err(error) => (format!("Plugin execution failed: {error}"), true),
                };
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
    Ok(RuntimeFinal {
        snapshot,
        status: "Ready".to_owned(),
        approval: None,
    })
}

async fn cancel_runtime(runtime: &mut RuntimeResources) -> Result<RuntimeFinal, AppError> {
    runtime.plugins.retire().await;
    let response = runtime.core.event(&CoreEvent::abort()).await?;
    Ok(RuntimeFinal {
        snapshot: response.snapshot,
        status: "Cancelled".to_owned(),
        approval: None,
    })
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
fn draw(frame: &mut ratatui::Frame<'_>, app: &mut App) {
    let [header, transcript, input, status] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(4),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    let phase_color = match app.snapshot.phase.as_str() {
        "idle" => COLOR_ASSISTANT,
        "waiting_approval" => COLOR_TOOL,
        "waiting_model" | "waiting_tool" => COLOR_ACCENT,
        _ => COLOR_MUTED,
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
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
        ])),
        header,
    );

    let content_width = transcript.width.saturating_sub(2);
    let content_height = transcript.height;
    let visible_transcript = app.visible_transcript(content_width, content_height);
    app.transcript_top = transcript.y;
    app.transcript_bottom = transcript.y.saturating_add(transcript.height);
    frame.render_widget(
        Paragraph::new(visible_transcript).block(Block::default().padding(Padding::horizontal(1))),
        transcript,
    );

    let (input_title, input_color) = match app.approval.as_ref() {
        Some(call) => (
            format!(" Approval · {} · y allow · n deny ", call.display_label()),
            COLOR_TOOL,
        ),
        None if app.busy => (" Message · Esc cancel ".to_owned(), COLOR_ACCENT),
        None => (" Message · Enter send ".to_owned(), COLOR_MUTED),
    };
    frame.render_widget(
        Paragraph::new(app.input.as_str())
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
        " · Ctrl+E collapse details"
    } else {
        " · Ctrl+E expand details"
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
                    " · pending {} · safe {}{scroll}{details}",
                    app.snapshot.pending_calls.len(),
                    app.snapshot.safe_tools.len()
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
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
    use ratatui::{Terminal, backend::TestBackend, text::Line};

    use super::{
        App, COLOR_ACCENT, COLOR_MUTED, CoreMessage, CoreSnapshot, CoreToolCall,
        TranscriptLayoutCache, anthropic_messages, build_transcript_lines, draw, handle_key,
        parse_openai_tool_calls,
    };

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
            80,
            true,
        );
        assert_eq!(lines[1].to_string(), "  first");
        assert_eq!(lines[2].to_string(), "  second");
        let tool_line = lines
            .iter()
            .find(|line| line.to_string().contains("git status --short"))
            .expect("tool line must be present");
        assert_eq!(tool_line.spans[0].style.fg, Some(COLOR_MUTED));
        assert_eq!(tool_line.spans[1].style.fg, Some(COLOR_ACCENT));
        assert_eq!(tool_line.spans[2].style.fg, Some(COLOR_MUTED));
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
        let collapsed = build_transcript_lines(&messages, 32, false);
        let expanded = build_transcript_lines(&messages, 32, true);
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
        let collapsed = build_transcript_lines(&messages, 80, false);
        let expanded = build_transcript_lines(&messages, 80, true);
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
    fn control_e_toggles_transcript_details_and_invalidates_layout() {
        let mut app = App::new(CoreSnapshot::default());
        app.transcript_layout = Some(TranscriptLayoutCache {
            revision: app.transcript_revision,
            width: 80,
            lines: vec![Line::default()],
        });
        let _ = handle_key(
            KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL),
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
        let backend = TestBackend::new(100, 18);
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
            "› You",
            "• Assistant",
            "└─ bash  git status --short",
            "└ Tool result",
            "Message · Enter send",
            "Ready · pending 0 · safe 2",
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
    fn mouse_wheel_only_scrolls_over_transcript() {
        let mut app = App::new(CoreSnapshot::default());
        app.max_scroll = 20;
        app.scroll_offset = 20;
        app.transcript_top = 2;
        app.transcript_bottom = 12;

        let inside = MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 3,
            row: 5,
            modifiers: KeyModifiers::NONE,
        };
        app.scroll_delta(app.mouse_scroll_delta(inside));
        assert_eq!(app.scroll_offset, 19);

        let outside = MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 3,
            row: 15,
            modifiers: KeyModifiers::NONE,
        };
        app.scroll_delta(app.mouse_scroll_delta(outside));
        assert_eq!(app.scroll_offset, 19);
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

        let rendered_lines = 7_usize;
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
