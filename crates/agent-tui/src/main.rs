#![deny(warnings)]
#![deny(clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    env,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use agent_plugin_protocol::{
    CallToolParams, DEFAULT_MAX_FRAME_BYTES, InitializeParams, InitializeResult, Limits,
    RequestEnvelope as PluginRequest, RequestOperation as PluginOperation, ToolListResult,
    ToolResult, ToolSpec, read_response, write_request,
};
use clap::{Parser, ValueEnum};
use crossterm::{
    event::{Event as CrosstermEvent, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures_util::StreamExt;
#[cfg(unix)]
use nix::{
    sys::signal::{Signal, killpg},
    unistd::Pid,
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    style::{Color, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
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
    time::timeout,
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

const CORE_PROTOCOL_VERSION: u64 = 1;
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a careful coding agent. Use the declared tools when needed. Explain changes briefly.";
const MAX_PROVIDER_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Provider {
    Openai,
    Anthropic,
    Linewise,
}

#[derive(Debug, Parser)]
#[command(
    name = "agent-tui",
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct CoreToolCall {
    call_id: String,
    name: String,
    arguments: Value,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
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
    Protocol(#[from] agent_plugin_protocol::ProtocolError),
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
                    host: agent_plugin_protocol::HostIdentity {
                        name: "lean-coding-agent".to_owned(),
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
                operation: PluginOperation::ListTools(agent_plugin_protocol::EmptyParams {}),
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
            operation: PluginOperation::Shutdown(agent_plugin_protocol::EmptyParams {}),
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

struct App {
    input: String,
    status: String,
    snapshot: CoreSnapshot,
    approval: Option<CoreToolCall>,
    busy: bool,
}

impl App {
    fn new(snapshot: CoreSnapshot) -> Self {
        Self {
            input: String::new(),
            status: "Ready".to_owned(),
            snapshot,
            approval: None,
            busy: false,
        }
    }

    fn transcript(&self) -> Text<'static> {
        let mut lines = Vec::new();
        for message in &self.snapshot.messages {
            let color = match message.role.as_str() {
                "user" => Color::Cyan,
                "assistant" => Color::Green,
                "tool" if message.is_error => Color::Red,
                "tool" => Color::Yellow,
                _ => Color::Gray,
            };
            let label = message.role.to_uppercase();
            lines.push(Line::from(Span::styled(
                format!("{label}: "),
                Style::default().fg(color),
            )));
            if !message.content.is_empty() {
                lines.push(Line::from(message.content.clone()));
            }
            for call in &message.tool_calls {
                lines.push(Line::from(Span::styled(
                    format!("tool {} {}({})", call.call_id, call.name, call.arguments),
                    Style::default().fg(Color::Magenta),
                )));
            }
            lines.push(Line::from(""));
        }
        Text::from(lines)
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
    plugin: PluginClient,
    plugin_path: PathBuf,
    provider: ModelClient,
    tool_names: Vec<String>,
}

impl RuntimeResources {
    async fn ensure_plugin(&mut self) -> Result<(), AppError> {
        if self.plugin.available {
            return Ok(());
        }
        let replacement = PluginClient::spawn(&self.plugin_path).await?;
        let replacement_names: Vec<&str> = replacement
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect();
        let expected_names: Vec<&str> = self.tool_names.iter().map(String::as_str).collect();
        if replacement_names != expected_names {
            return Err(PluginClientError::CatalogueChanged.into());
        }
        self.plugin = replacement;
        Ok(())
    }
}

type SharedRuntime = Arc<Mutex<RuntimeResources>>;

struct ActiveRuntime {
    handle: JoinHandle<()>,
    cancel: CancellationToken,
}

#[tokio::main]
async fn main() -> Result<(), AppError> {
    let args = Args::parse();
    let session = args.session.clone();
    if let Some(path) = &session {
        let parent = path.parent().ok_or(AppError::SessionPathWithoutParent)?;
        tokio::fs::create_dir_all(parent).await?;
    }
    let core_path = args.core.clone().unwrap_or_else(default_core_path);
    let mut core = CoreClient::spawn(&core_path, session.as_deref()).await?;
    let plugin = PluginClient::spawn(&args.plugin).await?;
    let safe_tools: Vec<String> = plugin
        .tools
        .iter()
        .filter(|tool| tool.name == "read")
        .map(|tool| tool.name.clone())
        .collect();
    let tool_names = plugin.tools.iter().map(|tool| tool.name.clone()).collect();
    let provider = ModelClient::from_args(&args).await?;
    let initial = core.snapshot().await?;
    let mut app = App::new(initial.snapshot);
    let initial_action = match app.snapshot.phase.as_str() {
        "idle" => {
            let configured = core.event(&CoreEvent::configure_tools(safe_tools)).await?;
            app.snapshot = configured.snapshot;
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
            app.status = format!("Approve {}? [y]es / [n]o", call.name);
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
        plugin,
        plugin_path: args.plugin.clone(),
        provider,
        tool_names,
    }));
    run_tui(&mut app, Arc::clone(&runtime), initial_action).await?;
    let mut runtime = runtime.lock().await;
    let _ = runtime.core.shutdown().await;
    runtime.plugin.shutdown().await;
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

async fn run_tui(
    app: &mut App,
    runtime: SharedRuntime,
    initial_action: Option<RuntimeAction>,
) -> Result<(), AppError> {
    enable_raw_mode()?;
    let mut terminal_stdout = std::io::stdout();
    execute!(terminal_stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(terminal_stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = tui_loop(&mut terminal, app, runtime, initial_action).await;
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
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
    loop {
        terminal.draw(|frame| draw(frame, app))?;
        tokio::select! {
            event = events.next() => {
                let Some(event) = event else {
                    cancel_and_wait(&mut active, app).await;
                    return Ok(());
                };
                match event? {
                    CrosstermEvent::Key(key) if key.kind == KeyEventKind::Press => {
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
                    CrosstermEvent::Resize(_, _) => {}
                    _ => {}
                }
            }
            update = update_rx.recv() => {
                match update {
                    Some(RuntimeUpdate::Progress { snapshot, status }) => {
                        app.snapshot = snapshot;
                        app.status = status;
                        app.busy = true;
                    }
                    Some(RuntimeUpdate::Settled(final_state)) => {
                        if let Some(active) = active.take() {
                            let _ = active.handle.await;
                        }
                        app.snapshot = final_state.snapshot;
                        app.status = final_state.status;
                        app.approval = final_state.approval;
                        app.busy = false;
                    }
                    Some(RuntimeUpdate::Failed(error)) => {
                        if let Some(active) = active.take() {
                            let _ = active.handle.await;
                        }
                        app.status = format!("Error: {error}");
                        app.busy = false;
                    }
                    None => return Ok(()),
                }
            }
        }
    }
}

fn handle_key(key: KeyEvent, app: &mut App) -> KeyAction {
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
        return KeyAction::Quit;
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
                    result = runtime.provider.complete(&snapshot, &runtime.plugin.tools) => result?,
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
                    status: format!("Approve {}? [y]es / [n]o", call.name),
                    approval: Some(call),
                });
            }
            "invoke_tool" => {
                let call = effect.call.ok_or_else(|| AppError::MissingCoreEffectCall {
                    kind: "invoke_tool".to_owned(),
                })?;
                runtime.ensure_plugin().await?;
                send_runtime_progress(updates, &snapshot, &format!("Running {}…", call.name));
                let tool_result = tokio::select! {
                    () = cancel.cancelled() => None,
                    result = runtime.plugin.call(&call) => Some(result),
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
    runtime.plugin.retire().await;
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

fn draw(frame: &mut ratatui::Frame<'_>, app: &App) {
    let [header, transcript, input, status] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(4),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    let pending = app.snapshot.pending_calls.len();
    let safe = app.snapshot.safe_tools.len();
    let approval = app
        .approval
        .as_ref()
        .map(|call| format!(" approval: {}", call.name))
        .unwrap_or_default();
    frame.render_widget(
        Paragraph::new(format!(
            "Lean Coding Agent · phase: {} · pending: {pending} · active: {} · safe tools: {safe}{approval}",
            app.snapshot.phase, app.snapshot.current_call
        ))
        .style(Style::default().fg(Color::Blue)),
        header,
    );
    frame.render_widget(
        Paragraph::new(app.transcript())
            .block(Block::default().borders(Borders::ALL).title("Transcript"))
            .wrap(Wrap { trim: false }),
        transcript,
    );
    let input_title = if app.approval.is_some() {
        "Approval is pending"
    } else {
        "Prompt"
    };
    frame.render_widget(
        Paragraph::new(app.input.as_str())
            .block(Block::default().borders(Borders::ALL).title(input_title)),
        input,
    );
    frame.render_widget(
        Paragraph::new(app.status.as_str()).style(Style::default().fg(Color::Gray)),
        status,
    );
}

fn default_core_path() -> PathBuf {
    if let Some(path) = env::var_os("LEAN_AGENT_CORE_BIN") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../lean_agent_core/.lake/build/bin/lean_agent_core")
}

#[cfg(test)]
mod tests {
    use super::{CoreMessage, anthropic_messages, parse_openai_tool_calls};

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
}
