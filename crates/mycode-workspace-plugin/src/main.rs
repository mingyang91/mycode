#![deny(warnings)]
#![deny(clippy::unwrap_used)]

use std::{
    env,
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use ignore::WalkBuilder;
use mycode_plugin_protocol::{
    CallToolParams, DEFAULT_MAX_FRAME_BYTES, DEFAULT_MAX_TOOL_OUTPUT_BYTES, EmptyParams,
    InitializeResult, PluginErrorCode, PluginIdentity, ProtocolRange, RequestEnvelope,
    RequestOperation, ResponseEnvelope, ToolListResult, ToolResult, ToolSpec, read_request,
    tool_failure, write_response,
};
use regex::RegexBuilder;
use serde_json::{Value, json};
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt, stdin, stdout},
    process::Command,
    time::timeout,
};
use uuid::Uuid;

const MAX_BASH_TIMEOUT_MS: u64 = 300_000;
const DEFAULT_BASH_TIMEOUT_MS: u64 = 120_000;
const MAX_EDIT_FILE_BYTES: usize = 4 * 1024 * 1024;
const MAX_GREP_PATTERN_BYTES: usize = 4 * 1024;
const MAX_GREP_REGEX_BYTES: usize = 1024 * 1024;
const MAX_GREP_FILE_BYTES: usize = 4 * 1024 * 1024;
const MAX_GREP_LINE_BYTES: usize = 4 * 1024;
const DEFAULT_GREP_RESULTS: usize = 200;
const MAX_GREP_RESULTS: usize = 1000;

#[derive(Debug, Error)]
enum WorkspaceError {
    #[error("argument {name} must be a string")]
    MissingStringArgument { name: &'static str },
    #[error("argument timeoutMs must be an integer between 1 and {MAX_BASH_TIMEOUT_MS}")]
    InvalidTimeout,
    #[error("argument {name} must be a boolean")]
    InvalidBooleanArgument { name: &'static str },
    #[error("argument maxResults must be an integer between 1 and {MAX_GREP_RESULTS}")]
    InvalidMaxResults,
    #[error("path must be a non-empty relative path")]
    InvalidRelativePath,
    #[error("path resolves outside the configured workspace")]
    PathEscapesWorkspace,
    #[error("path parent does not exist")]
    MissingParent,
    #[error("tool is not declared by this plugin")]
    UnknownTool,
    #[error("grep pattern must not be empty")]
    EmptyGrepPattern,
    #[error("grep pattern exceeds {MAX_GREP_PATTERN_BYTES} bytes")]
    GrepPatternTooLong,
    #[error("grep pattern is invalid: {0}")]
    InvalidGrepPattern(#[source] regex::Error),
    #[error("file content is not valid UTF-8")]
    InvalidUtf8,
    #[error("the expected text must occur exactly once")]
    EditMatchCount,
    #[error("tool output exceeded {limit} bytes")]
    OutputTooLarge { limit: usize },
    #[error("child process did not provide piped output")]
    MissingChildPipe,
    #[error("shell command exceeded its deadline")]
    CommandTimedOut,
    #[error("workspace search traversal failed: {0}")]
    Walk(#[from] ignore::Error),
    #[error("workspace search task failed: {0}")]
    SearchTask(#[from] tokio::task::JoinError),
    #[error("filesystem or process operation failed: {0}")]
    Io(#[from] std::io::Error),
}

#[tokio::main]
async fn main() -> Result<(), WorkspaceError> {
    let workspace = canonical_workspace()?;
    let mut input = stdin();
    let mut output = stdout();
    let mut initialized = false;

    loop {
        let request = match read_request(&mut input, DEFAULT_MAX_FRAME_BYTES).await {
            Ok(request) => request,
            Err(error) => {
                eprintln!("plugin protocol error: {error}");
                return Ok(());
            }
        };
        let was_initialize = matches!(&request.operation, RequestOperation::Initialize(_));
        let (response, shutdown) = handle_request(request, &workspace, initialized).await;
        if was_initialize && response.ok {
            initialized = true;
        }
        if let Err(error) = write_response(&mut output, &response).await {
            eprintln!("plugin response write failed: {error}");
            return Ok(());
        }
        if shutdown {
            return Ok(());
        }
    }
}

async fn handle_request(
    request: RequestEnvelope,
    workspace: &Path,
    initialized: bool,
) -> (ResponseEnvelope, bool) {
    let request_id = request.id.clone();
    match request.operation {
        RequestOperation::Initialize(_) if !initialized => success(
            request_id,
            &InitializeResult {
                plugin: PluginIdentity {
                    name: "workspace".to_owned(),
                    version: env!("CARGO_PKG_VERSION").to_owned(),
                },
                protocol: ProtocolRange {
                    min_version: 1,
                    max_version: 1,
                },
            },
        ),
        RequestOperation::Initialize(_) => failure(
            request_id,
            PluginErrorCode::InvalidRequest,
            "plugin is already initialized",
        ),
        RequestOperation::ListTools(_) if initialized => {
            success(request_id, &ToolListResult { tools: tools() })
        }
        RequestOperation::ListTools(_) => failure(
            request_id,
            PluginErrorCode::InvalidRequest,
            "initialize must complete before tool discovery",
        ),
        RequestOperation::CallTool(call) if initialized => {
            match execute_tool(workspace, call).await {
                Ok(result) => success(request_id, &result),
                Err(error) => failure(request_id, PluginErrorCode::ToolFailed, error.to_string()),
            }
        }
        RequestOperation::CallTool(_) => failure(
            request_id,
            PluginErrorCode::InvalidRequest,
            "initialize must complete before tool execution",
        ),
        RequestOperation::Shutdown(EmptyParams {}) if initialized => {
            let (response, _) = success(request_id, &json!({}));
            (response, true)
        }
        RequestOperation::Shutdown(EmptyParams {}) => failure(
            request_id,
            PluginErrorCode::InvalidRequest,
            "initialize must complete before shutdown",
        ),
    }
}

fn success<T: serde::Serialize>(id: String, result: &T) -> (ResponseEnvelope, bool) {
    let fallback_id = id.clone();
    match ResponseEnvelope::success(id, result) {
        Ok(response) => (response, false),
        Err(error) => failure(fallback_id, PluginErrorCode::Internal, error.to_string()),
    }
}

fn failure(
    id: String,
    code: PluginErrorCode,
    message: impl Into<String>,
) -> (ResponseEnvelope, bool) {
    (
        ResponseEnvelope::failure(id, tool_failure(code, message.into(), false)),
        false,
    )
}

fn tools() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "read".to_owned(),
            description: "Read a UTF-8 text file inside the workspace.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "grep".to_owned(),
            description: "Search UTF-8 workspace files with a Rust regular expression. Read-only and respects ignore files.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "minLength": 1},
                    "path": {"type": "string", "default": "."},
                    "caseSensitive": {"type": "boolean", "default": true},
                    "maxResults": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_GREP_RESULTS,
                        "default": DEFAULT_GREP_RESULTS
                    }
                },
                "required": ["pattern"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "write".to_owned(),
            description: "Replace a UTF-8 text file inside the workspace. Requires approval.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "edit".to_owned(),
            description: "Replace exactly one text occurrence inside a UTF-8 workspace file. Requires approval.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "oldText": {"type": "string"},
                    "newText": {"type": "string"}
                },
                "required": ["path", "oldText", "newText"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "bash".to_owned(),
            description: "Run one shell command from the workspace. Requires approval.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeoutMs": {"type": "integer", "minimum": 1, "maximum": MAX_BASH_TIMEOUT_MS}
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        },
    ]
}

async fn execute_tool(
    workspace: &Path,
    call: CallToolParams,
) -> Result<ToolResult, WorkspaceError> {
    if call.name == "grep" {
        return grep_files(workspace, &call.arguments).await;
    }
    let output = match call.name.as_str() {
        "read" => read_file(workspace, &call.arguments).await?,
        "write" => write_file(workspace, &call.arguments).await?,
        "edit" => edit_file(workspace, &call.arguments).await?,
        "bash" => run_bash(workspace, &call.arguments).await?,
        _ => return Err(WorkspaceError::UnknownTool),
    };
    bounded_output(output)
}

async fn read_file(workspace: &Path, arguments: &Value) -> Result<String, WorkspaceError> {
    let path = existing_path(workspace, required_string(arguments, "path")?).await?;
    let file = tokio::fs::File::open(path).await?;
    let bytes = read_bounded(file, DEFAULT_MAX_TOOL_OUTPUT_BYTES).await?;
    String::from_utf8(bytes).map_err(|_| WorkspaceError::InvalidUtf8)
}
async fn grep_files(workspace: &Path, arguments: &Value) -> Result<ToolResult, WorkspaceError> {
    let pattern = required_string(arguments, "pattern")?;
    if pattern.is_empty() {
        return Err(WorkspaceError::EmptyGrepPattern);
    }
    if pattern.len() > MAX_GREP_PATTERN_BYTES {
        return Err(WorkspaceError::GrepPatternTooLong);
    }
    let path = existing_path(
        workspace,
        optional_string(arguments, "path")?.unwrap_or("."),
    )
    .await?;
    let case_sensitive = optional_bool(arguments, "caseSensitive", true)?;
    let max_results = optional_grep_results(arguments)?;
    let workspace = workspace.to_owned();
    let pattern = pattern.to_owned();
    tokio::task::spawn_blocking(move || {
        search_files(&workspace, &path, &pattern, case_sensitive, max_results)
    })
    .await?
}

fn search_files(
    workspace: &Path,
    root: &Path,
    pattern: &str,
    case_sensitive: bool,
    max_results: usize,
) -> Result<ToolResult, WorkspaceError> {
    let expression = RegexBuilder::new(pattern)
        .case_insensitive(!case_sensitive)
        .size_limit(MAX_GREP_REGEX_BYTES)
        .dfa_size_limit(MAX_GREP_REGEX_BYTES)
        .build()
        .map_err(WorkspaceError::InvalidGrepPattern)?;
    let mut walker = WalkBuilder::new(root);
    walker
        .standard_filters(true)
        .require_git(false)
        .follow_links(false)
        .sort_by_file_path(|left, right| left.cmp(right));
    let mut output = String::new();
    let mut matches = 0_usize;
    let mut truncated = false;

    'entries: for entry in walker.build() {
        let entry = entry?;
        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }
        let file = File::open(entry.path())?;
        let mut bytes = Vec::with_capacity(8192);
        file.take((MAX_GREP_FILE_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_GREP_FILE_BYTES {
            continue;
        }
        let Ok(content) = String::from_utf8(bytes) else {
            continue;
        };
        let relative = entry
            .path()
            .strip_prefix(workspace)
            .map_err(|_| WorkspaceError::PathEscapesWorkspace)?;
        for (line_index, line) in content.lines().enumerate() {
            if !expression.is_match(line) {
                continue;
            }
            if matches == max_results {
                truncated = true;
                break 'entries;
            }
            let (line, line_truncated) = grep_line_fragment(line);
            let suffix = if line_truncated { "…" } else { "" };
            let record = format!("{}:{}:{line}{suffix}", relative.display(), line_index + 1);
            let separator_bytes = usize::from(!output.is_empty());
            if output
                .len()
                .saturating_add(separator_bytes)
                .saturating_add(record.len())
                > DEFAULT_MAX_TOOL_OUTPUT_BYTES
            {
                truncated = true;
                break 'entries;
            }
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(&record);
            matches += 1;
            truncated |= line_truncated;
        }
    }
    if output.is_empty() {
        output.push_str("No matches.");
    }
    Ok(ToolResult { output, truncated })
}

fn grep_line_fragment(line: &str) -> (&str, bool) {
    if line.len() <= MAX_GREP_LINE_BYTES {
        return (line, false);
    }
    let mut end = MAX_GREP_LINE_BYTES;
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    (&line[..end], true)
}

async fn write_file(workspace: &Path, arguments: &Value) -> Result<String, WorkspaceError> {
    let path = writable_path(workspace, required_string(arguments, "path")?).await?;
    let content = required_string(arguments, "content")?;
    atomic_replace(&path, content.as_bytes()).await?;
    Ok(format!(
        "Wrote {} bytes to {}.",
        content.len(),
        path.display()
    ))
}

async fn edit_file(workspace: &Path, arguments: &Value) -> Result<String, WorkspaceError> {
    let path = existing_path(workspace, required_string(arguments, "path")?).await?;
    let old_text = required_string(arguments, "oldText")?;
    let new_text = required_string(arguments, "newText")?;
    let content = String::from_utf8(
        read_bounded(tokio::fs::File::open(&path).await?, MAX_EDIT_FILE_BYTES).await?,
    )
    .map_err(|_| WorkspaceError::InvalidUtf8)?;
    if old_text.is_empty() || content.matches(old_text).count() != 1 {
        return Err(WorkspaceError::EditMatchCount);
    }
    let replacement = content.replacen(old_text, new_text, 1);
    atomic_replace(&path, replacement.as_bytes()).await?;
    Ok(format!("Edited {}.", path.display()))
}

async fn atomic_replace(path: &Path, content: &[u8]) -> Result<(), WorkspaceError> {
    let parent = path.parent().ok_or(WorkspaceError::MissingParent)?;
    let temporary = parent.join(format!(".mycode-write-{}.tmp", Uuid::new_v4()));
    let result = async {
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .await?;
        file.write_all(content).await?;
        file.flush().await?;
        drop(file);
        tokio::fs::rename(&temporary, path).await?;
        Ok(())
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&temporary).await;
    }
    result
}
async fn run_bash(workspace: &Path, arguments: &Value) -> Result<String, WorkspaceError> {
    let command = required_string(arguments, "command")?;
    let timeout_ms = optional_timeout(arguments)?;
    let mut child = Command::new("/bin/sh")
        .arg("-lc")
        .arg(command)
        .current_dir(workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or(WorkspaceError::MissingChildPipe)?;
    let stderr = child
        .stderr
        .take()
        .ok_or(WorkspaceError::MissingChildPipe)?;
    let completed = timeout(Duration::from_millis(timeout_ms), async {
        let (stdout, stderr, status) = tokio::try_join!(
            read_bounded(stdout, DEFAULT_MAX_TOOL_OUTPUT_BYTES),
            read_bounded(stderr, DEFAULT_MAX_TOOL_OUTPUT_BYTES),
            async { child.wait().await.map_err(WorkspaceError::from) }
        )?;
        Ok::<_, WorkspaceError>((stdout, stderr, status))
    })
    .await;
    let (stdout, stderr, status) = match completed {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(error);
        }
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(WorkspaceError::CommandTimedOut);
        }
    };
    let mut text = String::from_utf8_lossy(&stdout).into_owned();
    let stderr = String::from_utf8_lossy(&stderr);
    if !stderr.is_empty() {
        text.push_str("\n[stderr]\n");
        text.push_str(&stderr);
    }
    if !status.success() {
        text.push_str(&format!("\n[exit status: {status}]"));
    }
    Ok(text)
}

async fn read_bounded<R>(mut reader: R, limit: usize) -> Result<Vec<u8>, WorkspaceError>
where
    R: AsyncRead + Unpin,
{
    let mut bytes = Vec::with_capacity(limit.min(8192));
    let mut chunk = [0_u8; 8192];
    loop {
        let count = reader.read(&mut chunk).await?;
        if count == 0 {
            return Ok(bytes);
        }
        if bytes.len() + count > limit {
            return Err(WorkspaceError::OutputTooLarge { limit });
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
}

fn required_string<'a>(
    arguments: &'a Value,
    name: &'static str,
) -> Result<&'a str, WorkspaceError> {
    arguments
        .get(name)
        .and_then(Value::as_str)
        .ok_or(WorkspaceError::MissingStringArgument { name })
}
fn optional_string<'a>(
    arguments: &'a Value,
    name: &'static str,
) -> Result<Option<&'a str>, WorkspaceError> {
    match arguments.get(name) {
        None => Ok(None),
        Some(value) => value
            .as_str()
            .map(Some)
            .ok_or(WorkspaceError::MissingStringArgument { name }),
    }
}

fn optional_bool(
    arguments: &Value,
    name: &'static str,
    default: bool,
) -> Result<bool, WorkspaceError> {
    match arguments.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or(WorkspaceError::InvalidBooleanArgument { name }),
    }
}

fn optional_grep_results(arguments: &Value) -> Result<usize, WorkspaceError> {
    match arguments.get("maxResults") {
        None => Ok(DEFAULT_GREP_RESULTS),
        Some(value) => value
            .as_u64()
            .and_then(|results| usize::try_from(results).ok())
            .filter(|results| (1..=MAX_GREP_RESULTS).contains(results))
            .ok_or(WorkspaceError::InvalidMaxResults),
    }
}

fn optional_timeout(arguments: &Value) -> Result<u64, WorkspaceError> {
    match arguments.get("timeoutMs") {
        None => Ok(DEFAULT_BASH_TIMEOUT_MS),
        Some(value) => match value.as_u64() {
            Some(timeout_ms) if (1..=MAX_BASH_TIMEOUT_MS).contains(&timeout_ms) => Ok(timeout_ms),
            _ => Err(WorkspaceError::InvalidTimeout),
        },
    }
}

fn canonical_workspace() -> Result<PathBuf, WorkspaceError> {
    env::current_dir()?
        .canonicalize()
        .map_err(WorkspaceError::from)
}

async fn existing_path(workspace: &Path, raw_path: &str) -> Result<PathBuf, WorkspaceError> {
    let candidate = lexical_workspace_path(workspace, raw_path)?;
    let canonical = tokio::fs::canonicalize(candidate).await?;
    ensure_under_workspace(workspace, &canonical)?;
    Ok(canonical)
}

async fn writable_path(workspace: &Path, raw_path: &str) -> Result<PathBuf, WorkspaceError> {
    let candidate = lexical_workspace_path(workspace, raw_path)?;
    let parent = candidate.parent().ok_or(WorkspaceError::MissingParent)?;
    let canonical_parent = tokio::fs::canonicalize(parent).await?;
    ensure_under_workspace(workspace, &canonical_parent)?;
    let filename = candidate
        .file_name()
        .ok_or(WorkspaceError::InvalidRelativePath)?;
    Ok(canonical_parent.join(filename))
}

fn lexical_workspace_path(workspace: &Path, raw_path: &str) -> Result<PathBuf, WorkspaceError> {
    let path = Path::new(raw_path);
    if raw_path.is_empty() || path.is_absolute() {
        return Err(WorkspaceError::InvalidRelativePath);
    }
    if path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(WorkspaceError::InvalidRelativePath);
    }
    Ok(workspace.join(path))
}

fn ensure_under_workspace(workspace: &Path, candidate: &Path) -> Result<(), WorkspaceError> {
    if candidate.starts_with(workspace) {
        Ok(())
    } else {
        Err(WorkspaceError::PathEscapesWorkspace)
    }
}

fn bounded_output(output: String) -> Result<ToolResult, WorkspaceError> {
    if output.len() > DEFAULT_MAX_TOOL_OUTPUT_BYTES {
        return Err(WorkspaceError::OutputTooLarge {
            limit: DEFAULT_MAX_TOOL_OUTPUT_BYTES,
        });
    }
    Ok(ToolResult {
        output,
        truncated: false,
    })
}

#[cfg(test)]
mod tests {
    use std::{
        env, fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use mycode_plugin_protocol::{
        CallToolParams, InitializeParams, RequestEnvelope, RequestOperation,
    };
    use serde_json::json;

    use super::{WorkspaceError, execute_tool, handle_request, tools};

    fn request(operation: RequestOperation) -> RequestEnvelope {
        RequestEnvelope {
            v: 1,
            id: "test_1".to_owned(),
            operation,
        }
    }

    #[tokio::test]
    async fn requires_initialization_before_tool_discovery() {
        let workspace = env::current_dir().expect("current directory must exist");
        let (response, shutdown) = handle_request(
            request(RequestOperation::ListTools(
                mycode_plugin_protocol::EmptyParams {},
            )),
            &workspace,
            false,
        )
        .await;
        assert!(!response.ok);
        assert!(!shutdown);

        let (initialized, _) = handle_request(
            request(RequestOperation::Initialize(InitializeParams {
                host: mycode_plugin_protocol::HostIdentity {
                    name: "test-host".to_owned(),
                    version: "1".to_owned(),
                },
                limits: mycode_plugin_protocol::Limits::default(),
            })),
            &workspace,
            false,
        )
        .await;
        assert!(initialized.ok);
    }

    #[tokio::test]
    async fn reads_only_inside_the_workspace() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock must be after the Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!("mycode-plugin-{nonce}"));
        fs::create_dir(&root).expect("temporary workspace must be created");
        fs::write(root.join("note.txt"), "safe content").expect("fixture must write");
        let workspace = root.canonicalize().expect("workspace must canonicalize");
        let result = execute_tool(
            &workspace,
            CallToolParams {
                name: "read".to_owned(),
                arguments: json!({"path": "note.txt"}),
            },
        )
        .await
        .expect("workspace file must be readable");
        assert_eq!(result.output, "safe content");
        let outside = execute_tool(
            &workspace,
            CallToolParams {
                name: "read".to_owned(),
                arguments: json!({"path": "../outside.txt"}),
            },
        )
        .await;
        assert!(outside.is_err());
        fs::remove_dir_all(root).expect("temporary workspace must be removed");
    }

    #[tokio::test]
    async fn refuses_to_buffer_a_read_larger_than_the_output_limit() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock must be after the Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!("mycode-plugin-limit-{nonce}"));
        fs::create_dir(&root).expect("temporary workspace must be created");
        fs::write(
            root.join("large.txt"),
            vec![b'x'; mycode_plugin_protocol::DEFAULT_MAX_TOOL_OUTPUT_BYTES + 1],
        )
        .expect("fixture must write");
        let workspace = root.canonicalize().expect("workspace must canonicalize");
        let result = execute_tool(
            &workspace,
            CallToolParams {
                name: "read".to_owned(),
                arguments: json!({"path": "large.txt"}),
            },
        )
        .await;
        assert!(matches!(
            result,
            Err(super::WorkspaceError::OutputTooLarge { .. })
        ));
        fs::remove_dir_all(root).expect("temporary workspace must be removed");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn write_replaces_a_final_symlink_without_following_it() {
        use std::os::unix::fs::symlink;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock must be after the Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!("mycode-plugin-symlink-{nonce}"));
        let outside = env::temp_dir().join(format!("mycode-outside-{nonce}"));
        fs::create_dir(&root).expect("temporary workspace must be created");
        fs::write(&outside, "outside").expect("outside fixture must write");
        symlink(&outside, root.join("link.txt")).expect("symlink fixture must be created");
        let workspace = root.canonicalize().expect("workspace must canonicalize");
        execute_tool(
            &workspace,
            CallToolParams {
                name: "write".to_owned(),
                arguments: json!({"path": "link.txt", "content": "inside"}),
            },
        )
        .await
        .expect("write must replace the final symlink itself");
        assert_eq!(
            fs::read_to_string(&outside).expect("outside fixture must remain readable"),
            "outside"
        );
        assert_eq!(
            fs::read_to_string(root.join("link.txt")).expect("replacement must be readable"),
            "inside"
        );
        assert!(
            !fs::symlink_metadata(root.join("link.txt"))
                .expect("replacement metadata must exist")
                .file_type()
                .is_symlink()
        );
        fs::remove_dir_all(root).expect("temporary workspace must be removed");
        fs::remove_file(outside).expect("outside fixture must be removed");
    }

    #[test]
    fn declares_grep_as_a_read_only_search_tool() {
        let grep = tools()
            .into_iter()
            .find(|tool| tool.name == "grep")
            .expect("grep tool must be declared");
        assert_eq!(grep.input_schema["required"], json!(["pattern"]));
        assert_eq!(
            grep.input_schema["properties"]["caseSensitive"]["default"],
            true
        );
    }

    #[tokio::test]
    async fn grep_searches_text_files_and_respects_ignore_rules() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock must be after the Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!("mycode-plugin-grep-{nonce}"));
        fs::create_dir_all(root.join("src")).expect("temporary workspace must be created");
        fs::write(root.join(".gitignore"), "ignored.txt\n").expect("ignore fixture must write");
        fs::write(root.join("src/note.txt"), "Alpha\nbeta\nALPHA\n")
            .expect("search fixture must write");
        fs::write(root.join("ignored.txt"), "alpha\n").expect("ignored fixture must write");
        let workspace = root.canonicalize().expect("workspace must canonicalize");

        let result = execute_tool(
            &workspace,
            CallToolParams {
                name: "grep".to_owned(),
                arguments: json!({
                    "pattern": "^alpha$",
                    "caseSensitive": false,
                    "maxResults": 10
                }),
            },
        )
        .await
        .expect("grep search must succeed");
        assert_eq!(result.output, "src/note.txt:1:Alpha\nsrc/note.txt:3:ALPHA");
        assert!(!result.truncated);

        let limited = execute_tool(
            &workspace,
            CallToolParams {
                name: "grep".to_owned(),
                arguments: json!({
                    "pattern": "^alpha$",
                    "caseSensitive": false,
                    "maxResults": 1
                }),
            },
        )
        .await
        .expect("limited grep search must succeed");
        assert_eq!(limited.output, "src/note.txt:1:Alpha");
        assert!(limited.truncated);
        fs::remove_dir_all(root).expect("temporary workspace must be removed");
    }

    #[tokio::test]
    async fn grep_rejects_invalid_regular_expressions() {
        let workspace = env::current_dir().expect("current directory must exist");
        let result = execute_tool(
            &workspace,
            CallToolParams {
                name: "grep".to_owned(),
                arguments: json!({"pattern": "["}),
            },
        )
        .await;
        assert!(matches!(result, Err(WorkspaceError::InvalidGrepPattern(_))));
    }
}
