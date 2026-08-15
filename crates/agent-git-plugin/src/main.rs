#![deny(warnings)]
#![deny(clippy::unwrap_used)]

use std::{
    env,
    path::{Component, Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use agent_plugin_protocol::{
    CallToolParams, DEFAULT_MAX_FRAME_BYTES, DEFAULT_MAX_TOOL_OUTPUT_BYTES, EmptyParams,
    InitializeResult, PluginErrorCode, PluginIdentity, ProtocolRange, RequestEnvelope,
    RequestOperation, ResponseEnvelope, ToolListResult, ToolResult, ToolSpec, read_request,
    tool_failure, write_response,
};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncReadExt, stdin, stdout},
    process::Command,
    time::timeout,
};

const GIT_TIMEOUT: Duration = Duration::from_secs(120);
const REMOVED_GIT_ENVIRONMENT: &[&str] = &[
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_DIR",
    "GIT_EXEC_PATH",
    "GIT_EXTERNAL_DIFF",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_WORK_TREE",
];

#[derive(Debug, Error)]
enum GitError {
    #[error("argument {name} must be a string")]
    MissingStringArgument { name: &'static str },
    #[error("argument arguments must be an array of strings")]
    InvalidArguments,
    #[error("operation {operation} is not valid for tool {tool}")]
    InvalidOperation { tool: String, operation: String },
    #[error("Git path must be a non-empty relative path inside the repository")]
    InvalidPath,
    #[error("Git command contains an option outside the verified subset")]
    ForbiddenOption,
    #[error("the configured workspace must be the root of a Git repository")]
    NotRepositoryRoot,
    #[error("Git command exceeded its deadline")]
    CommandTimedOut,
    #[error("Git output exceeded {limit} bytes")]
    OutputTooLarge { limit: usize },
    #[error("Git process did not provide piped output")]
    MissingChildPipe,
    #[error("filesystem or Git process operation failed: {0}")]
    Io(#[from] std::io::Error),
}

struct GitContext {
    workspace: PathBuf,
    available: bool,
}

impl GitContext {
    async fn discover(workspace: &Path) -> Result<Self, GitError> {
        let workspace = workspace.canonicalize()?;
        let available = match Command::new("git")
            .args(["--no-optional-locks", "rev-parse", "--show-toplevel"])
            .current_dir(&workspace)
            .stdin(Stdio::null())
            .output()
            .await
        {
            Ok(output) if output.status.success() => {
                let root = String::from_utf8_lossy(&output.stdout).trim().to_owned();
                PathBuf::from(root)
                    .canonicalize()
                    .is_ok_and(|root| root == workspace)
            }
            Ok(_) | Err(_) => false,
        };
        Ok(Self {
            workspace,
            available,
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), GitError> {
    let workspace = env::current_dir()?;
    let context = GitContext::discover(&workspace).await?;
    let mut input = stdin();
    let mut output = stdout();
    let mut initialized = false;

    loop {
        let request = match read_request(&mut input, DEFAULT_MAX_FRAME_BYTES).await {
            Ok(request) => request,
            Err(error) => {
                eprintln!("Git plugin protocol error: {error}");
                return Ok(());
            }
        };
        let was_initialize = matches!(&request.operation, RequestOperation::Initialize(_));
        let (response, shutdown) = handle_request(request, &context, initialized).await;
        if was_initialize && response.ok {
            initialized = true;
        }
        if let Err(error) = write_response(&mut output, &response).await {
            eprintln!("Git plugin response write failed: {error}");
            return Ok(());
        }
        if shutdown {
            return Ok(());
        }
    }
}

async fn handle_request(
    request: RequestEnvelope,
    context: &GitContext,
    initialized: bool,
) -> (ResponseEnvelope, bool) {
    let request_id = request.id.clone();
    match request.operation {
        RequestOperation::Initialize(_) if !initialized => success(
            request_id,
            &InitializeResult {
                plugin: PluginIdentity {
                    name: "git".to_owned(),
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
        RequestOperation::ListTools(_) if initialized => success(
            request_id,
            &ToolListResult {
                tools: tools(context),
            },
        ),
        RequestOperation::ListTools(_) => failure(
            request_id,
            PluginErrorCode::InvalidRequest,
            "initialize must complete before tool discovery",
        ),
        RequestOperation::CallTool(call) if initialized => {
            match execute_tool(context, call).await {
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

fn tools(context: &GitContext) -> Vec<ToolSpec> {
    if !context.available {
        return Vec::new();
    }
    vec![
        ToolSpec {
            name: "git_read".to_owned(),
            description: "Internal verified read-only Git effect.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["status", "diff", "log", "show", "rev_parse", "branch_current"]
                    },
                    "arguments": {"type": "array", "items": {"type": "string"}},
                    "command": {"type": "string"}
                },
                "required": ["operation", "arguments"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "git_write".to_owned(),
            description: "Internal verified mutating Git effect. Requires approval.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "restore_staged", "commit"]
                    },
                    "arguments": {"type": "array", "items": {"type": "string"}},
                    "command": {"type": "string"}
                },
                "required": ["operation", "arguments"],
                "additionalProperties": false
            }),
        },
    ]
}

async fn execute_tool(context: &GitContext, call: CallToolParams) -> Result<ToolResult, GitError> {
    if !context.available {
        return Err(GitError::NotRepositoryRoot);
    }
    let operation = required_string(&call.arguments, "operation")?;
    let arguments = required_arguments(&call.arguments)?;
    let argv = verified_argv(&call.name, operation, &arguments)?;
    run_git(context, &argv).await
}

fn verified_argv(
    tool: &str,
    operation: &str,
    arguments: &[String],
) -> Result<Vec<String>, GitError> {
    let mut argv = Vec::new();
    match (tool, operation) {
        ("git_read", "status") => {
            validate_read_arguments(arguments)?;
            argv.push("status".to_owned());
            argv.extend(arguments.iter().cloned());
        }
        ("git_read", "diff") => {
            validate_read_arguments(arguments)?;
            argv.extend([
                "diff".to_owned(),
                "--no-ext-diff".to_owned(),
                "--no-textconv".to_owned(),
            ]);
            argv.extend(arguments.iter().cloned());
        }
        ("git_read", "log") => {
            validate_read_arguments(arguments)?;
            argv.extend([
                "log".to_owned(),
                "--no-ext-diff".to_owned(),
                "--no-textconv".to_owned(),
            ]);
            argv.extend(arguments.iter().cloned());
        }
        ("git_read", "show") => {
            validate_read_arguments(arguments)?;
            argv.extend([
                "show".to_owned(),
                "--no-ext-diff".to_owned(),
                "--no-textconv".to_owned(),
            ]);
            argv.extend(arguments.iter().cloned());
        }
        ("git_read", "rev_parse") => {
            validate_read_arguments(arguments)?;
            argv.push("rev-parse".to_owned());
            argv.extend(arguments.iter().cloned());
        }
        ("git_read", "branch_current") if arguments.is_empty() => {
            argv.extend(["branch".to_owned(), "--show-current".to_owned()]);
        }
        ("git_write", "add") => {
            validate_paths(arguments)?;
            argv.extend(["add".to_owned(), "--".to_owned()]);
            argv.extend(arguments.iter().cloned());
        }
        ("git_write", "restore_staged") => {
            validate_paths(arguments)?;
            argv.extend(["restore".to_owned(), "--staged".to_owned(), "--".to_owned()]);
            argv.extend(arguments.iter().cloned());
        }
        ("git_write", "commit") if arguments.len() == 1 && !arguments[0].is_empty() => {
            argv.extend([
                "commit".to_owned(),
                "--no-verify".to_owned(),
                "-m".to_owned(),
                arguments[0].clone(),
            ]);
        }
        _ => {
            return Err(GitError::InvalidOperation {
                tool: tool.to_owned(),
                operation: operation.to_owned(),
            });
        }
    }
    Ok(argv)
}

fn validate_read_arguments(arguments: &[String]) -> Result<(), GitError> {
    if arguments.iter().any(|argument| {
        argument.contains('\0')
            || argument == "-c"
            || argument == "-C"
            || argument == "--help"
            || argument == "-h"
            || argument == "--ext-diff"
            || argument == "--textconv"
            || argument == "--no-index"
            || argument.starts_with("--output")
            || argument.starts_with("--git-dir")
            || argument.starts_with("--work-tree")
            || argument.starts_with("--exec-path")
            || argument.starts_with("--config-env")
    }) {
        Err(GitError::ForbiddenOption)
    } else {
        Ok(())
    }
}

fn validate_paths(paths: &[String]) -> Result<(), GitError> {
    if paths.is_empty() || paths.iter().any(|path| !valid_path(path)) {
        Err(GitError::InvalidPath)
    } else {
        Ok(())
    }
}

fn valid_path(raw: &str) -> bool {
    let path = Path::new(raw);
    !raw.is_empty()
        && !raw.starts_with('-')
        && !path.is_absolute()
        && !path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            ) || matches!(component, Component::Normal(name) if name == ".git")
        })
}

async fn run_git(context: &GitContext, argv: &[String]) -> Result<ToolResult, GitError> {
    let mut command = Command::new("git");
    command
        .arg("--no-optional-locks")
        .args(["-c", "core.hooksPath=/dev/null"])
        .args(argv)
        .current_dir(&context.workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_PAGER", "cat")
        .env("PAGER", "cat");
    for variable in REMOVED_GIT_ENVIRONMENT {
        command.env_remove(variable);
    }
    let mut child = command.spawn()?;
    let stdout = child.stdout.take().ok_or(GitError::MissingChildPipe)?;
    let stderr = child.stderr.take().ok_or(GitError::MissingChildPipe)?;
    let completed = timeout(GIT_TIMEOUT, async {
        let (stdout, stderr, status) = tokio::try_join!(
            read_bounded(stdout, DEFAULT_MAX_TOOL_OUTPUT_BYTES),
            read_bounded(stderr, DEFAULT_MAX_TOOL_OUTPUT_BYTES),
            async { child.wait().await.map_err(GitError::from) }
        )?;
        Ok::<_, GitError>((stdout, stderr, status))
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
            return Err(GitError::CommandTimedOut);
        }
    };
    let mut output = String::from_utf8_lossy(&stdout).into_owned();
    let stderr = String::from_utf8_lossy(&stderr);
    if !stderr.is_empty() {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str("[stderr]\n");
        output.push_str(&stderr);
    }
    if !status.success() {
        output.push_str(&format!("\n[exit status: {status}]"));
    }
    Ok(ToolResult {
        output,
        truncated: false,
    })
}

async fn read_bounded<R>(mut reader: R, limit: usize) -> Result<Vec<u8>, GitError>
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
            return Err(GitError::OutputTooLarge { limit });
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
}

fn required_string<'a>(arguments: &'a Value, name: &'static str) -> Result<&'a str, GitError> {
    arguments
        .get(name)
        .and_then(Value::as_str)
        .ok_or(GitError::MissingStringArgument { name })
}

fn required_arguments(arguments: &Value) -> Result<Vec<String>, GitError> {
    arguments
        .get("arguments")
        .and_then(Value::as_array)
        .ok_or(GitError::InvalidArguments)?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or(GitError::InvalidArguments)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{
        env, fs,
        process::Command as StdCommand,
        time::{SystemTime, UNIX_EPOCH},
    };

    use agent_plugin_protocol::CallToolParams;
    use serde_json::json;

    use super::{GitContext, GitError, execute_tool, tools, verified_argv};

    fn repository() -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock must be after the Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!("lean-agent-git-plugin-{nonce}"));
        fs::create_dir(&root).expect("repository directory must be created");
        let status = StdCommand::new("git")
            .args(["init", "-q"])
            .current_dir(&root)
            .status()
            .expect("git init must run");
        assert!(status.success());
        root
    }

    #[tokio::test]
    async fn advertises_no_tools_outside_a_repository_root() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock must be after the Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!("lean-agent-no-git-{nonce}"));
        fs::create_dir(&root).expect("plain directory must be created");
        let context = GitContext::discover(&root)
            .await
            .expect("plain directory discovery must not fail");
        assert!(!context.available);
        assert!(tools(&context).is_empty());
        fs::remove_dir_all(root).expect("fixture must be removed");
    }

    #[test]
    fn rejects_read_options_with_side_effects() {
        let result = verified_argv("git_read", "diff", &["--output=leak.patch".to_owned()]);
        assert!(matches!(result, Err(GitError::ForbiddenOption)));
    }

    #[test]
    fn rejects_paths_outside_the_repository() {
        let result = verified_argv("git_write", "add", &["../outside".to_owned()]);
        assert!(matches!(result, Err(GitError::InvalidPath)));
    }

    #[tokio::test]
    async fn reads_status_and_stages_a_scoped_path() {
        let root = repository();
        fs::write(root.join("note.txt"), "content").expect("fixture must be written");
        let context = GitContext::discover(&root)
            .await
            .expect("repository root must be discovered");
        let status = execute_tool(
            &context,
            CallToolParams {
                name: "git_read".to_owned(),
                arguments: json!({"operation": "status", "arguments": ["--short"]}),
            },
        )
        .await
        .expect("status must run");
        assert!(status.output.contains("?? note.txt"));
        execute_tool(
            &context,
            CallToolParams {
                name: "git_write".to_owned(),
                arguments: json!({"operation": "add", "arguments": ["note.txt"]}),
            },
        )
        .await
        .expect("add must run");
        let staged = StdCommand::new("git")
            .args(["status", "--short"])
            .current_dir(&root)
            .output()
            .expect("status must run");
        assert!(String::from_utf8_lossy(&staged.stdout).contains("A  note.txt"));
        fs::remove_dir_all(root).expect("fixture must be removed");
    }
    #[tokio::test]
    async fn commits_the_staged_index_with_a_fixed_message() {
        let root = repository();
        for (key, value) in [
            ("user.name", "Lean Agent"),
            ("user.email", "agent@example.invalid"),
        ] {
            let status = StdCommand::new("git")
                .args(["config", key, value])
                .current_dir(&root)
                .status()
                .expect("git config must run");
            assert!(status.success());
        }
        fs::write(root.join("note.txt"), "content").expect("fixture must be written");
        let context = GitContext::discover(&root)
            .await
            .expect("repository root must be discovered");
        execute_tool(
            &context,
            CallToolParams {
                name: "git_write".to_owned(),
                arguments: json!({"operation": "add", "arguments": ["note.txt"]}),
            },
        )
        .await
        .expect("add must run");
        execute_tool(
            &context,
            CallToolParams {
                name: "git_write".to_owned(),
                arguments: json!({
                    "operation": "commit",
                    "arguments": ["verified commit"],
                    "command": "git commit -m \"verified commit\""
                }),
            },
        )
        .await
        .expect("commit must run");
        let subject = StdCommand::new("git")
            .args(["log", "-1", "--pretty=%s"])
            .current_dir(&root)
            .output()
            .expect("git log must run");
        assert_eq!(
            String::from_utf8_lossy(&subject.stdout).trim(),
            "verified commit"
        );
        fs::remove_dir_all(root).expect("fixture must be removed");
    }
}
