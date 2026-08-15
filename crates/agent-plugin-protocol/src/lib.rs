#![deny(warnings)]
#![deny(clippy::unwrap_used)]

use std::fmt::Display;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const PROTOCOL_VERSION: u16 = 1;
pub const DEFAULT_MAX_FRAME_BYTES: usize = 1_048_576;
pub const DEFAULT_MAX_TOOL_OUTPUT_BYTES: usize = 262_144;
pub const DEFAULT_MAX_ERROR_MESSAGE_BYTES: usize = 4_096;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    #[serde(rename = "maxFrameBytes")]
    pub max_frame_bytes: usize,
    #[serde(rename = "maxToolOutputBytes")]
    pub max_tool_output_bytes: usize,
    #[serde(rename = "maxErrorMessageBytes")]
    pub max_error_message_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            max_tool_output_bytes: DEFAULT_MAX_TOOL_OUTPUT_BYTES,
            max_error_message_bytes: DEFAULT_MAX_ERROR_MESSAGE_BYTES,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostIdentity {
    pub name: String,
    pub version: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InitializeParams {
    pub host: HostIdentity,
    pub limits: Limits,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EmptyParams {}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CallToolParams {
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "op", content = "params", rename_all = "snake_case")]
pub enum RequestOperation {
    Initialize(InitializeParams),
    ListTools(EmptyParams),
    CallTool(CallToolParams),
    Shutdown(EmptyParams),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequestEnvelope {
    pub v: u16,
    pub id: String,
    #[serde(flatten)]
    pub operation: RequestOperation,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PluginIdentity {
    pub name: String,
    pub version: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolRange {
    #[serde(rename = "minVersion")]
    pub min_version: u16,
    #[serde(rename = "maxVersion")]
    pub max_version: u16,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InitializeResult {
    pub plugin: PluginIdentity,
    pub protocol: ProtocolRange,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolListResult {
    pub tools: Vec<ToolSpec>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolResult {
    pub output: String,
    pub truncated: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginErrorCode {
    InvalidRequest,
    UnsupportedOperation,
    UnknownTool,
    InvalidArguments,
    ToolFailed,
    Internal,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PluginFailure {
    pub code: PluginErrorCode,
    pub message: String,
    pub retryable: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseEnvelope {
    pub v: u16,
    pub id: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<PluginFailure>,
}

impl ResponseEnvelope {
    pub fn success<T>(id: String, result: &T) -> Result<Self, ProtocolError>
    where
        T: Serialize,
    {
        Ok(Self {
            v: PROTOCOL_VERSION,
            id,
            ok: true,
            result: Some(serde_json::to_value(result)?),
            error: None,
        })
    }

    pub fn failure(id: String, failure: PluginFailure) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id,
            ok: false,
            result: None,
            error: Some(failure),
        }
    }

    pub fn into_result<T>(self) -> Result<T, ProtocolError>
    where
        T: for<'de> Deserialize<'de>,
    {
        validate_response(&self)?;
        if !self.ok {
            return match self.error {
                Some(error) => Err(ProtocolError::PluginFailure(error)),
                None => Err(ProtocolError::Violation(ProtocolViolation::MissingError)),
            };
        }
        match self.result {
            Some(value) => Ok(serde_json::from_value(value)?),
            None => Err(ProtocolError::Violation(ProtocolViolation::MissingResult)),
        }
    }
}

#[derive(Clone, Debug, Error)]
pub enum ProtocolViolation {
    #[error("unsupported protocol version {found}")]
    UnsupportedVersion { found: u16 },
    #[error("empty correlation id")]
    EmptyCorrelationId,
    #[error("correlation id exceeds 64 bytes")]
    CorrelationIdTooLong,
    #[error("correlation id contains unsupported characters")]
    InvalidCorrelationId,
    #[error("request or response carried an invalid shape")]
    InvalidEnvelope,
    #[error("successful response omitted its result")]
    MissingResult,
    #[error("failed response omitted its error")]
    MissingError,
    #[error("successful response included an error")]
    SuccessIncludedError,
    #[error("failed response included a result")]
    FailureIncludedResult,
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("plugin transport I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("plugin frame declared {declared} bytes, maximum is {maximum}")]
    FrameTooLarge { declared: usize, maximum: usize },
    #[error("plugin frame payload was not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("plugin protocol violation: {0}")]
    Violation(#[from] ProtocolViolation),
    #[error("plugin returned {0:?}")]
    PluginFailure(PluginFailure),
}

pub fn validate_request(request: &RequestEnvelope) -> Result<(), ProtocolError> {
    validate_version(request.v)?;
    validate_correlation_id(&request.id)?;
    Ok(())
}

pub fn validate_response(response: &ResponseEnvelope) -> Result<(), ProtocolError> {
    validate_version(response.v)?;
    validate_correlation_id(&response.id)?;
    match (
        response.ok,
        response.result.is_some(),
        response.error.is_some(),
    ) {
        (true, true, false) | (false, false, true) => Ok(()),
        (true, false, _) => Err(ProtocolViolation::MissingResult.into()),
        (true, _, true) => Err(ProtocolViolation::SuccessIncludedError.into()),
        (false, true, _) => Err(ProtocolViolation::FailureIncludedResult.into()),
        (false, false, false) => Err(ProtocolViolation::MissingError.into()),
    }
}

pub async fn read_request<R>(
    reader: &mut R,
    max_frame_bytes: usize,
) -> Result<RequestEnvelope, ProtocolError>
where
    R: AsyncRead + Unpin,
{
    let request: RequestEnvelope = read_frame(reader, max_frame_bytes).await?;
    validate_request(&request)?;
    Ok(request)
}

pub async fn read_response<R>(
    reader: &mut R,
    max_frame_bytes: usize,
) -> Result<ResponseEnvelope, ProtocolError>
where
    R: AsyncRead + Unpin,
{
    let response: ResponseEnvelope = read_frame(reader, max_frame_bytes).await?;
    validate_response(&response)?;
    Ok(response)
}

pub async fn write_request<W>(
    writer: &mut W,
    request: &RequestEnvelope,
) -> Result<(), ProtocolError>
where
    W: AsyncWrite + Unpin,
{
    validate_request(request)?;
    write_frame(writer, request).await
}

pub async fn write_response<W>(
    writer: &mut W,
    response: &ResponseEnvelope,
) -> Result<(), ProtocolError>
where
    W: AsyncWrite + Unpin,
{
    validate_response(response)?;
    write_frame(writer, response).await
}

async fn read_frame<R, T>(reader: &mut R, max_frame_bytes: usize) -> Result<T, ProtocolError>
where
    R: AsyncRead + Unpin,
    T: for<'de> Deserialize<'de>,
{
    let mut header = [0_u8; 4];
    reader.read_exact(&mut header).await?;
    let declared = u32::from_be_bytes(header) as usize;
    if declared == 0 || declared > max_frame_bytes {
        return Err(ProtocolError::FrameTooLarge {
            declared,
            maximum: max_frame_bytes,
        });
    }
    let mut payload = vec![0_u8; declared];
    reader.read_exact(&mut payload).await?;
    Ok(serde_json::from_slice(&payload)?)
}

async fn write_frame<W, T>(writer: &mut W, value: &T) -> Result<(), ProtocolError>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let payload = serde_json::to_vec(value)?;
    if payload.is_empty() || payload.len() > DEFAULT_MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge {
            declared: payload.len(),
            maximum: DEFAULT_MAX_FRAME_BYTES,
        });
    }
    let length = u32::try_from(payload.len()).map_err(|_| ProtocolError::FrameTooLarge {
        declared: payload.len(),
        maximum: u32::MAX as usize,
    })?;
    writer.write_all(&length.to_be_bytes()).await?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}

fn validate_version(version: u16) -> Result<(), ProtocolError> {
    if version == PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(ProtocolViolation::UnsupportedVersion { found: version }.into())
    }
}

fn validate_correlation_id(id: &str) -> Result<(), ProtocolError> {
    if id.is_empty() {
        return Err(ProtocolViolation::EmptyCorrelationId.into());
    }
    if id.len() > 64 {
        return Err(ProtocolViolation::CorrelationIdTooLong.into());
    }
    if id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        Ok(())
    } else {
        Err(ProtocolViolation::InvalidCorrelationId.into())
    }
}

pub fn tool_failure(
    code: PluginErrorCode,
    message: impl Display,
    retryable: bool,
) -> PluginFailure {
    let mut text = message.to_string();
    if text.len() > DEFAULT_MAX_ERROR_MESSAGE_BYTES {
        let mut boundary = DEFAULT_MAX_ERROR_MESSAGE_BYTES;
        while !text.is_char_boundary(boundary) {
            boundary -= 1;
        }
        text.truncate(boundary);
    }
    PluginFailure {
        code,
        message: text,
        retryable,
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::AsyncWriteExt;
    use tokio::io::duplex;

    use super::{
        CallToolParams, DEFAULT_MAX_FRAME_BYTES, EmptyParams, PROTOCOL_VERSION, ProtocolError,
        RequestEnvelope, RequestOperation, read_request, write_request,
    };

    #[tokio::test]
    async fn round_trips_a_tool_request() {
        let (mut writer, mut reader) = duplex(4096);
        let request = RequestEnvelope {
            v: PROTOCOL_VERSION,
            id: "call_1".to_owned(),
            operation: RequestOperation::CallTool(CallToolParams {
                name: "read".to_owned(),
                arguments: serde_json::json!({"path": "Main.lean"}),
            }),
        };
        write_request(&mut writer, &request)
            .await
            .expect("request must serialize");
        let decoded = read_request(&mut reader, DEFAULT_MAX_FRAME_BYTES)
            .await
            .expect("request must deserialize");
        assert_eq!(decoded.id, request.id);
    }

    #[tokio::test]
    async fn rejects_an_oversized_frame_before_reading_body() {
        let (mut writer, mut reader) = duplex(32);
        writer
            .write_all(&(1024_u32.to_be_bytes()))
            .await
            .expect("header must write");
        let error = read_request(&mut reader, 32)
            .await
            .expect_err("oversized frame must fail");
        assert!(matches!(error, ProtocolError::FrameTooLarge { .. }));
    }

    #[test]
    fn serializes_shutdown_with_an_empty_object() {
        let request = RequestEnvelope {
            v: PROTOCOL_VERSION,
            id: "shutdown_1".to_owned(),
            operation: RequestOperation::Shutdown(EmptyParams {}),
        };
        let json = serde_json::to_value(request).expect("request must encode");
        assert_eq!(json["params"], serde_json::json!({}));
    }

    #[test]
    fn error_messages_truncate_at_a_utf8_boundary() {
        let message = format!(
            "{}ésuffix",
            "a".repeat(super::DEFAULT_MAX_ERROR_MESSAGE_BYTES - 1)
        );
        let failure = super::tool_failure(super::PluginErrorCode::Internal, message, false);
        assert!(failure.message.len() < super::DEFAULT_MAX_ERROR_MESSAGE_BYTES + 2);
        assert!(std::str::from_utf8(failure.message.as_bytes()).is_ok());
    }
}
