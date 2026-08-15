use std::{
    env,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
};

fn main() -> std::io::Result<()> {
    let port = env::args().nth(1).unwrap_or_else(|| "18765".to_owned());
    let listener = TcpListener::bind(format!("127.0.0.1:{port}"))?;
    eprintln!("mock-openai ready on {port}");
    for response in [first_response(), second_response()] {
        loop {
            let (stream, _) = listener.accept()?;
            match respond(stream, response) {
                Ok(()) => break,
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::UnexpectedEof
                    ) => {}
                Err(error) => return Err(error),
            }
        }
    }
    Ok(())
}

fn respond(mut stream: TcpStream, body: &'static str) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut content_length = 0_usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line == "\n" {
            break;
        }
        let lowercase = line.to_ascii_lowercase();
        if let Some(value) = lowercase.strip_prefix("content-length:") {
            content_length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut request_body = vec![0_u8; content_length];
    reader.read_exact(&mut request_body)?;
    eprintln!("mock-openai received {} request bytes", request_body.len());
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()
}

fn first_response() -> &'static str {
    r#"{"choices":[{"message":{"content":"","tool_calls":[{"id":"call_mock_read","type":"function","function":{"name":"read","arguments":"{\"path\":\"Cargo.toml\"}"}}]}}]}"#
}

fn second_response() -> &'static str {
    r#"{"choices":[{"message":{"content":"Mock provider confirmed the workspace file was read."}}]}"#
}
