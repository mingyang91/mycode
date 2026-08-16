use std::{
    env,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
};

fn main() -> std::io::Result<()> {
    let port = env::args().nth(1).unwrap_or_else(|| "18771".to_owned());
    let high_usage = env::args().nth(2).as_deref() == Some("high");
    let listener = TcpListener::bind(format!("127.0.0.1:{port}"))?;
    eprintln!("mock-compaction-openai ready on {port}");
    for response in [
        turn_response("First compact smoke answer.", 100),
        turn_response("Second compact smoke answer.", 200),
        turn_response(
            "Third compact smoke answer.",
            if high_usage { 900 } else { 300 },
        ),
        turn_response(
            "Summary: three smoke turns established the compaction contract.",
            50,
        ),
    ] {
        loop {
            let (stream, _) = listener.accept()?;
            match respond(stream, &response) {
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

fn respond(mut stream: TcpStream, body: &str) -> std::io::Result<()> {
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
    eprintln!(
        "mock-compaction-openai received {} request bytes",
        request_body.len()
    );
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()
}

fn turn_response(content: &str, prompt_tokens: u64) -> String {
    serde_json::json!({
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 10}
    })
    .to_string()
}
