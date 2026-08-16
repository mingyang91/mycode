use std::{
    env,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
};

fn main() -> std::io::Result<()> {
    let port = env::args().nth(1).unwrap_or_else(|| "18770".to_owned());
    let listener = TcpListener::bind(format!("127.0.0.1:{port}"))?;
    eprintln!("mock-plan-openai ready on {port}");
    for response in [plan_response(), execution_response()] {
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
    eprintln!(
        "mock-plan-openai received {} request bytes",
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

fn plan_response() -> &'static str {
    r##"{"choices":[{"message":{"content":"I inspected the request and prepared a bounded plan.","tool_calls":[{"id":"call_mock_todo","type":"function","function":{"name":"todo","arguments":"{\"op\":\"init\",\"phases\":[{\"name\":\"Delivery\",\"tasks\":[{\"content\":\"Inspect constraints\",\"status\":\"completed\"},{\"content\":\"Implement approved plan\",\"status\":\"pending\"}]}]}"}},{"id":"call_mock_plan","type":"function","function":{"name":"plan","arguments":"{\"op\":\"propose\",\"content\":\"# Bounded storage plan\\n\\n1. Inspect constraints.\\n2. Implement in bounded steps.\\n3. Verify the result.\"}"}}]}}]}"##
}

fn execution_response() -> &'static str {
    r#"{"choices":[{"message":{"content":"Mock executed the approved plan."}}]}"#
}
