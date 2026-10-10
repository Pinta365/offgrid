//! Interactive streaming chat.
//!
//! cargo run --release -p offgrid --example chat -- path/to/model.gguf [system prompt]
//! Pass a prompt on stdin non-interactively: echo "Hi" | cargo run ... -- model.gguf

use std::io::{BufRead, Write};

use offgrid::{ChatMessage, ChatRequest, ChatSession, Model, ModelParams, SessionParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: chat <model.gguf> [system prompt]")?;
    let system = args.next().unwrap_or_else(|| "You are a helpful assistant.".into());

    for d in offgrid::devices() {
        eprintln!("device: {} ({:?}) {}", d.name, d.kind, d.description);
    }
    let model = Model::load(&path, &ModelParams::default())?;
    let info = model.info();
    eprintln!(
        "model: {} [{}], {:.1}B params",
        info.name.as_deref().unwrap_or("?"),
        info.description,
        info.n_params as f64 / 1e9
    );

    let mut chat = ChatSession::new(model, &SessionParams::default())?;
    let mut messages = vec![ChatMessage::system(system)];

    let stdin = std::io::stdin();
    loop {
        eprint!("\n> ");
        std::io::stderr().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        messages.push(ChatMessage::user(line));
        let resp = chat.complete(&ChatRequest::new(messages.clone()), |piece| {
            print!("{piece}");
            std::io::stdout().flush().is_ok()
        })?;
        let u = &resp.usage;
        eprintln!(
            "\n[{:?} | prompt {} tok ({} cached) {:.0} ms | gen {} tok {:.1} tok/s]",
            resp.finish_reason,
            u.prompt_tokens,
            u.cached_tokens,
            u.prompt_ms,
            u.completion_tokens,
            u.completion_tokens as f64 / (u.generation_ms / 1000.0).max(1e-9),
        );
        messages.push(ChatMessage::assistant(resp.content));
    }
    Ok(())
}
