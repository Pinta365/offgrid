//! Measures per-turn overhead with large media in the history.
//! cargo run --release -p offgrid --example media_turns -- model.gguf mmproj.gguf file...
use std::time::Instant;

use offgrid::*;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let params = ModelParams { mmproj: Some(args[1].clone().into()), ..Default::default() };
    let model = Model::load(&args[0], &params)?;
    let media: Vec<Media> = args[2..].iter().map(|p| Media::from_path(p).unwrap()).collect();
    let mut chat = ChatSession::new(model, &SessionParams { n_ctx: 32768, ..Default::default() })?;
    let mut msgs = vec![ChatMessage::user_with_media("Describe this briefly.", media)];
    for turn in 0..4 {
        let req = ChatRequest {
            max_tokens: Some(1),
            sampling: SamplingParams { temperature: 0.0, ..Default::default() },
            ..ChatRequest::new(msgs.clone())
        };
        let t = Instant::now();
        let r = chat.complete(&req, |_| true)?;
        let total = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "turn {turn}: total {total:7.1} ms | prompt eval {:7.1} ms | other (tokenize/decode media) {:7.1} ms | cached {}/{}",
            r.usage.prompt_ms, total - r.usage.prompt_ms - r.usage.generation_ms, r.usage.cached_tokens, r.usage.prompt_tokens
        );
        msgs.push(r.to_message());
        msgs.push(ChatMessage::user(format!("Follow-up question {turn}.")));
    }
    Ok(())
}
