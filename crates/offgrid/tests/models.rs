//! Integration tests against real GGUF files.
//! Set OFFGRID_TEST_MODEL (chat model) and/or OFFGRID_TEST_EMBED_MODEL; tests skip otherwise.

use std::sync::Arc;

use offgrid::*;

fn env_model(var: &str) -> Option<Arc<Model>> {
    let Ok(path) = std::env::var(var) else {
        eprintln!("skip: {var} not set");
        return None;
    };
    Some(Model::load(path, &ModelParams::default()).expect("model loads"))
}

fn greedy(messages: Vec<ChatMessage>, max_tokens: u32) -> ChatRequest {
    ChatRequest {
        max_tokens: Some(max_tokens),
        sampling: SamplingParams { temperature: 0.0, ..Default::default() },
        ..ChatRequest::new(messages)
    }
}

#[test]
fn lists_devices() {
    let devs = devices();
    println!("{devs:#?}");
    assert!(!devs.is_empty());
}

#[test]
fn chat_streams_and_reuses_prefix() {
    let Some(model) = env_model("OFFGRID_TEST_MODEL") else { return };
    println!("{:#?}", model.info());
    let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();

    let mut msgs = vec![
        ChatMessage::system("You are terse."),
        ChatMessage::user("What is the capital of France? One word."),
    ];
    let mut streamed = String::new();
    let r1 = chat.complete(&greedy(msgs.clone(), 16), |p| { streamed.push_str(p); true }).unwrap();
    println!("r1: {r1:?}");
    assert_eq!(streamed, r1.content);
    assert!(r1.content.contains("Paris"));
    assert_eq!(r1.finish_reason, FinishReason::Stop);

    msgs.push(ChatMessage::assistant(r1.content.clone()));
    msgs.push(ChatMessage::user("And of Germany?"));
    let r2 = chat.complete(&greedy(msgs, 16), |_| true).unwrap();
    println!("r2: {r2:?}");
    assert!(r2.content.contains("Berlin"));
    assert!(r2.usage.cached_tokens > 10, "second turn should reuse the KV prefix");
}

#[test]
fn chat_limits_stop_and_cancel() {
    let Some(model) = env_model("OFFGRID_TEST_MODEL") else { return };
    let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
    let count = || vec![ChatMessage::user("Count from 1 to 30, comma separated.")];

    let r = chat.complete(&greedy(count(), 5), |_| true).unwrap();
    assert_eq!(r.finish_reason, FinishReason::Length);
    assert_eq!(r.usage.completion_tokens, 5);

    let mut req = greedy(count(), 200);
    req.stop = vec!["5".into()];
    let r = chat.complete(&req, |_| true).unwrap();
    assert_eq!(r.finish_reason, FinishReason::Stop);
    assert!(!r.content.contains('5'), "{:?}", r.content);

    let mut n = 0;
    let r = chat.complete(&greedy(count(), 200), |_| { n += 1; n < 3 }).unwrap();
    assert_eq!(r.finish_reason, FinishReason::Cancelled);

    // A cancel while idle (e.g. one that arrives just after a request finished) must not
    // leak into the next request.
    let token = chat.cancel_token();
    token.cancel();
    let r = chat.complete(&greedy(count(), 5), |_| true).unwrap();
    assert_eq!(r.finish_reason, FinishReason::Length, "stale cancel leaked into the next request");

    // Cancelling the next request specifically works, even before it starts...
    let next = token.next_request();
    next.cancel();
    let r = chat.complete(&greedy(count(), 200), |_| true).unwrap();
    assert_eq!(r.finish_reason, FinishReason::Cancelled);
    // ...and only that one: a late cancel of a finished request is a no-op.
    next.cancel();
    let r = chat.complete(&greedy(count(), 5), |_| true).unwrap();
    assert_eq!(r.finish_reason, FinishReason::Length);

    // Cancel from another thread while generating.
    let t = token.clone();
    let canceller = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(150));
        t.cancel();
    });
    let r = chat.complete(&greedy(count(), 2000), |_| true).unwrap();
    canceller.join().unwrap();
    assert_eq!(r.finish_reason, FinishReason::Cancelled);
}

#[test]
fn chat_grammar() {
    let Some(model) = env_model("OFFGRID_TEST_MODEL") else { return };
    let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
    let mut req = greedy(vec![ChatMessage::user("Is the sky blue? Answer yes or no.")], 8);
    req.sampling.grammar = Some(r#"root ::= "YES" | "NO""#.into());
    let r = chat.complete(&req, |_| true).unwrap();
    assert!(r.content == "YES" || r.content == "NO", "{:?}", r.content);
}

#[test]
fn embeddings_rank_similarity() {
    let Some(model) = env_model("OFFGRID_TEST_EMBED_MODEL") else { return };
    let mut emb = Embedder::new(model, &EmbedParams::default()).unwrap();
    let v = emb
        .embed(&["The cat sits on the mat.", "A kitten is resting on a rug.", "Quarterly tax filing deadlines"])
        .unwrap();
    assert_eq!(v.len(), 3);
    assert_eq!(v[0].len(), emb.dim());
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let (sim, diff) = (dot(&v[0], &v[1]), dot(&v[0], &v[2]));
    println!("cat~kitten {sim:.3}  cat~tax {diff:.3}");
    assert!(sim > diff + 0.2);
    // Batched and single-input results agree.
    let single = emb.embed_one("The cat sits on the mat.").unwrap();
    assert!(dot(&single, &v[0]) > 0.999);

    // Inputs well beyond n_ctx / max_batch_inputs (the per-sequence KV split) must work.
    let long_doc = "The quick brown fox jumps over the lazy dog. ".repeat(150); // ~1500 tokens
    let long = emb.embed(&[long_doc.as_str(), "short"]).expect("long input embeds");
    assert_eq!(long[0].len(), emb.dim());
}
