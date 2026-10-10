//! HTTP API tests against real models (skipped unless env vars are set):
//!   OFFGRID_TEST_MODEL, OFFGRID_TEST_EMBED_MODEL, OFFGRID_TEST_THINKING_MODEL
//! Each test runs its own server, freed when the test ends; on small GPUs use --test-threads=1.

use std::sync::Arc;
use std::time::{Duration, Instant};

use offgrid::SessionParams;
use offgrid_server::{AppState, ServerConfig, router};
use base64::Engine;
use serde_json::{Value, json};

/// A server on an ephemeral port; shut down and its models freed on drop.
struct TestServer {
    url: String,
    state: Arc<AppState>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Display for TestServer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.url)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.state.shutdown();
        let _ = self.stop.take().map(|s| s.send(()));
        let _ = self.thread.take().map(|t| t.join());
    }
}

fn spawn(config: ServerConfig) -> TestServer {
    let config = ServerConfig { session: SessionParams { n_ctx: 2048, ..config.session }, ..config };
    let state = AppState::load(&config).expect("models load");
    let (tx, rx) = std::sync::mpsc::channel();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serve_state = state.clone();
    let thread = std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(listener.local_addr().unwrap()).unwrap();
            axum::serve(listener, router(serve_state))
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
    });
    let url = format!("http://{}", rx.recv().unwrap());
    TestServer { url, state, stop: Some(stop), thread: Some(thread) }
}

/// Chat + embeddings, 2 sessions, API key "sekret".
fn server() -> Option<TestServer> {
    let model = std::env::var("OFFGRID_TEST_MODEL").ok()?;
    Some(spawn(ServerConfig {
        model: Some(model.into()),
        embedding_model: std::env::var("OFFGRID_TEST_EMBED_MODEL").ok().map(Into::into),
        alias: Some("test-chat".into()),
        parallel: 2,
        api_key: Some("sekret".into()),
        ..Default::default()
    }))
}

fn client() -> reqwest::Client {
    let mut h = reqwest::header::HeaderMap::new();
    h.insert("authorization", "Bearer sekret".parse().unwrap());
    reqwest::Client::builder().default_headers(h).build().unwrap()
}

async fn post(url: &str, path: &str, body: Value) -> (u16, Value) {
    let r = client().post(format!("{url}{path}")).json(&body).send().await.unwrap();
    let status = r.status().as_u16();
    (status, r.json().await.unwrap())
}

/// Collect an SSE response into its JSON chunks (stops at [DONE]).
async fn sse(url: &str, body: Value) -> Vec<Value> {
    let r = client().post(format!("{url}/v1/chat/completions")).json(&body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
    let text = r.text().await.unwrap();
    let mut chunks = Vec::new();
    let mut done = false;
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data == "[DONE]" {
            done = true;
            break;
        }
        chunks.push(serde_json::from_str(data).unwrap());
    }
    assert!(done, "stream did not end with [DONE]");
    chunks
}

fn greedy(content: &str) -> Value {
    json!({"model": "test-chat", "temperature": 0, "max_tokens": 64, "messages": [{"role": "user", "content": content}]})
}

macro_rules! need_server {
    ($url:ident) => {
        let Some(server) = server() else {
            eprintln!("skip: OFFGRID_TEST_MODEL not set");
            return;
        };
        let $url: &str = &server.url;
    };
}

#[tokio::test]
async fn health_models_auth_and_errors() {
    need_server!(url);
    let r = reqwest::get(format!("{url}/health")).await.unwrap();
    assert_eq!(r.status(), 200);

    // Auth required on /v1.
    let r = reqwest::get(format!("{url}/v1/models")).await.unwrap();
    assert_eq!(r.status(), 401);
    let e: Value = r.json().await.unwrap();
    assert_eq!(e["error"]["type"], "invalid_api_key");

    let r = client().get(format!("{url}/v1/models")).send().await.unwrap();
    assert_eq!(r.headers()["content-type"], "application/json; charset=utf-8");
    let models: Value = r.json().await.unwrap();
    assert_eq!(models["object"], "list");
    assert_eq!(models["data"][0]["id"], "test-chat");
    assert_eq!(client().get(format!("{url}/v1/models/test-chat")).send().await.unwrap().status(), 200);
    assert_eq!(client().get(format!("{url}/v1/models/nope")).send().await.unwrap().status(), 404);
    assert_eq!(client().get(format!("{url}/v1/nope")).send().await.unwrap().status(), 404);

    let (s, e) = post(url, "/v1/chat/completions", json!({"messages": []})).await;
    assert_eq!(s, 400);
    assert!(e["error"]["message"].as_str().unwrap().contains("empty"));
    let (s, _) = post(url, "/v1/chat/completions", json!({"n": 2, "messages": [{"role": "user", "content": "x"}]})).await;
    assert_eq!(s, 400);
    let r = client().post(format!("{url}/v1/chat/completions")).body("{not json").send().await.unwrap();
    assert_eq!(r.status(), 400);
    let (s, e) = post(url, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "x"}], "tool_choice": {"type": "function", "function": {"name": "missing"}}, "tools": [{"type": "function", "function": {"name": "f"}}]})).await;
    assert_eq!(s, 400, "{e}");
}

#[tokio::test]
async fn chat_and_stream_agree() {
    need_server!(url);
    let (s, r) = post(url, "/v1/chat/completions", greedy("What is the capital of France? One word.")).await;
    assert_eq!(s, 200, "{r}");
    assert_eq!(r["object"], "chat.completion");
    assert_eq!(r["model"], "test-chat");
    let msg = &r["choices"][0]["message"];
    assert_eq!(msg["role"], "assistant");
    let content = msg["content"].as_str().unwrap();
    assert!(content.contains("Paris"));
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    let u = &r["usage"];
    assert_eq!(u["total_tokens"].as_u64(), Some(u["prompt_tokens"].as_u64().unwrap() + u["completion_tokens"].as_u64().unwrap()));

    let mut body = greedy("What is the capital of France? One word.");
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    let chunks = sse(url, body).await;
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert!(chunks.iter().all(|c| c["object"] == "chat.completion.chunk" && c["id"] == chunks[0]["id"]));
    let streamed: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(streamed, content, "stream and non-stream differ");
    let finish = chunks.iter().find_map(|c| c["choices"][0]["finish_reason"].as_str()).unwrap();
    assert_eq!(finish, "stop");
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"], json!([]));
    assert!(last["usage"]["completion_tokens"].as_u64().unwrap() > 0);
}

fn weather_request(stream: bool) -> Value {
    json!({
        "temperature": 0, "max_tokens": 200, "stream": stream,
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    })
}

#[tokio::test]
async fn tool_calls_http() {
    need_server!(url);
    let (s, r) = post(url, "/v1/chat/completions", weather_request(false)).await;
    assert_eq!(s, 200, "{r}");
    let choice = &r["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert!(choice["message"]["content"].is_null());
    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "get_weather");
    let args: Value = serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert!(args["city"].as_str().unwrap().contains("Paris"));

    // Round trip: send the result back.
    let mut body = weather_request(false);
    body["messages"].as_array_mut().unwrap().extend([
        choice["message"].clone(),
        json!({"role": "tool", "tool_call_id": call["id"], "content": "{\"temperature_c\": 23, \"conditions\": \"sunny\"}"}),
    ]);
    let (s, r) = post(url, "/v1/chat/completions", body).await;
    assert_eq!(s, 200, "{r}");
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    assert!(r["choices"][0]["message"]["content"].as_str().unwrap().contains("23"));

    let chunks = sse(url, weather_request(true)).await;
    let calls: Vec<&Value> = chunks.iter().filter_map(|c| c["choices"][0]["delta"].get("tool_calls")).collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0][0]["index"], 0);
    assert_eq!(calls[0][0]["function"]["name"], "get_weather");
    assert!(!chunks.iter().any(|c| c["choices"][0]["delta"]["content"].as_str().is_some_and(|t| t.contains("tool_call"))));
    assert_eq!(chunks.iter().find_map(|c| c["choices"][0]["finish_reason"].as_str()), Some("tool_calls"));
}

#[tokio::test]
async fn json_schema_over_http() {
    need_server!(url);
    let mut body = greedy("Extract: 'Alan Turing, 41'.");
    body["response_format"] = json!({"type": "json_schema", "json_schema": {"name": "p", "schema": {
        "type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"], "additionalProperties": false}}});
    let (s, r) = post(url, "/v1/chat/completions", body).await;
    assert_eq!(s, 200, "{r}");
    let v: Value = serde_json::from_str(r["choices"][0]["message"]["content"].as_str().unwrap()).unwrap();
    assert_eq!(v["age"], 41);
}

#[tokio::test]
async fn embeddings_float_and_base64() {
    need_server!(url);
    if std::env::var("OFFGRID_TEST_EMBED_MODEL").is_err() {
        return;
    }
    let (s, r) = post(url, "/v1/embeddings", json!({"input": ["a cat", "a kitten"], "model": "x"})).await;
    assert_eq!(s, 200, "{r}");
    assert_eq!(r["data"].as_array().unwrap().len(), 2);
    let f: Vec<f32> = serde_json::from_value(r["data"][1]["embedding"].clone()).unwrap();
    let (_, r) = post(url, "/v1/embeddings", json!({"input": ["a cat", "a kitten"], "encoding_format": "base64"})).await;
    let bytes = base64::engine::general_purpose::STANDARD.decode(r["data"][1]["embedding"].as_str().unwrap()).unwrap();
    let b: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    assert_eq!(f, b);
    assert!(r["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    let (s, _) = post(url, "/v1/embeddings", json!({"input": ""})).await;
    assert_eq!(s, 400);
}

#[tokio::test]
async fn parallel_requests() {
    need_server!(url);
    let t = Instant::now();
    let reqs = (0..4).map(|i| post(url, "/v1/chat/completions", greedy(&format!("Repeat exactly, nothing else: token-{i}"))));
    let results = futures_util::future::join_all(reqs).await;
    for (i, (s, r)) in results.iter().enumerate() {
        assert_eq!(*s, 200, "{r}");
        assert!(r["choices"][0]["message"]["content"].as_str().unwrap().contains(&format!("token-{i}")), "{r}");
    }
    println!("4 requests on 2 sessions: {:?}", t.elapsed());
}

#[tokio::test]
async fn disconnect_cancels_generation() {
    let Ok(model) = std::env::var("OFFGRID_TEST_MODEL") else { return };
    // Dedicated single-session server so the follow-up request must wait for the session.
    let url = spawn(ServerConfig { model: Some(model.into()), parallel: 1, ..Default::default() });
    let long = json!({"stream": true, "max_tokens": 3000, "temperature": 0.7,
        "messages": [{"role": "user", "content": "Write a very long essay about the history of rivers."}]});
    let mut resp = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&long).send().await.unwrap();
    // Read a few chunks, then hang up.
    for _ in 0..5 {
        resp.chunk().await.unwrap();
    }
    drop(resp);
    let t = Instant::now();
    let r = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&greedy("Say hi.")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let waited = t.elapsed();
    println!("follow-up served after {waited:?}");
    // 3000 tokens at ~150 tok/s would take ~20 s if the stream kept generating.
    assert!(waited < Duration::from_secs(5), "session not released after disconnect: {waited:?}");
}

#[tokio::test]
async fn reasoning_streams_over_sse() {
    let Ok(model) = std::env::var("OFFGRID_TEST_THINKING_MODEL") else { return };
    let url = spawn(ServerConfig { model: Some(model.into()), parallel: 1, ..Default::default() });
    // Qwen3 thinking mode loops under greedy decoding; use its recommended sampling, seeded.
    let body = json!({"stream": true, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 7, "max_tokens": 3000,
        "messages": [{"role": "user", "content": "What is 17 + 25? Answer with just the number."}]});
    let r = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&body).send().await.unwrap();
    let text = r.text().await.unwrap();
    let chunks: Vec<Value> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let reasoning: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["reasoning_content"].as_str()).collect();
    let content: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    let first_content = chunks.iter().position(|c| c["choices"][0]["delta"]["content"].as_str().is_some_and(|t| !t.is_empty()));
    let last_reasoning = chunks.iter().rposition(|c| c["choices"][0]["delta"].get("reasoning_content").is_some());
    let finish = chunks.iter().find_map(|c| c["choices"][0]["finish_reason"].as_str());
    println!("reasoning {} chars, content {content:?}, finish {finish:?}", reasoning.len());
    assert_eq!(finish, Some("stop"));
    assert!(!reasoning.is_empty());
    assert!(content.contains("42"));
    assert!(last_reasoning < first_content, "reasoning must precede content");
    assert!(!content.contains("think"));
}

#[tokio::test]
async fn vision_over_http() {
    let (Ok(model), Ok(mmproj), Ok(images)) =
        (std::env::var("OFFGRID_TEST_VL_MODEL"), std::env::var("OFFGRID_TEST_VL_MMPROJ"), std::env::var("OFFGRID_TEST_IMAGES"))
    else {
        return;
    };
    let url = spawn(ServerConfig { model: Some(model.into()), mmproj: Some(mmproj.into()), parallel: 1, ..Default::default() });
    let png = std::fs::read(std::path::Path::new(&images).join("text_hello.png")).unwrap();
    let data_url = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(png));
    let body = json!({"temperature": 0, "max_tokens": 40, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What text is in this image?"},
        {"type": "image_url", "image_url": {"url": data_url}}
    ]}]});
    let r = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    let text = v["choices"][0]["message"]["content"].as_str().unwrap().to_uppercase();
    println!("vision reply: {text:?}");
    assert!(text.contains("HELLO") && text.contains("42"));

    // Remote URLs are refused (offline: never fetched).
    let bad = json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://example.com/a.png"}}]}]});
    let r = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&bad).send().await.unwrap();
    assert_eq!(r.status(), 400);
    let e: Value = r.json().await.unwrap();
    assert!(e["error"]["message"].as_str().unwrap().contains("data: URL"), "{e}");
}

#[tokio::test]
async fn audio_over_http() {
    let (Ok(model), Ok(mmproj), Ok(audio)) = (
        std::env::var("OFFGRID_TEST_OMNI_MODEL"),
        std::env::var("OFFGRID_TEST_OMNI_MMPROJ"),
        std::env::var("OFFGRID_TEST_AUDIO"),
    ) else {
        return;
    };
    let url = spawn(ServerConfig { model: Some(model.into()), mmproj: Some(mmproj.into()), parallel: 1, ..Default::default() });
    let mp3 = std::fs::read(std::path::Path::new(&audio).join("jfk.mp3")).unwrap();
    let body = json!({"temperature": 0, "max_tokens": 60, "stream": true, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe this audio exactly."},
        {"type": "input_audio", "input_audio": {"data": base64::engine::general_purpose::STANDARD.encode(mp3), "format": "mp3"}}
    ]}]});
    let r = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let text: String = r
        .text()
        .await
        .unwrap()
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str().map(String::from))
        .collect();
    println!("audio reply (streamed): {text:?}");
    assert!(text.to_lowercase().contains("ask not"), "{text}");

    let bad = json!({"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "ogg"}}]}]});
    let r = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&bad).send().await.unwrap();
    assert_eq!(r.status(), 400);
}

#[tokio::test]
async fn large_bodies_and_body_errors() {
    let Ok(model) = std::env::var("OFFGRID_TEST_MODEL") else { return };
    let url = spawn(ServerConfig { model: Some(model.into()), parallel: 1, max_body_bytes: 8 << 20, ..Default::default() });
    // 5 MB body (over axum's 2 MB default, under our 8 MB limit) reaches the handler.
    let filler = "x".repeat(5 << 20);
    let body = json!({"max_tokens": 1, "messages": [{"role": "user", "content": "hi"}], "pad": filler});
    let r = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    // Over the limit: 413 with an OpenAI-style JSON error.
    let huge = json!({"messages": [{"role": "user", "content": "hi"}], "pad": "x".repeat(9 << 20)});
    let r = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&huge).send().await.unwrap();
    assert_eq!(r.status(), 413);
    let e: Value = r.json().await.unwrap();
    assert_eq!(e["error"]["type"], "request_too_large");
    // Malformed embeddings JSON: JSON error, not plain text.
    let r = reqwest::Client::new()
        .post(format!("{url}/v1/embeddings"))
        .header("content-type", "application/json")
        .body("{nope")
        .send()
        .await
        .unwrap();
    assert!(r.status().is_client_error());
    let e: Value = r.json().await.unwrap();
    assert!(e["error"]["message"].is_string());
}

#[tokio::test]
async fn disconnect_cancels_hidden_tool_call_output() {
    let Ok(model) = std::env::var("OFFGRID_TEST_MODEL") else { return };
    let url = spawn(ServerConfig { model: Some(model.into()), parallel: 1, ..Default::default() });
    // Forced tool call with a long string argument: all generated text is tool-call markup
    // that the stream filter hides, so no content events are sent while generating.
    let body = json!({"stream": true, "max_tokens": 3000, "temperature": 0.7,
        "tool_choice": "required",
        "tools": [{"type": "function", "function": {"name": "save_essay", "parameters": {
            "type": "object", "properties": {"essay": {"type": "string"}}, "required": ["essay"]}}}],
        "messages": [{"role": "user", "content": "Write a 2000-word essay about rivers and save it."}]});
    let mut resp = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&body).send().await.unwrap();
    resp.chunk().await.unwrap(); // role chunk
    tokio::time::sleep(Duration::from_millis(300)).await;
    drop(resp);
    let t = Instant::now();
    let r = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&greedy("Say hi.")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    println!("follow-up after hidden-output disconnect: {:?}", t.elapsed());
    assert!(t.elapsed() < Duration::from_secs(5), "session stayed busy after disconnect");
}
