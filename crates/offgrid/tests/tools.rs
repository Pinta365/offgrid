//! Tool calling end-to-end. Runs per model that is configured:
//!   OFFGRID_TEST_MODEL        Qwen3 (Hermes <tool_call> format)
//!   OFFGRID_TEST_LLAMA_MODEL  Llama 3.x (bare JSON format)
//!   OFFGRID_TEST_GEMMA_MODEL  Gemma 3 (template ignores tools: they are described in the
//!                            system message, generic <tool_call> format)
//!
//! Format/plumbing tests run on every model. Judgment tests (don't call when not
//! needed, answer from the tool result) only run on OFFGRID_TEST_MODEL: Llama 3.2 3B
//! re-calls tools instead of answering, and llama.cpp's own llama-server shows the same
//! behaviour with it (checked 2026-10-08), so that's the model, not this code.

use std::sync::{Arc, OnceLock};

use offgrid::*;
use serde_json::{Value, json};

/// Models trusted to make good tool-use decisions.
fn judging_models() -> Vec<(&'static str, Arc<Model>)> {
    models().into_iter().filter(|(v, _)| matches!(*v, "OFFGRID_TEST_MODEL" | "OFFGRID_TEST_GEMMA_MODEL")).collect()
}

/// Each configured model, loaded once and shared by the tests (they run in parallel).
fn models() -> Vec<(&'static str, Arc<Model>)> {
    static MODELS: OnceLock<Vec<(&'static str, Arc<Model>)>> = OnceLock::new();
    MODELS
        .get_or_init(|| {
            ["OFFGRID_TEST_MODEL", "OFFGRID_TEST_LLAMA_MODEL", "OFFGRID_TEST_GEMMA_MODEL"]
                .iter()
                .filter_map(|var| {
                    let path = std::env::var(var).ok()?;
                    Some((*var, Model::load(path, &ModelParams::default()).expect("model loads")))
                })
                .collect()
        })
        .clone()
}

fn weather_tool() -> Tool {
    Tool::function(
        "get_weather",
        "Get the current weather for a city.",
        json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
                "unit": {"enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"],
            "additionalProperties": false
        }),
    )
}

fn time_tool() -> Tool {
    Tool::function(
        "get_time",
        "Get the current local time in a timezone.",
        json!({"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"], "additionalProperties": false}),
    )
}

fn request(messages: Vec<ChatMessage>, choice: Option<ToolChoice>) -> ChatRequest {
    ChatRequest {
        max_tokens: Some(300),
        tools: vec![weather_tool(), time_tool()],
        tool_choice: choice,
        sampling: SamplingParams { temperature: 0.0, ..Default::default() },
        ..ChatRequest::new(messages)
    }
}

fn user(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage::system("You are a helpful assistant. Use tools when they help."), ChatMessage::user(text)]
}

/// Arguments must be a JSON object valid against the tool's parameter schema.
fn assert_valid_args(call: &ToolCall, tools: &[Tool]) -> Value {
    let tool = tools.iter().find(|t| t.function.name == call.function.name).expect("known tool");
    let args: Value = serde_json::from_str(&call.function.arguments).expect("arguments are JSON");
    let schema = tool.function.parameters.clone().unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    let errors: Vec<String> = validator.iter_errors(&args).map(|e| e.to_string()).collect();
    assert!(errors.is_empty(), "{}: {errors:?}", call.function.arguments);
    args
}

fn complete(chat: &mut ChatSession, req: &ChatRequest) -> (ChatResponse, String) {
    let mut streamed = String::new();
    let r = chat.complete(req, |p| { streamed.push_str(p); true }).unwrap();
    for leak in ["<tool_call>", "</tool_call>", "\"arguments\"", "\"parameters\"", "<think>"] {
        assert!(!streamed.contains(leak), "streamed text leaked {leak}: {streamed:?}");
    }
    (r, streamed)
}

#[test]
fn auto_calls_tool_when_useful() {
    for (name, model) in models() {
        let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
        let req = request(user("What's the weather like in Paris right now?"), None);
        let (r, _) = complete(&mut chat, &req);
        println!("[{name}] {r:?}");
        assert_eq!(r.finish_reason, FinishReason::ToolCalls, "[{name}]");
        assert_eq!(r.tool_calls.len(), 1, "[{name}]");
        assert_eq!(r.tool_calls[0].function.name, "get_weather");
        let args = assert_valid_args(&r.tool_calls[0], &req.tools);
        assert!(args["city"].as_str().unwrap().contains("Paris"));
        assert!(r.tool_calls[0].id.starts_with("call_"));
    }
}

#[test]
fn auto_answers_directly_when_no_tool_fits() {
    for (name, model) in judging_models() {
        let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
        let (r, streamed) = complete(&mut chat, &request(user("What is 12 times 12? Reply with just the number."), None));
        println!("[{name}] {r:?}");
        assert!(r.tool_calls.is_empty(), "[{name}] unexpected call");
        assert!(r.content.contains("144"), "[{name}]");
        assert_eq!(streamed.trim(), r.content, "[{name}]");
    }
}

#[test]
fn required_and_named_choice() {
    for (name, model) in models() {
        let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
        let req = request(user("Hello!"), Some(ToolChoice::Required));
        let (r, _) = complete(&mut chat, &req);
        println!("[{name}] required: {r:?}");
        assert!(!r.tool_calls.is_empty(), "[{name}] required must call");
        assert_valid_args(&r.tool_calls[0], &req.tools);

        let req = request(user("What's the weather in Oslo?"), Some(ToolChoice::Function("get_time".into())));
        let (r, _) = complete(&mut chat, &req);
        println!("[{name}] named: {r:?}");
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].function.name, "get_time", "[{name}] forced tool");
        assert_valid_args(&r.tool_calls[0], &req.tools);
    }
}

#[test]
fn round_trip_uses_tool_result() {
    for (name, model) in judging_models() {
        let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
        let mut messages = user("What's the weather in Berlin? Answer in one sentence.");
        let (r, _) = complete(&mut chat, &request(messages.clone(), None));
        assert_eq!(r.finish_reason, FinishReason::ToolCalls, "[{name}] {r:?}");
        messages.push(r.to_message());
        for call in &r.tool_calls {
            messages.push(ChatMessage::tool_result(call, r#"{"temperature_c": 17, "conditions": "light rain"}"#));
        }
        let (r2, _) = complete(&mut chat, &request(messages, None));
        println!("[{name}] final: {r2:?}");
        assert!(r2.tool_calls.is_empty(), "[{name}] should answer now");
        assert_eq!(r2.finish_reason, FinishReason::Stop);
        let text = r2.content.to_lowercase();
        assert!(text.contains("17") && text.contains("rain"), "[{name}] answer ignores tool result: {}", r2.content);
        assert!(r2.usage.cached_tokens > 0, "[{name}] prefix reuse across tool turn");
    }
}

#[test]
fn parallel_calls_hermes() {
    let Some((_, model)) = models().into_iter().find(|(v, _)| *v == "OFFGRID_TEST_MODEL") else { return };
    let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
    let req = request(user("What's the weather in Paris and in Tokyo?"), None);
    let (r, _) = complete(&mut chat, &req);
    println!("{r:?}");
    let cities: Vec<String> = r
        .tool_calls
        .iter()
        .map(|c| assert_valid_args(c, &req.tools)["city"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(cities.len(), 2, "{cities:?}");
    assert!(cities.iter().any(|c| c.contains("Paris")) && cities.iter().any(|c| c.contains("Tokyo")));
    assert_ne!(r.tool_calls[0].id, r.tool_calls[1].id);
}

#[test]
fn choice_none_hides_tools() {
    for (name, model) in models() {
        let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
        let req = request(user("What's the weather in Paris?"), Some(ToolChoice::None));
        assert!(!chat.render_prompt(&req).unwrap().contains("get_weather"), "[{name}]");
        let (r, _) = complete(&mut chat, &req);
        println!("[{name}] none: {r:?}");
        assert!(r.tool_calls.is_empty(), "[{name}]");
        assert_ne!(r.finish_reason, FinishReason::ToolCalls);
    }
}

#[test]
fn tools_visible_in_prompt() {
    for (name, model) in models() {
        let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
        let prompt = chat.render_prompt(&request(user("hi"), None)).unwrap();
        assert!(prompt.contains("get_weather") && prompt.contains("get_time"), "[{name}] {prompt}");
    }
}

#[test]
fn tool_history_renders_natively() {
    // Assistant tool calls and tool results go through the model's own template, with
    // arguments as objects and JSON results not double-encoded.
    for (name, model) in models() {
        let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
        let call = ToolCall {
            id: "call_1".into(),
            kind: "function".into(),
            function: FunctionCall { name: "get_weather".into(), arguments: r#"{"city":"Berlin"}"#.into() },
        };
        let mut messages = user("Weather in Berlin?");
        messages.push(ChatMessage::assistant_tool_calls("", vec![call.clone()]));
        messages.push(ChatMessage::tool_result(&call, r#"{"temperature_c": 17}"#));
        let prompt = chat.render_prompt(&request(messages, None)).unwrap();
        assert!(prompt.contains(r#"{"temperature_c": 17}"#), "[{name}] {prompt}");
        assert!(!prompt.contains(r#"\"temperature_c\""#), "[{name}] double-encoded: {prompt}");
        assert!(prompt.contains(r#""city": "Berlin""#), "[{name}] call arguments missing: {prompt}");
    }
}
