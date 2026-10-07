//! Chat templates + structured (JSON) output against a real model.
//! Needs AICORE_TEST_MODEL; skipped otherwise.

use std::sync::Arc;

use aicore::*;
use serde_json::{Value, json};

fn model() -> Option<Arc<Model>> {
    let Ok(path) = std::env::var("AICORE_TEST_MODEL") else {
        eprintln!("skip: AICORE_TEST_MODEL not set");
        return None;
    };
    Some(Model::load(path, &ModelParams::default()).expect("model loads"))
}

fn request(content: &str, format: Option<ResponseFormat>) -> ChatRequest {
    ChatRequest {
        max_tokens: Some(400),
        response_format: format,
        sampling: SamplingParams { temperature: 0.0, ..Default::default() },
        ..ChatRequest::new(vec![ChatMessage::system("You are a precise assistant."), ChatMessage::user(content)])
    }
}

#[test]
fn jinja_matches_builtin_for_chatml_model() {
    let Some(model) = model() else { return };
    let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
    let mut req = request("Hello", None);
    req.messages.push(ChatMessage::assistant("Hi!"));
    req.messages.push(ChatMessage::user("Bye"));
    let jinja = chat.render_prompt(&req).unwrap();
    req.chat_template = Some("chatml".into());
    let builtin = chat.render_prompt(&req).unwrap();
    println!("{jinja}");
    assert_eq!(jinja, builtin);
}

#[test]
fn template_kwargs_and_override() {
    let Some(model) = model() else { return };
    let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
    let mut req = request("Hello", None);
    req.chat_template = Some("{{ greeting }}:{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}".into());
    req.chat_template_kwargs = Some(json!({"greeting": "HI"}).as_object().unwrap().clone());
    assert_eq!(chat.render_prompt(&req).unwrap(), "HI:[system]You are a precise assistant.[user]Hello");
}

#[test]
fn json_object_mode() {
    let Some(model) = model() else { return };
    let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
    let r = chat
        .complete(&request("Describe a cat with a few fields, as JSON.", Some(ResponseFormat::JsonObject { schema: None })), |_| true)
        .unwrap();
    println!("{}", r.content);
    let v: Value = serde_json::from_str(&r.content).expect("valid JSON");
    assert!(v.is_object());
}

#[test]
fn json_schema_outputs_validate() {
    let Some(model) = model() else { return };
    let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
    let cases = [
        (
            "Extract the person: 'Ada Lovelace, 36, mathematician and writer, born 1815-12-10'.",
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "age": {"type": "integer"},
                    "occupations": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
                    "born": {"type": "string", "format": "date"}
                },
                "required": ["name", "age", "occupations", "born"],
                "additionalProperties": false
            }),
        ),
        (
            "Classify the sentiment of: 'This product is wonderful, I love it'.",
            json!({
                "type": "object",
                "properties": {
                    "sentiment": {"enum": ["positive", "neutral", "negative"]},
                    "confidence": {"type": "number"}
                },
                "required": ["sentiment", "confidence"],
                "additionalProperties": false
            }),
        ),
        (
            "Give a shopping list for pancakes.",
            json!({
                "$defs": {"item": {"type": "object", "properties": {
                    "name": {"type": "string"}, "qty": {"type": "integer"}},
                    "required": ["name", "qty"], "additionalProperties": false}},
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/item"}, "minItems": 2, "maxItems": 6}},
                "required": ["items"],
                "additionalProperties": false
            }),
        ),
        (
            "Is 17 prime? Answer with the result.",
            json!({
                "anyOf": [
                    {"type": "object", "properties": {"prime": {"const": true}, "reason": {"type": "string"}}, "required": ["prime", "reason"], "additionalProperties": false},
                    {"type": "object", "properties": {"prime": {"const": false}, "factor": {"type": "integer"}}, "required": ["prime", "factor"], "additionalProperties": false}
                ]
            }),
        ),
    ];
    for (prompt, schema) in cases {
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat { name: Some("out".into()), schema: schema.clone(), strict: Some(true) },
        };
        let r = chat.complete(&request(prompt, Some(format)), |_| true).unwrap();
        println!("{prompt}\n  -> {}  [{:?}]", r.content, r.finish_reason);
        assert_eq!(r.finish_reason, FinishReason::Stop, "output truncated");
        let v: Value = serde_json::from_str(&r.content).expect("valid JSON");
        let validator = jsonschema::validator_for(&schema).expect("schema compiles");
        let errors: Vec<String> = validator.iter_errors(&v).map(|e| e.to_string()).collect();
        assert!(errors.is_empty(), "schema violations: {errors:?}");
    }
}

#[test]
fn grammar_and_response_format_conflict() {
    let Some(model) = model() else { return };
    let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
    let mut req = request("x", Some(ResponseFormat::JsonObject { schema: None }));
    req.sampling.grammar = Some(r#"root ::= "a""#.into());
    assert!(matches!(chat.complete(&req, |_| true), Err(Error::InvalidArgument(_))));
}

#[test]
fn response_format_json_shapes() {
    // OpenAI wire format round-trips.
    let f: ResponseFormat =
        serde_json::from_value(json!({"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}})).unwrap();
    assert!(matches!(f, ResponseFormat::JsonSchema { .. }));
    let f: ResponseFormat = serde_json::from_value(json!({"type": "json_object"})).unwrap();
    assert_eq!(f, ResponseFormat::JsonObject { schema: None });
}
