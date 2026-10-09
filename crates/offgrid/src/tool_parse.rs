//! Tool-call output parsing for the supported chat templates.
//!
//! Pure functions: no llama.cpp calls, no I/O (except tests reading template
//! files). Each parser extracts zero or more tool calls from model output and
//! returns the remaining natural-language content plus optional reasoning.
//!
//! Parsed `arguments` is always a compact JSON object string.

use serde_json::{Map, Value};

/// Detected tool-call convention of a chat template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolFormat {
    Hermes,
    Llama3Json,
    MistralList,
    MistralArgs,
    Qwen3CoderXml,
    Generic,
    /// Hermes-style `<tool_call>` blocks that offgrid taught the model in the system message,
    /// for templates that ignore `tools` (see `tools::polyfill_messages`).
    Polyfill,
}

/// One parsed tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedCall {
    pub name: String,
    /// Compact JSON object text.
    pub arguments: String,
}

/// Result of parsing a model output.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Parsed {
    pub content: String,
    pub reasoning: Option<String>,
    pub calls: Vec<ParsedCall>,
}

/// Detect the tool format from a chat template. Rules are evaluated in order.
pub(crate) fn detect_format(chat_template: &str) -> ToolFormat {
    if chat_template.contains("<function=") && chat_template.contains("<parameter=") {
        ToolFormat::Qwen3CoderXml
    } else if chat_template.contains("[TOOL_CALLS]") && chat_template.contains("[ARGS]") {
        ToolFormat::MistralArgs
    } else if chat_template.contains("[TOOL_CALLS]") {
        ToolFormat::MistralList
    } else if chat_template.contains("<tool_call>") {
        ToolFormat::Hermes
    } else if chat_template.contains("<|python_tag|>") || chat_template.contains("ipython") {
        ToolFormat::Llama3Json
    } else {
        ToolFormat::Generic
    }
}

/// Text that begins a tool call, used to stop streaming content.
pub(crate) fn tool_call_start_marker(format: ToolFormat) -> Option<&'static str> {
    match format {
        ToolFormat::Hermes => Some("<tool_call>"),
        ToolFormat::Qwen3CoderXml => Some("<tool_call>"),
        ToolFormat::MistralList | ToolFormat::MistralArgs => Some("[TOOL_CALLS]"),
        ToolFormat::Llama3Json => Some("<|python_tag|>"),
        ToolFormat::Generic | ToolFormat::Polyfill => Some("<tool_call>"),
    }
}

/// Split off a reasoning block, using the same rule as the stream filter
/// (`text::StreamFilter`) so streamed and final output agree:
/// - the reply starts (after whitespace) with `<think>`: up to `</think>` is reasoning;
///   if it never closes (cut off), all of it is reasoning;
/// - the prompt already opened `<think>`: up to `</think>` is reasoning, or everything if
///   it never closes;
/// - otherwise there is no reasoning; a `<think>` later in the text is ordinary content.
pub(crate) fn split_reasoning(text: &str, think_open_in_prompt: bool) -> (Option<String>, String) {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let body = if think_open_in_prompt {
        text
    } else if let Some(rest) = text.trim_start().strip_prefix(OPEN) {
        rest
    } else {
        return (None, text.to_string());
    };
    match body.find(CLOSE) {
        Some(i) => (Some(body[..i].trim().to_string()), body[i + CLOSE.len()..].trim_start().to_string()),
        None => (Some(body.trim().to_string()), String::new()),
    }
}

/// Parse a model output: reasoning split first, then tool calls from the rest.
pub(crate) fn parse_output(format: ToolFormat, text: &str, think_open_in_prompt: bool) -> Parsed {
    let (reasoning, rest) = split_reasoning(text, think_open_in_prompt);
    let (content, calls) = parse_calls(format, &rest);
    Parsed {
        content: content.trim().to_string(),
        reasoning,
        calls,
    }
}

fn parse_calls(format: ToolFormat, text: &str) -> (String, Vec<ParsedCall>) {
    match format {
        ToolFormat::Hermes | ToolFormat::Polyfill => parse_hermes(text),
        ToolFormat::Llama3Json => parse_llama3(text),
        ToolFormat::MistralList => parse_mistral_list(text),
        ToolFormat::MistralArgs => parse_mistral_args(text),
        ToolFormat::Qwen3CoderXml => parse_qwen3_xml(text),
        ToolFormat::Generic => parse_generic(text),
    }
}

/// Normalize an arguments value into compact JSON object text.
fn normalize_arguments(v: Option<&Value>) -> String {
    match v {
        None => "{}".to_string(),
        Some(Value::Object(o)) => Value::Object(o.clone()).to_string(),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(o)) => Value::Object(o).to_string(),
            _ => "{}".to_string(),
        },
        Some(other) => {
            let mut m = Map::new();
            m.insert("value".to_string(), other.clone());
            Value::Object(m).to_string()
        }
    }
}

fn call_from_object(obj: &Map<String, Value>) -> Option<ParsedCall> {
    let name = obj.get("name")?.as_str()?.to_string();
    let args = obj.get("arguments").or_else(|| obj.get("parameters"));
    Some(ParsedCall {
        name,
        arguments: normalize_arguments(args),
    })
}

fn call_from_json(text: &str) -> Option<ParsedCall> {
    let v: Value = serde_json::from_str(text.trim()).ok()?;
    call_from_object(v.as_object()?)
}

/// Parse the first JSON value in `s`, ignoring trailing text.
fn parse_first_json(s: &str) -> Option<Value> {
    let mut it = serde_json::Deserializer::from_str(s).into_iter::<Value>();
    it.next().and_then(|r| r.ok())
}

fn parse_hermes(text: &str) -> (String, Vec<ParsedCall>) {
    parse_tagged(text, "<tool_call>", "</tool_call>", call_from_json)
}

fn parse_qwen3_xml(text: &str) -> (String, Vec<ParsedCall>) {
    parse_tagged(text, "<tool_call>", "</tool_call>", parse_xml_function)
}

/// Generic scanner for `<open> ... <close>` blocks; invalid blocks stay in content.
fn parse_tagged(
    text: &str,
    open: &str,
    close: &str,
    parse_block: impl Fn(&str) -> Option<ParsedCall>,
) -> (String, Vec<ParsedCall>) {
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find(open) {
        content.push_str(&rest[..pos]);
        let after = &rest[pos + open.len()..];
        match after.find(close) {
            Some(cpos) => {
                let inner = &after[..cpos];
                let tail = &after[cpos + close.len()..];
                if let Some(call) = parse_block(inner) {
                    calls.push(call);
                    rest = tail;
                } else {
                    content.push_str(open);
                    rest = after;
                }
            }
            None => {
                if let Some(call) = parse_block(after) {
                    calls.push(call);
                    rest = "";
                    break;
                }
                content.push_str(open);
                rest = after;
            }
        }
    }
    content.push_str(rest);
    (content, calls)
}

/// Parse a Qwen3-Coder XML function block into a single call.
fn parse_xml_function(inner: &str) -> Option<ParsedCall> {
    const FOPEN: &str = "<function=";
    const POPEN: &str = "<parameter=";
    const PCLOSE: &str = "</parameter>";
    let fpos = inner.find(FOPEN)?;
    let after = &inner[fpos + FOPEN.len()..];
    let name_end = after.find('>')?;
    let name = after[..name_end].trim();
    if name.is_empty() {
        return None;
    }
    let mut body = &after[name_end + 1..];
    let mut map = Map::new();
    while let Some(p) = body.find(POPEN) {
        let a = &body[p + POPEN.len()..];
        let ke = a.find('>')?;
        let key = a[..ke].trim();
        let vbody = &a[ke + 1..];
        let vpos = vbody.find(PCLOSE)?;
        map.insert(key.to_string(), parse_param_value(&vbody[..vpos]));
        body = &vbody[vpos + PCLOSE.len()..];
    }
    Some(ParsedCall {
        name: name.to_string(),
        arguments: Value::Object(map).to_string(),
    })
}

/// XML parameter value: strip one leading/trailing newline, then use a JSON
/// number/bool/null/object/array if it parses, otherwise a JSON string.
fn parse_param_value(raw: &str) -> Value {
    let s = raw.strip_prefix('\n').unwrap_or(raw);
    let s = s.strip_suffix('\n').unwrap_or(s);
    match serde_json::from_str::<Value>(s) {
        Ok(v @ (Value::Number(_) | Value::Bool(_) | Value::Null | Value::Object(_) | Value::Array(_))) => v,
        _ => Value::String(s.to_string()),
    }
}

fn parse_llama3(text: &str) -> (String, Vec<ParsedCall>) {
    let trimmed = text.trim();
    let body = trimmed
        .strip_prefix("<|python_tag|>")
        .map(|s| s.trim_start())
        .unwrap_or(trimmed);

    if let Some(call) = call_from_json(body) {
        return (String::new(), vec![call]);
    }
    let parts: Vec<&str> = body
        .split([';', '\n'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if !parts.is_empty() {
        let mut calls = Vec::new();
        let mut all_ok = true;
        for part in &parts {
            match call_from_json(part) {
                Some(c) => calls.push(c),
                None => {
                    all_ok = false;
                    break;
                }
            }
        }
        if all_ok && !calls.is_empty() {
            return (String::new(), calls);
        }
    }
    (text.to_string(), Vec::new())
}

fn parse_mistral_list(text: &str) -> (String, Vec<ParsedCall>) {
    const MARKER: &str = "[TOOL_CALLS]";
    let pos = match text.find(MARKER) {
        Some(p) => p,
        None => return (text.to_string(), Vec::new()),
    };
    let content = text[..pos].to_string();
    let rest = text[pos + MARKER.len()..].trim();
    if let Some(Value::Array(arr)) = parse_first_json(rest) {
        let mut calls = Vec::new();
        for item in &arr {
            if let Some(obj) = item.as_object()
                && let Some(c) = call_from_object(obj) {
                    calls.push(c);
                }
        }
        return (content, calls);
    }
    (content, Vec::new())
}

fn parse_mistral_args(text: &str) -> (String, Vec<ParsedCall>) {
    const MARKER: &str = "[TOOL_CALLS]";
    const ARGS: &str = "[ARGS]";
    let pos = match text.find(MARKER) {
        Some(p) => p,
        None => return (text.to_string(), Vec::new()),
    };
    let content = text[..pos].to_string();
    let rest = &text[pos + MARKER.len()..];
    let mut calls = Vec::new();
    for seg in rest.split(MARKER) {
        let seg = seg.trim();
        if seg.is_empty() {
            continue;
        }
        if let Some(apos) = seg.find(ARGS) {
            let name = seg[..apos].trim();
            if name.is_empty() {
                continue;
            }
            let argtext = seg[apos + ARGS.len()..].trim();
            if let Some(v) = parse_first_json(argtext) {
                calls.push(ParsedCall {
                    name: name.to_string(),
                    arguments: normalize_arguments(Some(&v)),
                });
            }
        }
    }
    (content, calls)
}

fn parse_generic(text: &str) -> (String, Vec<ParsedCall>) {
    if text.contains("<tool_call>") {
        return parse_hermes(text);
    }
    let trimmed = text.trim();
    if let Some(call) = call_from_json(trimmed) {
        return (String::new(), vec![call]);
    }
    if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(trimmed)
        && let Some(Value::Array(arr)) = obj.get("tool_calls") {
            let mut calls = Vec::new();
            for item in arr {
                if let Some(o) = item.as_object()
                    && let Some(c) = call_from_object(o) {
                        calls.push(c);
                    }
            }
            if !calls.is_empty() {
                return (String::new(), calls);
            }
        }
    (text.to_string(), Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    /// Locate the repository root so tests can read the real chat templates.
    fn repo_root() -> PathBuf {
        let mut cands: Vec<PathBuf> = Vec::new();
        let f = std::path::Path::new(file!());
        if f.is_absolute()
            && let Some(b) = f
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
            {
                cands.push(b.to_path_buf());
            }
        let m = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        cands.push(m.join("../.."));
        cands.push(m.clone());
        cands.push(m.join(".."));
        for c in cands {
            if c.join("vendor/llama.cpp").is_dir() {
                return c;
            }
        }
        PathBuf::from(".")
    }

    fn template(name: &str) -> String {
        let p = repo_root().join("vendor/llama.cpp/models/templates").join(name);
        std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
    }

    fn args(call: &ParsedCall) -> Value {
        serde_json::from_str(&call.arguments).expect("arguments must be JSON")
    }

    // --- detect_format on real templates -------------------------------------

    #[test]
    fn detect_qwen3_is_hermes() {
        assert_eq!(detect_format(&template("Qwen-Qwen3-0.6B.jinja")), ToolFormat::Hermes);
    }

    #[test]
    fn detect_llama32_is_llama3json() {
        assert_eq!(
            detect_format(&template("meta-llama-Llama-3.2-3B-Instruct.jinja")),
            ToolFormat::Llama3Json
        );
    }

    #[test]
    fn detect_qwen3_coder_is_xml() {
        assert_eq!(
            detect_format(&template("Qwen3-Coder.jinja")),
            ToolFormat::Qwen3CoderXml
        );
    }

    #[test]
    fn detect_mistral_nemo_is_list() {
        assert_eq!(
            detect_format(&template("mistralai-Mistral-Nemo-Instruct-2407.jinja")),
            ToolFormat::MistralList
        );
    }

    #[test]
    fn detect_devstral_is_args() {
        assert_eq!(
            detect_format(&template("unsloth-mistral-Devstral-Small-2507.jinja")),
            ToolFormat::MistralArgs
        );
    }

    #[test]
    fn detect_plain_is_generic() {
        assert_eq!(detect_format("You are a helpful assistant."), ToolFormat::Generic);
    }

    // --- Hermes --------------------------------------------------------------

    #[test]
    fn hermes_single_call() {
        let text = "Hello\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        let p = parse_output(ToolFormat::Hermes, text, false);
        assert_eq!(p.content, "Hello");
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "get_weather");
        assert_eq!(args(&p.calls[0]), json!({"city": "Paris"}));
    }

    #[test]
    fn hermes_multiple_calls() {
        let text = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call> and <tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>";
        let p = parse_output(ToolFormat::Hermes, text, false);
        assert_eq!(p.calls.len(), 2);
        assert_eq!(p.calls[0].name, "a");
        assert_eq!(p.calls[1].name, "b");
    }

    #[test]
    fn hermes_content_before_call() {
        let text = "I will check.\n<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>";
        let p = parse_output(ToolFormat::Hermes, text, false);
        assert_eq!(p.content, "I will check.");
        assert_eq!(p.calls.len(), 1);
    }

    #[test]
    fn hermes_unclosed_tag_at_end() {
        let text = "<tool_call>{\"name\":\"a\",\"arguments\":{\"x\":1}}";
        let p = parse_output(ToolFormat::Hermes, text, false);
        assert_eq!(p.calls.len(), 1);
        assert_eq!(args(&p.calls[0]), json!({"x": 1}));
    }

    #[test]
    fn hermes_malformed_block_stays_in_content() {
        let text = "prefix <tool_call>not json</tool_call> suffix";
        let p = parse_output(ToolFormat::Hermes, text, false);
        assert!(p.calls.is_empty());
        assert!(p.content.contains("not json"));
        assert!(p.content.contains("suffix"));
    }

    #[test]
    fn hermes_string_arguments_are_parsed() {
        let text = "<tool_call>{\"name\":\"a\",\"arguments\":\"{\\\"x\\\": 5}\"}</tool_call>";
        let p = parse_output(ToolFormat::Hermes, text, false);
        assert_eq!(p.calls.len(), 1);
        assert_eq!(args(&p.calls[0]), json!({"x": 5}));
    }

    // --- Llama3Json ----------------------------------------------------------

    #[test]
    fn llama3_python_tag_single() {
        let text = "<|python_tag|>{\"name\":\"f\",\"parameters\":{\"x\":1}}";
        let p = parse_output(ToolFormat::Llama3Json, text, false);
        assert_eq!(p.content, "");
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "f");
        assert_eq!(args(&p.calls[0]), json!({"x": 1}));
    }

    #[test]
    fn llama3_whole_object_no_tag() {
        let text = "  {\"name\":\"f\",\"arguments\":{}}  ";
        let p = parse_output(ToolFormat::Llama3Json, text, false);
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "f");
    }

    #[test]
    fn llama3_multiple_calls() {
        let a = parse_output(
            ToolFormat::Llama3Json,
            "{\"name\":\"a\",\"arguments\":{}};{\"name\":\"b\",\"arguments\":{}}",
            false,
        );
        assert_eq!(a.calls.len(), 2);
        let b = parse_output(
            ToolFormat::Llama3Json,
            "{\"name\":\"a\",\"arguments\":{}}\n{\"name\":\"b\",\"arguments\":{}}",
            false,
        );
        assert_eq!(b.calls.len(), 2);
    }

    #[test]
    fn llama3_invalid_is_all_content() {
        let text = "Sure, here is the answer.";
        let p = parse_output(ToolFormat::Llama3Json, text, false);
        assert!(p.calls.is_empty());
        assert_eq!(p.content, text);
    }

    // --- MistralList ---------------------------------------------------------

    #[test]
    fn mistral_list_calls() {
        let text = "[TOOL_CALLS][{\"name\":\"a\",\"arguments\":{\"x\":1}},{\"name\":\"b\",\"arguments\":{}}]";
        let p = parse_output(ToolFormat::MistralList, text, false);
        assert_eq!(p.calls.len(), 2);
        assert_eq!(p.calls[0].name, "a");
        assert_eq!(args(&p.calls[0]), json!({"x": 1}));
    }

    #[test]
    fn mistral_list_content_before() {
        let text = "Thinking...\n[TOOL_CALLS][{\"name\":\"a\",\"arguments\":{}}]";
        let p = parse_output(ToolFormat::MistralList, text, false);
        assert_eq!(p.content, "Thinking...");
        assert_eq!(p.calls.len(), 1);
    }

    // --- MistralArgs ---------------------------------------------------------

    #[test]
    fn mistral_args_single() {
        let text = "[TOOL_CALLS]get_weather[ARGS]{\"city\":\"Paris\"}";
        let p = parse_output(ToolFormat::MistralArgs, text, false);
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "get_weather");
        assert_eq!(args(&p.calls[0]), json!({"city": "Paris"}));
    }

    #[test]
    fn mistral_args_multiple() {
        let text = "ok [TOOL_CALLS]a[ARGS]{\"x\":1}[TOOL_CALLS]b[ARGS]{\"y\":2}";
        let p = parse_output(ToolFormat::MistralArgs, text, false);
        assert_eq!(p.content, "ok");
        assert_eq!(p.calls.len(), 2);
        assert_eq!(p.calls[1].name, "b");
        assert_eq!(args(&p.calls[1]), json!({"y": 2}));
    }

    // --- Qwen3CoderXml -------------------------------------------------------

    #[test]
    fn qwen3_xml_single() {
        let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let p = parse_output(ToolFormat::Qwen3CoderXml, text, false);
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "get_weather");
        assert_eq!(args(&p.calls[0]), json!({"city": "Paris", "days": 3}));
    }

    #[test]
    fn qwen3_xml_multiple_with_content() {
        let text = "Here you go:\n<tool_call><function=a><parameter=x>\n1\n</parameter></function></tool_call>\n<tool_call><function=b><parameter=y>\ntwo\n</parameter></function></tool_call>";
        let p = parse_output(ToolFormat::Qwen3CoderXml, text, false);
        assert_eq!(p.content, "Here you go:");
        assert_eq!(p.calls.len(), 2);
        assert_eq!(args(&p.calls[0]), json!({"x": 1}));
        assert_eq!(args(&p.calls[1]), json!({"y": "two"}));
    }

    #[test]
    fn qwen3_xml_value_types() {
        let text = "<tool_call><function=f><parameter=obj>\n{\"a\":1}\n</parameter><parameter=flag>\ntrue\n</parameter></function></tool_call>";
        let p = parse_output(ToolFormat::Qwen3CoderXml, text, false);
        assert_eq!(args(&p.calls[0]), json!({"obj": {"a": 1}, "flag": true}));
    }

    // --- Generic -------------------------------------------------------------

    #[test]
    fn generic_uses_hermes_blocks() {
        let text = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>";
        let p = parse_output(ToolFormat::Generic, text, false);
        assert_eq!(p.calls.len(), 1);
    }

    #[test]
    fn generic_single_object() {
        let p = parse_output(ToolFormat::Generic, "{\"name\":\"f\",\"arguments\":{\"a\":1}}", false);
        assert_eq!(p.calls.len(), 1);
        assert_eq!(args(&p.calls[0]), json!({"a": 1}));
    }

    #[test]
    fn generic_tool_calls_array() {
        let text = "{\"tool_calls\":[{\"name\":\"f\",\"arguments\":{\"a\":1}},{\"name\":\"g\",\"arguments\":{}}]}";
        let p = parse_output(ToolFormat::Generic, text, false);
        assert_eq!(p.calls.len(), 2);
        assert_eq!(p.calls[1].name, "g");
    }

    #[test]
    fn generic_plain_json_is_content() {
        let text = "{\"answer\": 42}";
        let p = parse_output(ToolFormat::Generic, text, false);
        assert!(p.calls.is_empty());
        assert_eq!(p.content, text);
    }

    // --- Reasoning -----------------------------------------------------------

    #[test]
    fn reasoning_leading_block() {
        let (r, rest) = split_reasoning("\n<think>\nlet me see\n</think>\n\nThe answer is 4.", false);
        assert_eq!(r.as_deref(), Some("let me see"));
        assert_eq!(rest, "The answer is 4.");
    }

    #[test]
    fn reasoning_unclosed_leading_block() {
        let (r, rest) = split_reasoning("<think>partial reasoning", false);
        assert_eq!(r.as_deref(), Some("partial reasoning"));
        assert_eq!(rest, "");
    }

    #[test]
    fn think_tag_mid_text_is_content() {
        // A reply that merely mentions the tag must not be swallowed as reasoning.
        let text = "Wrap it in a <think> tag like this, and close it later.";
        assert_eq!(split_reasoning(text, false), (None, text.to_string()));
        let text = "Sure. <think>not reasoning</think> done";
        assert_eq!(split_reasoning(text, false), (None, text.to_string()));
    }

    #[test]
    fn reasoning_open_in_prompt() {
        let (r, rest) = split_reasoning("hidden chain</think>visible", true);
        assert_eq!(r.as_deref(), Some("hidden chain"));
        assert_eq!(rest, "visible");
        // Opened by the prompt and cut off: all reasoning (matches the stream filter).
        let (r, rest) = split_reasoning("still thinking", true);
        assert_eq!((r.as_deref(), rest.as_str()), (Some("still thinking"), ""));
        let (r2, rest2) = split_reasoning("hidden chain</think>visible", false);
        assert_eq!(r2, None);
        assert_eq!(rest2, "hidden chain</think>visible");
    }

    #[test]
    fn reasoning_absent() {
        let (r, rest) = split_reasoning("just text", false);
        assert_eq!(r, None);
        assert_eq!(rest, "just text");
    }

    #[test]
    fn parse_output_combines_reasoning_and_calls() {
        let text = "<think>need weather</think>Sure!<tool_call>{\"name\":\"w\",\"arguments\":{\"city\":\"Rome\"}}</tool_call>";
        let p = parse_output(ToolFormat::Hermes, text, false);
        assert_eq!(p.reasoning.as_deref(), Some("need weather"));
        assert_eq!(p.content, "Sure!");
        assert_eq!(p.calls.len(), 1);
        assert_eq!(args(&p.calls[0]), json!({"city": "Rome"}));
    }

    // --- Robustness ----------------------------------------------------------

    #[test]
    fn never_panics_on_garbage() {
        let inputs = [
            "",
            "   ",
            "héllo 世界 こんにちは",
            "<tool_call>",
            "<tool_call><function=",
            "<tool_call><function=x><parameter=",
            "[TOOL_CALLS]name[ARGS]",
            "[TOOL_CALLS][{",
            "{\"name\":",
            "\u{0}\u{1}\u{2}",
            "{\"name\":\"x\",\"arguments\":",
        ];
        for fmt in [
            ToolFormat::Hermes,
            ToolFormat::Llama3Json,
            ToolFormat::MistralList,
            ToolFormat::MistralArgs,
            ToolFormat::Qwen3CoderXml,
            ToolFormat::Generic,
        ] {
            for input in inputs {
                let _ = parse_output(fmt, input, true);
            }
        }
    }

    #[test]
    fn tool_call_start_markers() {
        assert_eq!(tool_call_start_marker(ToolFormat::Hermes), Some("<tool_call>"));
        assert_eq!(tool_call_start_marker(ToolFormat::Qwen3CoderXml), Some("<tool_call>"));
        assert_eq!(tool_call_start_marker(ToolFormat::MistralList), Some("[TOOL_CALLS]"));
        assert_eq!(tool_call_start_marker(ToolFormat::MistralArgs), Some("[TOOL_CALLS]"));
        assert_eq!(tool_call_start_marker(ToolFormat::Llama3Json), Some("<|python_tag|>"));
        assert_eq!(tool_call_start_marker(ToolFormat::Generic), Some("<tool_call>"));
    }

    #[test]
    fn missing_arguments_becomes_empty_object() {
        let p = parse_output(ToolFormat::Hermes, "<tool_call>{\"name\":\"f\"}</tool_call>", false);
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].arguments, "{}");
    }
}
