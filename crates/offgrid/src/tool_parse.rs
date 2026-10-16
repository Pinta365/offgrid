//! Tool-call output parsing for the supported chat templates (pure, no llama.cpp).

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
    /// Hermes-style `<tool_call>` blocks described in the system message, for templates
    /// that ignore `tools` (see `tools::polyfill_messages`).
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

/// Texts that begin a tool call, used to stop streaming content.
pub(crate) fn tool_call_start_markers(format: ToolFormat) -> &'static [&'static str] {
    match format {
        ToolFormat::Hermes => &["<tool_call>"],
        // Qwen3-Coder sometimes leaves out `<tool_call>` and starts with the function block.
        ToolFormat::Qwen3CoderXml => &["<tool_call>", "<function="],
        ToolFormat::MistralList | ToolFormat::MistralArgs => &["[TOOL_CALLS]"],
        ToolFormat::Llama3Json => &["<|python_tag|>"],
        ToolFormat::Generic | ToolFormat::Polyfill => &["<tool_call>"],
    }
}

/// Split off a reasoning block that starts the reply (or was opened by the prompt) and runs
/// to `</think>` or the end. Must match `text::StreamFilter` so streamed and final agree.
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
    Parsed { content: content.trim().to_string(), reasoning, calls }
}

fn parse_calls(format: ToolFormat, text: &str) -> (String, Vec<ParsedCall>) {
    let (content, calls) = match format {
        ToolFormat::Hermes | ToolFormat::Polyfill => parse_hermes(text),
        ToolFormat::Llama3Json => parse_json_calls(text),
        ToolFormat::MistralList => parse_mistral_list(text),
        ToolFormat::MistralArgs => parse_mistral_args(text),
        ToolFormat::Qwen3CoderXml => parse_qwen3_xml(text),
        ToolFormat::Generic => parse_generic(text),
    };
    if !calls.is_empty() {
        return (strip_empty_fence(&content), calls);
    }
    // Models in any format sometimes drop their markup and reply with bare call objects, or
    // wrap calls in a Markdown code fence.
    let (content, calls) = parse_json_calls(text);
    if !calls.is_empty() {
        return (content, calls);
    }
    parse_fenced(text).unwrap_or((content, calls))
}

/// Info strings of code fences that may hold tool calls: ```tool_call, ```tool_code, ```json,
/// a bare ```, and the malformed ```tool_call> some models write.
pub(crate) fn is_call_fence(line: &str) -> bool {
    line.trim()
        .strip_prefix("```")
        .is_some_and(|info| matches!(info.trim(), "" | "tool_call" | "tool_call>" | "tool_code" | "json"))
}

/// Content left after taking tagged calls out of a call fence: drop the empty fence.
fn strip_empty_fence(content: &str) -> String {
    match content.find("```") {
        Some(start)
            if content[start..].lines().all(|l| l.trim().is_empty() || l.trim() == "```" || is_call_fence(l)) =>
        {
            content[..start].trim().to_string()
        }
        _ => content.to_string(),
    }
}

/// Calls inside a code fence (closed by ```, by `</tool_call>`, or cut off); text before the
/// fence is content.
fn parse_fenced(text: &str) -> Option<(String, Vec<ParsedCall>)> {
    let start = text.find("```")?;
    let newline = start + text[start..].find('\n')?;
    if !is_call_fence(&text[start..newline]) {
        return None;
    }
    let body = &text[newline + 1..];
    let end = ["```", "</tool_call>"].iter().filter_map(|close| body.find(close)).min().unwrap_or(body.len());
    let inner = &body[..end];
    let (_, calls) = if inner.contains("<tool_call>") { parse_hermes(inner) } else { parse_json_calls(inner) };
    (!calls.is_empty()).then(|| (text[..start].trim().to_string(), calls))
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

/// `{"name", "arguments"|"parameters"}`, also with the name in `"function"` (a string, or an
/// object holding name and arguments), as small models write it.
fn call_from_object(obj: &Map<String, Value>) -> Option<ParsedCall> {
    let inner = obj.get("function").and_then(Value::as_object);
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| obj.get("function").and_then(Value::as_str))
        .or_else(|| inner.and_then(|f| f.get("name")).and_then(Value::as_str))?
        .to_string();
    let args = obj
        .get("arguments")
        .or_else(|| obj.get("parameters"))
        .or_else(|| inner.and_then(|f| f.get("arguments").or_else(|| f.get("parameters"))));
    Some(ParsedCall { name, arguments: normalize_arguments(args) })
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
    let (content, calls) = parse_tagged(text, "<tool_call>", "</tool_call>", parse_xml_function);
    if !calls.is_empty() {
        return (content, calls);
    }
    // Function blocks without the `<tool_call>` wrapper, sometimes with a stray `</tool_call>`.
    // Only complete blocks count, so text that mentions `<function=` stays text.
    const OPEN: &str = "<function=";
    const CLOSE: &str = "</function>";
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find(OPEN) {
        let Some(len) = rest[pos..].find(CLOSE) else { break };
        match parse_xml_function(&rest[pos..pos + len]) {
            Some(call) => {
                content.push_str(&rest[..pos]);
                calls.push(call);
                rest = &rest[pos + len + CLOSE.len()..];
            }
            None => {
                content.push_str(&rest[..pos + OPEN.len()]);
                rest = &rest[pos + OPEN.len()..];
            }
        }
    }
    content.push_str(rest);
    if calls.is_empty() {
        return (text.to_string(), calls);
    }
    (content.replace("</tool_call>", ""), calls)
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
    Some(ParsedCall { name: name.to_string(), arguments: Value::Object(map).to_string() })
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

/// A reply made of bare call objects (optionally after Llama's `<|python_tag|>`), separated by
/// `;` or whitespace. Stops at the first value that isn't a complete call, so a reply cut off
/// by `max_tokens` still yields the calls before it.
fn parse_json_calls(text: &str) -> (String, Vec<ParsedCall>) {
    let trimmed = text.trim();
    let mut rest = trimmed.strip_prefix("<|python_tag|>").unwrap_or(trimmed);
    let mut calls = Vec::new();
    loop {
        rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ';');
        if !rest.starts_with('{') {
            break;
        }
        let mut values = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
        let Some(Ok(Value::Object(obj))) = values.next() else { break };
        let Some(call) = call_from_object(&obj) else { break };
        calls.push(call);
        rest = &rest[values.byte_offset()..];
    }
    if calls.is_empty() { (text.to_string(), calls) } else { (String::new(), calls) }
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
                && let Some(c) = call_from_object(obj)
            {
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
                calls.push(ParsedCall { name: name.to_string(), arguments: normalize_arguments(Some(&v)) });
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
    if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(trimmed)
        && let Some(Value::Array(arr)) = obj.get("tool_calls")
    {
        let mut calls = Vec::new();
        for item in arr {
            if let Some(o) = item.as_object()
                && let Some(c) = call_from_object(o)
            {
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

    /// llama.cpp's chat template corpus (in the offgrid-sys crate).
    fn templates_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../offgrid-sys/llama.cpp/models/templates")
    }

    fn template(name: &str) -> String {
        let p = templates_dir().join(name);
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
    }

    fn args(call: &ParsedCall) -> Value {
        serde_json::from_str(&call.arguments).expect("arguments must be JSON")
    }

    #[test]
    fn detect_qwen3_is_hermes() {
        assert_eq!(detect_format(&template("Qwen-Qwen3-0.6B.jinja")), ToolFormat::Hermes);
    }

    #[test]
    fn detect_llama32_is_llama3json() {
        assert_eq!(detect_format(&template("meta-llama-Llama-3.2-3B-Instruct.jinja")), ToolFormat::Llama3Json);
    }

    #[test]
    fn detect_qwen3_coder_is_xml() {
        assert_eq!(detect_format(&template("Qwen3-Coder.jinja")), ToolFormat::Qwen3CoderXml);
    }

    #[test]
    fn detect_mistral_nemo_is_list() {
        assert_eq!(detect_format(&template("mistralai-Mistral-Nemo-Instruct-2407.jinja")), ToolFormat::MistralList);
    }

    #[test]
    fn detect_devstral_is_args() {
        assert_eq!(detect_format(&template("unsloth-mistral-Devstral-Small-2507.jinja")), ToolFormat::MistralArgs);
    }

    #[test]
    fn detect_plain_is_generic() {
        assert_eq!(detect_format("You are a helpful assistant."), ToolFormat::Generic);
    }

    #[test]
    fn hermes_single_call() {
        let text =
            "Hello\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
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
    fn qwen3_xml_without_tool_call_wrapper() {
        // Seen from Qwen3-Coder-30B-A3B: no opening <tool_call>, a stray closing one.
        let text = "I need to find the default.\n\n<function=grep_search>\n<parameter=query>\nchatter.*threshold\n</parameter>\n</function>\n</tool_call>";
        let p = parse_output(ToolFormat::Qwen3CoderXml, text, false);
        assert_eq!(p.content, "I need to find the default.");
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "grep_search");
        assert_eq!(args(&p.calls[0]), json!({"query": "chatter.*threshold"}));
        // Plain text that mentions <function= is not a call.
        let p = parse_output(ToolFormat::Qwen3CoderXml, "Write <function=name> blocks.", false);
        assert!(p.calls.is_empty());
        assert_eq!(p.content, "Write <function=name> blocks.");
    }

    #[test]
    fn qwen3_xml_value_types() {
        let text = "<tool_call><function=f><parameter=obj>\n{\"a\":1}\n</parameter><parameter=flag>\ntrue\n</parameter></function></tool_call>";
        let p = parse_output(ToolFormat::Qwen3CoderXml, text, false);
        assert_eq!(args(&p.calls[0]), json!({"obj": {"a": 1}, "flag": true}));
    }

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
    fn llama3_python_tag_shapes_and_truncation() {
        let text = concat!(
            "<|python_tag|>{\"type\": \"function\", \"function\": \"a\", \"parameters\": {\"x\": \"1; 2\"}}; ",
            "{\"type\": \"function\", \"name\": \"b\", \"parameters\": {}}; ",
            "{\"type\": \"function\", \"function\": {\"name\": \"c\", \"arguments\": {\"y\": 2}}}; ",
            "{\"type\": \"function\", \"name\": \"d\", \"parameters\": {\"z\": \"cut off"
        );
        let p = parse_output(ToolFormat::Llama3Json, text, false);
        let names: Vec<&str> = p.calls.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "c"], "the truncated last call is dropped");
        assert_eq!(args(&p.calls[0]), json!({"x": "1; 2"}));
        assert_eq!(args(&p.calls[2]), json!({"y": 2}));
        assert!(p.content.is_empty());
    }

    #[test]
    fn bare_call_fallback_in_any_format() {
        let text = "{\"name\": \"f\", \"arguments\": {\"a\": 1}}";
        for format in [ToolFormat::Hermes, ToolFormat::Qwen3CoderXml, ToolFormat::MistralList, ToolFormat::Generic] {
            let p = parse_output(format, text, false);
            assert_eq!(p.calls.len(), 1, "{format:?}");
            assert!(p.content.is_empty());
        }
        let p = parse_output(ToolFormat::Hermes, "Just text {not a call}", false);
        assert!(p.calls.is_empty());
        assert_eq!(p.content, "Just text {not a call}");
    }

    #[test]
    fn calls_in_code_fences() {
        let call = r#"{"name": "f", "arguments": {"a": 1}}"#;
        let cases = [
            format!("```tool_call\n{call}\n```"),
            format!("```tool_call>\n{call}\n</tool_call>"),
            format!("Sure.\n```json\n{call}\n{call}\n```"),
            format!("```\n<tool_call>{call}</tool_call>\n```"),
            format!("```tool_code\n{call}"),
        ];
        for (i, text) in cases.iter().enumerate() {
            let p = parse_output(ToolFormat::Polyfill, text, false);
            assert_eq!(p.calls.len(), if i == 2 { 2 } else { 1 }, "{text}");
            assert_eq!(args(&p.calls[0]), json!({"a": 1}));
            assert_eq!(p.content, if i == 2 { "Sure." } else { "" });
        }
        let code = "```python\nprint({\"name\": \"f\"})\n```";
        assert!(parse_output(ToolFormat::Hermes, code, false).calls.is_empty());
    }

    #[test]
    fn polyfill_tagged_and_bare() {
        let tagged = parse_output(
            ToolFormat::Polyfill,
            "<tool_call>{\"name\":\"f\",\"arguments\":{\"a\":1}}</tool_call>",
            false,
        );
        let bare = parse_output(ToolFormat::Polyfill, "{\"name\": \"f\", \"arguments\": {\"a\": 1}}", false);
        for p in [tagged, bare] {
            assert_eq!(p.calls.len(), 1);
            assert_eq!(args(&p.calls[0]), json!({"a": 1}));
            assert!(p.content.is_empty());
        }
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
        let text =
            "<think>need weather</think>Sure!<tool_call>{\"name\":\"w\",\"arguments\":{\"city\":\"Rome\"}}</tool_call>";
        let p = parse_output(ToolFormat::Hermes, text, false);
        assert_eq!(p.reasoning.as_deref(), Some("need weather"));
        assert_eq!(p.content, "Sure!");
        assert_eq!(p.calls.len(), 1);
        assert_eq!(args(&p.calls[0]), json!({"city": "Rome"}));
    }

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
    fn start_markers_per_format() {
        assert_eq!(tool_call_start_markers(ToolFormat::Hermes), ["<tool_call>"]);
        assert_eq!(tool_call_start_markers(ToolFormat::Qwen3CoderXml), ["<tool_call>", "<function="]);
        assert_eq!(tool_call_start_markers(ToolFormat::MistralList), ["[TOOL_CALLS]"]);
        assert_eq!(tool_call_start_markers(ToolFormat::MistralArgs), ["[TOOL_CALLS]"]);
        assert_eq!(tool_call_start_markers(ToolFormat::Llama3Json), ["<|python_tag|>"]);
        assert_eq!(tool_call_start_markers(ToolFormat::Generic), ["<tool_call>"]);
    }

    #[test]
    fn missing_arguments_becomes_empty_object() {
        let p = parse_output(ToolFormat::Hermes, "<tool_call>{\"name\":\"f\"}</tool_call>", false);
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].arguments, "{}");
    }
}
