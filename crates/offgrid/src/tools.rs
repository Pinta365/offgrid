//! Tool (function) calling: OpenAI-compatible types and grammars that constrain a
//! model's tool calls to its native format with schema-valid arguments.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Value, json};

use crate::template::py_json;
use crate::tool_parse::ToolFormat;
use crate::{ChatMessage, Error, Result};

/// A tool the model may call: `{"type":"function","function":{"name","description","parameters"}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: FunctionDef,
}

impl Tool {
    pub fn function(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            kind: function_type(),
            function: FunctionDef { name: name.into(), description: Some(description.into()), parameters: Some(parameters) },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema of the arguments object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

/// A call the model made: `{"id","type":"function","function":{"name","arguments"}}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// Arguments as JSON text (OpenAI convention).
    pub arguments: String,
}

impl ToolCall {
    pub(crate) fn new(name: String, arguments: String) -> Self {
        Self { id: new_call_id(), kind: function_type(), function: FunctionCall { name, arguments } }
    }
}

fn function_type() -> String {
    "function".into()
}

/// `"none" | "auto" | "required" | {"type":"function","function":{"name":"..."}}`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ToolChoice {
    /// Don't call tools (tools are not shown to the model).
    None,
    /// Model decides; calls are grammar-checked once it starts one.
    #[default]
    Auto,
    /// Must call at least one tool.
    Required,
    /// Must call this tool.
    Function(String),
}

impl Serialize for ToolChoice {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            ToolChoice::None => s.serialize_str("none"),
            ToolChoice::Auto => s.serialize_str("auto"),
            ToolChoice::Required => s.serialize_str("required"),
            ToolChoice::Function(name) => json!({"type": "function", "function": {"name": name}}).serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Named {
            name: String,
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Mode(String),
            Function { function: Named },
        }
        match Repr::deserialize(d)? {
            Repr::Mode(m) => match m.as_str() {
                "none" => Ok(ToolChoice::None),
                "auto" => Ok(ToolChoice::Auto),
                "required" => Ok(ToolChoice::Required),
                other => Err(serde::de::Error::custom(format!("unknown tool_choice '{other}'"))),
            },
            Repr::Function { function } => Ok(ToolChoice::Function(function.name)),
        }
    }
}

/// `call_` + 24 alphanumerics, unique within the process.
fn new_call_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    // RandomState is randomly keyed (std's OS-seeded keys); hashing a process-wide counter
    // with two such states gives 128 unpredictable, collision-free-in-practice bits.
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let half = |n: u64| {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(n);
        h.finish() as u128
    };
    let mut x = half(n) << 64 | half(n);
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut id = String::from("call_");
    for _ in 0..24 {
        // 62^24 exceeds 2^128, so the last two or three chars are usually 'A'; fine.
        id.push(ALPHABET[(x % 62) as usize] as char);
        x /= 62;
    }
    id
}

/// Rewrite a conversation for a chat template that ignores `tools` (e.g. Gemma 3), the way
/// llama.cpp does for such templates: describe `tools` (if offered) in the system message,
/// or at the start of the first user message when `system_role` is false, and turn
/// assistant tool calls and tool results into plain text in the `<tool_call>` format that
/// the generic grammar constrains. Consecutive tool results become one user turn, so roles
/// still alternate.
pub(crate) fn polyfill_messages(messages: &[ChatMessage], tools: Option<&[Tool]>, system_role: bool) -> Vec<ChatMessage> {
    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len() + 1);
    // Call id → function name, to label tool results.
    let mut names: Vec<(&str, &str)> = Vec::new();
    let mut prev_tool = false;
    for m in messages {
        if m.role == "tool" {
            let name = m.tool_call_id.as_deref().and_then(|id| names.iter().find(|(i, _)| *i == id)).map(|(_, n)| *n);
            let content = serde_json::from_str::<Value>(&m.content).unwrap_or_else(|_| Value::String(m.content.clone()));
            let text = match name {
                Some(name) => format!(
                    "<tool_response>\n{{\"name\": {}, \"content\": {}}}\n</tool_response>",
                    py_json(&json!(name)),
                    py_json(&content)
                ),
                None => format!("<tool_response>\n{{\"content\": {}}}\n</tool_response>", py_json(&content)),
            };
            match out.last_mut() {
                Some(last) if prev_tool => {
                    last.content.push('\n');
                    last.content.push_str(&text);
                    last.media.extend(m.media.iter().cloned());
                }
                _ => out.push(ChatMessage { media: m.media.clone(), ..ChatMessage::user(text) }),
            }
            prev_tool = true;
            continue;
        }
        prev_tool = false;
        if m.tool_calls.is_empty() {
            out.push(m.clone());
            continue;
        }
        let mut text = m.content.trim().to_string();
        for call in &m.tool_calls {
            names.push((&call.id, &call.function.name));
            let args = serde_json::from_str::<Value>(&call.function.arguments).unwrap_or_else(|_| json!({}));
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&format!(
                "<tool_call>\n{{\"name\": {}, \"arguments\": {}}}\n</tool_call>",
                py_json(&json!(call.function.name)),
                py_json(&args)
            ));
        }
        out.push(ChatMessage { content: text, tool_calls: Vec::new(), ..m.clone() });
    }

    if let Some(tools) = tools.filter(|t| !t.is_empty()) {
        let instructions = tool_instructions(tools);
        match out.first_mut() {
            Some(first) if first.role == "system" => first.content = format!("{}\n\n{instructions}", first.content.trim_end()),
            _ => out.insert(0, ChatMessage::system(instructions)),
        }
    }
    if !system_role && out.first().is_some_and(|m| m.role == "system") {
        // Fold the system message into the first user turn.
        let system = out.remove(0);
        match out.iter_mut().find(|m| m.role == "user") {
            Some(user) => {
                user.content = format!("{}\n\n{}", system.content, user.content);
                user.media.splice(0..0, system.media);
            }
            None => out.insert(0, ChatMessage { role: "user".into(), ..system }),
        }
    }
    out
}

/// Ends generation in the polyfill format: the model starts writing a tool result itself.
pub(crate) const POLYFILL_STOP: &str = "<tool_response>";

/// System-prompt text that shows `tools` and the call format (Hermes style).
fn tool_instructions(tools: &[Tool]) -> String {
    let list: Vec<String> = tools.iter().map(|t| py_json(&serde_json::to_value(t).unwrap_or_default())).collect();
    format!(
        "# Tools\n\n\
         You may call one or more functions to help with the user's request. The available functions:\n\
         <tools>\n{}\n</tools>\n\n\
         To call a function, reply with one block per call, in exactly this format:\n\
         <tool_call>\n{{\"name\": <function name>, \"arguments\": <arguments as a JSON object>}}\n</tool_call>\n\n\
         The results come back in <tool_response></tool_response> blocks. Only call a function when \
         you need what it provides; otherwise answer directly. Never make up a function's result.",
        list.join("\n")
    )
}

/// A grammar plus the rule to start from and, if lazy, the regexes that activate it.
#[derive(Debug, Clone)]
pub(crate) struct GrammarSpec {
    pub text: String,
    pub root: String,
    /// Empty = enforce from the first token. Otherwise the grammar only applies from the
    /// first capture group of the first matching pattern (llama.cpp lazy grammars).
    pub lazy_patterns: Vec<String>,
}

impl GrammarSpec {
    pub fn strict(text: String) -> Self {
        Self { text, root: "root".into(), lazy_patterns: Vec::new() }
    }
}

/// Grammar constraining tool calls for `format`, or `None` when the format/choice
/// isn't grammar-constrained (calls are then only parsed).
pub(crate) fn tool_grammar(
    format: ToolFormat,
    tools: &[Tool],
    choice: &ToolChoice,
    parallel: bool,
) -> Result<Option<GrammarSpec>> {
    let selected: Vec<&Tool> = match choice {
        ToolChoice::None => return Ok(None),
        ToolChoice::Function(name) => {
            let t = tools
                .iter()
                .find(|t| &t.function.name == name)
                .ok_or_else(|| Error::InvalidArgument(format!("tool_choice names unknown tool '{name}'")))?;
            vec![t]
        }
        ToolChoice::Auto | ToolChoice::Required => tools.iter().collect(),
    };
    if selected.is_empty() {
        return Ok(None);
    }
    let lazy = *choice == ToolChoice::Auto;
    let (args_key, wrapper, trigger) = match format {
        // <tool_call>{"name": ..., "arguments": {...}}</tool_call>
        ToolFormat::Hermes | ToolFormat::Generic | ToolFormat::Polyfill => (
            "arguments",
            r#""<tool_call>" [ \n]* root [ \n]* "</tool_call>""#,
            r"(<tool_call>)",
        ),
        // {"name": ..., "parameters": {...}} as the whole reply
        ToolFormat::Llama3Json => ("parameters", "root", r"^\s*(\{)[\s\S]*$"),
        // Control-token / XML formats: parse only.
        ToolFormat::MistralList | ToolFormat::MistralArgs | ToolFormat::Qwen3CoderXml => {
            log::debug!(target: "offgrid", "tool calls for {format:?} are parsed but not grammar-constrained");
            return Ok(None);
        }
    };

    // Parameter schemas are nested inside a call wrapper, which becomes the root that local
    // `#/$defs/...` refs resolve against, so their definitions move up to the wrapper root.
    let mut defs: serde_json::Map<String, Value> = serde_json::Map::new();
    let mut calls: Vec<Value> = Vec::with_capacity(selected.len());
    for t in &selected {
        let mut params = t.function.parameters.clone().unwrap_or_else(|| json!({"type": "object"}));
        if let Value::Object(map) = &mut params {
            for key in ["$defs", "definitions"] {
                let Some(Value::Object(local)) = map.remove(key) else { continue };
                let section = defs.entry(key.to_string()).or_insert_with(|| json!({}));
                for (name, def) in local {
                    match section.get(&name) {
                        Some(existing) if *existing != def => {
                            return Err(Error::InvalidArgument(format!(
                                "tools define different schemas named {key}/{name}; rename one"
                            )));
                        }
                        _ => {
                            section[&name] = def;
                        }
                    }
                }
            }
        }
        calls.push(json!({
            "type": "object",
            "properties": {"name": {"const": t.function.name}, (args_key): params},
            "required": ["name", args_key],
            "additionalProperties": false
        }));
    }
    let mut schema = if calls.len() == 1 { calls.into_iter().next().unwrap() } else { json!({"anyOf": calls}) };
    if let Value::Object(root) = &mut schema {
        root.extend(defs);
    }
    let mut text = crate::json_schema::json_schema_to_grammar(&schema).map_err(Error::JsonSchema)?;

    // Llama 3 has no call delimiter, so only a single call is constrained.
    let repeat = parallel && format != ToolFormat::Llama3Json;
    text.push_str(&format!("tool-call ::= {wrapper}\n"));
    text.push_str(if repeat {
        "tool-calls ::= tool-call ([ \\n]* tool-call)*"
    } else {
        "tool-calls ::= tool-call"
    });
    // Models taught the format in the prompt tend to write the tool's response themselves
    // next. Allow its opening tag, which is a stop string (`POLYFILL_STOP`), so the turn
    // ends there instead of being forced into more calls.
    if format == ToolFormat::Polyfill {
        text.push_str(&format!(" ([ \\n]* \"{POLYFILL_STOP}\")?"));
    }
    text.push('\n');
    Ok(Some(GrammarSpec {
        text,
        root: "tool-calls".into(),
        lazy_patterns: if lazy { vec![trigger.into()] } else { Vec::new() },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather() -> Tool {
        Tool::function(
            "get_weather",
            "Weather for a city",
            json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        )
    }

    #[test]
    fn tool_choice_serde() {
        for (j, c) in [
            (json!("none"), ToolChoice::None),
            (json!("auto"), ToolChoice::Auto),
            (json!("required"), ToolChoice::Required),
            (json!({"type": "function", "function": {"name": "f"}}), ToolChoice::Function("f".into())),
        ] {
            assert_eq!(serde_json::from_value::<ToolChoice>(j.clone()).unwrap(), c);
            assert_eq!(serde_json::to_value(&c).unwrap(), j);
        }
        assert!(serde_json::from_value::<ToolChoice>(json!("sometimes")).is_err());
    }

    #[test]
    fn tool_wire_format() {
        let t: Tool = serde_json::from_value(json!({"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}})).unwrap();
        assert_eq!(t.function.name, "f");
        assert_eq!(serde_json::to_value(&t).unwrap()["type"], "function");
    }

    #[test]
    fn call_ids_unique() {
        let ids: std::collections::HashSet<String> = (0..1000).map(|_| new_call_id()).collect();
        assert_eq!(ids.len(), 1000);
        assert!(ids.iter().all(|id| id.len() == 29 && id.starts_with("call_")));
    }

    #[test]
    fn hermes_grammar_lazy_vs_strict() {
        let tools = [weather()];
        let g = tool_grammar(ToolFormat::Hermes, &tools, &ToolChoice::Auto, true).unwrap().unwrap();
        assert_eq!(g.root, "tool-calls");
        assert_eq!(g.lazy_patterns, vec!["(<tool_call>)".to_string()]);
        assert!(g.text.contains("tool-call ::= \"<tool_call>\""));
        assert!(g.text.contains("get_weather"));
        let g = tool_grammar(ToolFormat::Hermes, &tools, &ToolChoice::Required, false).unwrap().unwrap();
        assert!(g.lazy_patterns.is_empty());
        assert!(g.text.contains("tool-calls ::= tool-call\n"));
    }

    #[test]
    fn named_choice_and_none() {
        let tools = [weather(), Tool::function("noop", "x", json!({"type": "object"}))];
        let g = tool_grammar(ToolFormat::Llama3Json, &tools, &ToolChoice::Function("noop".into()), true).unwrap().unwrap();
        assert!(g.text.contains("noop") && !g.text.contains("get_weather"));
        assert!(g.text.contains("parameters"));
        assert!(tool_grammar(ToolFormat::Hermes, &tools, &ToolChoice::None, true).unwrap().is_none());
        assert!(tool_grammar(ToolFormat::Hermes, &tools, &ToolChoice::Function("zzz".into()), true).is_err());
        assert!(tool_grammar(ToolFormat::MistralList, &tools, &ToolChoice::Required, true).unwrap().is_none());
    }

    #[test]
    fn tool_schema_defs_are_hoisted() {
        // Pydantic-style schema with local $defs references.
        let tool = Tool::function(
            "save",
            "x",
            json!({"$defs": {"Addr": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
                   "type": "object", "properties": {"a": {"$ref": "#/$defs/Addr"}}, "required": ["a"]}),
        );
        let g = tool_grammar(ToolFormat::Hermes, std::slice::from_ref(&tool), &ToolChoice::Required, false).unwrap().unwrap();
        assert!(g.text.contains("city"), "{}", g.text);
        // Two tools sharing an identical definition is fine; conflicting ones are an error.
        let other = Tool::function("other", "y", json!({"$defs": {"Addr": {"type": "string"}}, "type": "object",
            "properties": {"b": {"$ref": "#/$defs/Addr"}}}));
        assert!(tool_grammar(ToolFormat::Hermes, &[tool.clone(), tool.clone()], &ToolChoice::Auto, true).is_ok());
        assert!(tool_grammar(ToolFormat::Hermes, &[tool, other], &ToolChoice::Auto, true).is_err());
    }

    fn call(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall { id: id.into(), kind: "function".into(), function: FunctionCall { name: name.into(), arguments: args.into() } }
    }

    #[test]
    fn polyfill_writes_tools_calls_and_results_as_text() {
        let a = call("call_a", "get_weather", r#"{"city":"Oslo"}"#);
        let b = call("call_b", "get_weather", r#"{"city":"Rome"}"#);
        let messages = vec![
            ChatMessage::system("Be brief."),
            ChatMessage::user("Weather in Oslo and Rome?"),
            ChatMessage::assistant_tool_calls("Checking.", vec![a.clone(), b.clone()]),
            ChatMessage::tool_result(&a, r#"{"temp_c": 4}"#),
            ChatMessage::tool_result(&b, "sunny"),
        ];
        let out = polyfill_messages(&messages, Some(&[weather()]), true);
        let roles: Vec<&str> = out.iter().map(|m| m.role.as_str()).collect();
        // Both results in one user turn, so roles still alternate.
        assert_eq!(roles, ["system", "user", "assistant", "user"]);
        assert!(out[0].content.starts_with("Be brief.\n\n# Tools"), "{}", out[0].content);
        assert!(out[0].content.contains(r#""name": "get_weather""#), "{}", out[0].content);
        assert_eq!(
            out[2].content,
            "Checking.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call>\n\
             <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call>"
        );
        assert!(out[2].tool_calls.is_empty());
        assert_eq!(
            out[3].content,
            "<tool_response>\n{\"name\": \"get_weather\", \"content\": {\"temp_c\": 4}}\n</tool_response>\n\
             <tool_response>\n{\"name\": \"get_weather\", \"content\": \"sunny\"}\n</tool_response>"
        );
    }

    #[test]
    fn polyfill_without_offered_tools_only_rewrites_history() {
        // tool_choice none (e.g. run_tools' last round): history is rewritten, tools not shown.
        let a = call("call_a", "get_weather", "{}");
        let messages = vec![ChatMessage::user("Hi"), ChatMessage::assistant_tool_calls("", vec![a.clone()]), ChatMessage::tool_result(&a, "ok")];
        let out = polyfill_messages(&messages, None, true);
        assert_eq!(out.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(), ["user", "assistant", "user"]);
        assert!(!out.iter().any(|m| m.content.contains("# Tools")));
    }

    #[test]
    fn polyfill_folds_system_into_user_without_system_role() {
        let out = polyfill_messages(&[ChatMessage::system("Be brief."), ChatMessage::user("Hi")], Some(&[weather()]), false);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, "user");
        assert!(out[0].content.starts_with("Be brief.\n\n# Tools") && out[0].content.ends_with("\n\nHi"), "{}", out[0].content);

        let out = polyfill_messages(&[ChatMessage::user("Hi")], Some(&[weather()]), false);
        assert_eq!(out.len(), 1);
        assert!(out[0].content.starts_with("# Tools") && out[0].content.ends_with("\n\nHi"));
    }
}
