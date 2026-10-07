//! Tool (function) calling: OpenAI-compatible types and grammars that constrain a
//! model's tool calls to its native format with schema-valid arguments.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Value, json};

use crate::tool_parse::ToolFormat;
use crate::{Error, Result};

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
        ToolFormat::Hermes | ToolFormat::Generic => (
            "arguments",
            r#""<tool_call>" [ \n]* root [ \n]* "</tool_call>""#,
            r"(<tool_call>)",
        ),
        // {"name": ..., "parameters": {...}} as the whole reply
        ToolFormat::Llama3Json => ("parameters", "root", r"^\s*(\{)[\s\S]*$"),
        // Control-token / XML formats: parse only.
        ToolFormat::MistralList | ToolFormat::MistralArgs | ToolFormat::Qwen3CoderXml => {
            log::debug!(target: "aicore", "tool calls for {format:?} are parsed but not grammar-constrained");
            return Ok(None);
        }
    };

    let calls: Vec<Value> = selected
        .iter()
        .map(|t| {
            let params = t.function.parameters.clone().unwrap_or_else(|| json!({"type": "object"}));
            json!({
                "type": "object",
                "properties": {"name": {"const": t.function.name}, (args_key): params},
                "required": ["name", args_key],
                "additionalProperties": false
            })
        })
        .collect();
    let schema = if calls.len() == 1 { calls.into_iter().next().unwrap() } else { json!({"anyOf": calls}) };
    let mut text = crate::json_schema::json_schema_to_grammar(&schema).map_err(Error::JsonSchema)?;

    // Llama 3 has no call delimiter, so only a single call is constrained.
    let repeat = parallel && format != ToolFormat::Llama3Json;
    text.push_str(&format!("tool-call ::= {wrapper}\n"));
    text.push_str(if repeat {
        "tool-calls ::= tool-call ([ \\n]* tool-call)*\n"
    } else {
        "tool-calls ::= tool-call\n"
    });
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
}
