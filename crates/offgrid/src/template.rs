//! Jinja chat templates (the `tokenizer.chat_template` stored in GGUF files), rendered
//! with minijinja in Hugging Face `apply_chat_template` style.

use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::value::{Kwargs, Value};
use minijinja::{Environment, ErrorKind};
use serde_json::Value as Json;

use crate::{Error, Result};

const NAME: &str = "chat_template";
/// minijinja fuel per render (roughly instructions executed).
const TEMPLATE_FUEL: u64 = 2_000_000;

/// Values passed to a template besides the messages.
pub(crate) struct RenderArgs<'a> {
    pub messages: &'a Json,
    pub tools: Option<&'a Json>,
    pub add_generation_prompt: bool,
    pub bos_token: &'a str,
    pub eos_token: &'a str,
    /// Extra template variables (e.g. `enable_thinking`), like llama-server's `chat_template_kwargs`.
    pub extra: Option<&'a serde_json::Map<String, Json>>,
}

pub(crate) struct ChatTemplate {
    env: Environment<'static>,
}

/// True if `s` looks like Jinja template text rather than a built-in template name.
pub(crate) fn is_jinja(s: &str) -> bool {
    s.contains("{{") || s.contains("{%")
}

impl ChatTemplate {
    pub fn new(source: &str) -> Result<Self> {
        let mut env = Environment::new();
        // Hugging Face renders with trim_blocks + lstrip_blocks.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        // Python string/list/dict methods: .strip(), .split(), .startswith(), .items(), ...
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Bound the work a template may do (request-supplied templates are untrusted):
        // real chat templates use a few thousand units; a runaway loop fails fast.
        env.set_fuel(Some(TEMPLATE_FUEL));
        env.add_filter("tojson", tojson);
        // Python's `is iterable`: not none and not strings (Llama 3.x would otherwise
        // double-encode a JSON string tool result).
        env.add_test("iterable", |v: Value| {
            use minijinja::value::ValueKind::*;
            matches!(v.kind(), Seq | Map | Iterable | Plain) && v.try_iter().is_ok()
        });
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned(NAME, rewrite_hf_extensions(source)).map_err(|e| Error::ChatTemplate(describe(&e)))?;
        Ok(Self { env })
    }

    pub fn render(&self, args: &RenderArgs) -> Result<String> {
        let mut ctx = serde_json::Map::new();
        if let Some(extra) = args.extra {
            ctx.extend(extra.clone());
        }
        ctx.insert("messages".into(), args.messages.clone());
        ctx.insert("add_generation_prompt".into(), Json::Bool(args.add_generation_prompt));
        ctx.insert("bos_token".into(), Json::String(args.bos_token.into()));
        ctx.insert("eos_token".into(), Json::String(args.eos_token.into()));
        // HF always passes `tools` (None when absent); templates test `tools is not none`.
        ctx.insert("tools".into(), args.tools.cloned().unwrap_or(Json::Null));
        let tmpl = self.env.get_template(NAME).map_err(|e| Error::ChatTemplate(describe(&e)))?;
        tmpl.render(Value::from_serialize(&ctx)).map_err(|e| Error::ChatTemplate(describe(&e)))
    }
}

/// Rewrite Hugging Face-only tags: `{% generation %}` / `{% endgeneration %}` mark
/// assistant spans for training masks and render their body unchanged, so they become
/// `{% if true %}` / `{% endif %}` (keeping `-` whitespace-control markers).
fn rewrite_hf_extensions(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(start) = rest.find("{%") {
        // Search for the closing "%}" after the opening "{%" (they must not overlap: "{%}").
        let Some(close) = rest[start + 2..].find("%}") else { break };
        let len = close + 2;
        let tag = &rest[start..start + len + 2];
        let inner = &tag[2..tag.len() - 2];
        let (lead, trail) = (inner.starts_with('-'), inner.ends_with('-'));
        let word = inner.trim_matches('-').trim();
        let replacement = match word {
            "generation" => Some("if true"),
            "endgeneration" => Some("endif"),
            _ => None,
        };
        out.push_str(&rest[..start]);
        match replacement {
            Some(r) => {
                let _ = write!(out, "{{%{} {r} {}%}}", if lead { "-" } else { "" }, if trail { "-" } else { "" });
            }
            None => out.push_str(tag),
        }
        rest = &rest[start + len + 2..];
    }
    out.push_str(rest);
    out
}

fn describe(e: &minijinja::Error) -> String {
    let mut msg = e.to_string();
    let mut source = std::error::Error::source(e);
    while let Some(s) = source {
        let _ = write!(msg, ": {s}");
        source = s.source();
    }
    msg
}

fn raise_exception(msg: String) -> std::result::Result<Value, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, format!("template raised: {msg}")))
}

/// Python `json.dumps` style `tojson` (minijinja's is compact and HTML-escaped, which
/// differs from what models were trained on).
fn tojson(value: Value, kwargs: Kwargs) -> std::result::Result<Value, minijinja::Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    // Accepted for compatibility; behaviour is fixed to ensure_ascii=False, sort_keys=False.
    let _: Option<bool> = kwargs.get("ensure_ascii")?;
    let _: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;
    let json =
        serde_json::to_value(&value).map_err(|e| minijinja::Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    let mut out = String::new();
    write_py_json(&json, indent, 0, &mut out);
    Ok(Value::from_safe_string(out))
}

/// JSON as Python's `json.dumps` writes it (`", "` and `": "` separators), the way models
/// see JSON in their training data.
pub(crate) fn py_json(v: &Json) -> String {
    let mut out = String::new();
    write_py_json(v, None, 0, &mut out);
    out
}

fn write_py_json(v: &Json, indent: Option<usize>, level: usize, out: &mut String) {
    let newline = |out: &mut String, level: usize| {
        if let Some(n) = indent {
            out.push('\n');
            out.push_str(&" ".repeat(n * level));
        }
    };
    let item_sep = if indent.is_some() { "," } else { ", " };
    match v {
        Json::Array(items) if !items.is_empty() => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                newline(out, level + 1);
                write_py_json(item, indent, level + 1, out);
            }
            newline(out, level);
            out.push(']');
        }
        Json::Object(map) if !map.is_empty() => {
            out.push('{');
            for (i, (k, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                newline(out, level + 1);
                out.push_str(&Json::String(k.clone()).to_string());
                out.push_str(": ");
                write_py_json(item, indent, level + 1, out);
            }
            newline(out, level);
            out.push('}');
        }
        other => out.push_str(&other.to_string()),
    }
}

/// `strftime_now(fmt)` (used by Llama 3.x and others) in UTC. Supports the common
/// directives: %Y %y %m %d %e %H %M %S %b %B %a %A %j %p %I %%.
fn strftime_now(fmt: String) -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    strftime(&fmt, secs)
}

fn strftime(fmt: &str, unix: i64) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const DAYS: [&str; 7] = ["Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"];
    let days = unix.div_euclid(86_400);
    let tod = unix.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let (hour, min, sec) = (tod / 3600, tod / 60 % 60, tod % 60);
    let yday = days - days_from_civil(y, 1, 1) + 1;
    let weekday = DAYS[days.rem_euclid(7) as usize];
    let month = MONTHS[(m - 1) as usize];

    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let _ = match chars.next() {
            Some('Y') => write!(out, "{y}"),
            Some('y') => write!(out, "{:02}", y.rem_euclid(100)),
            Some('m') => write!(out, "{m:02}"),
            Some('d') => write!(out, "{d:02}"),
            Some('e') => write!(out, "{d:2}"),
            Some('H') => write!(out, "{hour:02}"),
            Some('I') => write!(out, "{:02}", if hour % 12 == 0 { 12 } else { hour % 12 }),
            Some('p') => write!(out, "{}", if hour < 12 { "AM" } else { "PM" }),
            Some('M') => write!(out, "{min:02}"),
            Some('S') => write!(out, "{sec:02}"),
            Some('j') => write!(out, "{yday:03}"),
            Some('b') => write!(out, "{}", &month[..3]),
            Some('B') => write!(out, "{month}"),
            Some('a') => write!(out, "{}", &weekday[..3]),
            Some('A') => write!(out, "{weekday}"),
            Some('%') => write!(out, "%"),
            Some(other) => write!(out, "%{other}"),
            None => write!(out, "%"),
        };
    }
    out
}

// Howard Hinnant's civil date algorithms (proleptic Gregorian).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn render(src: &str, messages: Json) -> Result<String> {
        ChatTemplate::new(src)?.render(&RenderArgs {
            messages: &messages,
            tools: None,
            add_generation_prompt: true,
            bos_token: "<s>",
            eos_token: "</s>",
            extra: None,
        })
    }

    #[test]
    fn strftime_known_date() {
        // 2026-10-07 13:05:09 UTC was a Wednesday.
        assert_eq!(strftime("%d %b %Y %H:%M:%S %A %j %%", 1_791_378_309), "07 Oct 2026 13:05:09 Wednesday 280 %");
        assert_eq!(strftime("%Y-%m-%d", 0), "1970-01-01");
    }

    #[test]
    fn tojson_python_style() {
        let data = json!({"a": [1, "é<b>"], "b": {}});
        let out = render("{{ messages | tojson }}", data.clone()).unwrap();
        assert_eq!(out, r#"{"a": [1, "é<b>"], "b": {}}"#);
        let out = render("{{ messages | tojson(indent=2) }}", data).unwrap();
        assert_eq!(out, "{\n  \"a\": [\n    1,\n    \"é<b>\"\n  ],\n  \"b\": {}\n}");
    }

    #[test]
    fn pycompat_and_namespace() {
        let src = "{% set ns = namespace(n=0) %}{% for m in messages[::-1] %}{% set ns.n = ns.n + 1 %}\
                   {{ m.content.strip().split(',')|first }}{% endfor %}{{ ns.n }}";
        let out = render(src, json!([{"role": "user", "content": " a,b "}, {"role": "user", "content": "c"}])).unwrap();
        assert_eq!(out, "ca2");
    }

    #[test]
    fn raise_exception_is_error() {
        let err = render("{{ raise_exception('nope') }}", json!([])).unwrap_err();
        assert!(err.to_string().contains("nope"), "{err}");
    }

    /// Every template in llama.cpp's corpus must render a plain conversation.
    #[test]
    fn llama_cpp_template_corpus() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../offgrid-sys/llama.cpp/models/templates");
        let messages = json!([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "What is 2+2?"}
        ]);
        let mut entries: Vec<_> = std::fs::read_dir(&dir).expect("template corpus").flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        let (mut ok, mut failed) = (0, Vec::new());
        for entry in entries {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "jinja") {
                continue;
            }
            let src = std::fs::read_to_string(&path).unwrap();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            match render(&src, messages.clone()) {
                Ok(out) if out.contains("What is 2+2?") => ok += 1,
                Ok(out) => failed.push(format!("{name}: output lacks last message: {out:?}")),
                Err(e) => failed.push(format!("{name}: {e}")),
            }
        }
        // Known, legitimate refusals with this generic input:
        // gemma-2 raises "System role not supported"; firefunction needs `functions`/`datetime`
        // vars; IFM-K2-Horizon requires a reasoning field on every assistant message.
        let expected =
            ["google-gemma-2-2b-it.jinja", "fireworks-ai-llama-3-firefunction-v2.jinja", "IFM-K2-Horizon.jinja"];
        failed.retain(|f| !expected.iter().any(|e| f.starts_with(e)));
        for f in &failed {
            println!("FAIL {f}");
        }
        println!("{ok} rendered, {} unexpected failures", failed.len());
        assert!(failed.is_empty());
        assert!(ok >= 69, "corpus shrank? only {ok} rendered");
    }

    #[test]
    fn iterable_test_lists_not_strings() {
        let out = render(
            "{{ none is iterable }} {{ [] is iterable }} {{ 'ab' is iterable }} {{ {} is iterable }} {{ 3 is iterable }}",
            json!([]),
        )
        .unwrap();
        assert_eq!(out, "False True False True False");
    }

    #[test]
    fn malformed_tags_do_not_panic() {
        for src in ["{%}", "{%%}", "a{%", "{% generation", "{%-}", "%}{%"] {
            let _ = rewrite_hf_extensions(src);
            let _ = ChatTemplate::new(src); // error or Ok, never a panic
        }
        assert_eq!(rewrite_hf_extensions("{%}x"), "{%}x");
    }

    #[test]
    fn generation_tags_rewritten() {
        assert_eq!(rewrite_hf_extensions("a{%- generation -%}b{% endgeneration %}c"), "a{%- if true -%}b{% endif %}c");
        let out = render(
            "{% for m in messages %}{% generation %}{{ m.content }}{% endgeneration %}{% endfor %}",
            json!([{"role": "user", "content": "x"}]),
        )
        .unwrap();
        assert_eq!(out, "x");
    }

    #[test]
    fn runaway_template_is_stopped() {
        let t = std::time::Instant::now();
        let err = render("{% for i in range(99999) %}{% for j in range(99999) %}{% endfor %}{% endfor %}", json!([]))
            .unwrap_err();
        assert!(err.to_string().contains("fuel"), "{err}");
        assert!(t.elapsed() < std::time::Duration::from_secs(5));
    }
}
