//! JSON Schema -> GBNF grammar, ported from llama.cpp's `common/json-schema-to-grammar.cpp`.
//!
//! Not supported: remote `$ref`, regex lookaround/named groups/anchors in `pattern`;
//! numeric minimum/maximum are ignored.

use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};

const SPACE_RULE: &str = r#"| " " | "\n"{1,2} [ \t]{0,20}"#;
const MAX_PATTERN_DEPTH: usize = 100;
const NO_MAX: i64 = i32::MAX as i64;

fn primitive_rule(name: &str) -> Option<(&'static str, &'static [&'static str])> {
    let rule: (&'static str, &'static [&'static str]) = match name {
        "boolean" => (r#"("true" | "false")"#, &[]),
        "decimal-part" => (r#"[0-9]{1,16}"#, &[]),
        "integral-part" => (r#"[0] | [1-9] [0-9]{0,15}"#, &[]),
        "number" => (
            r#"("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?"#,
            &["integral-part", "decimal-part"],
        ),
        "integer" => (r#"("-"? integral-part)"#, &["integral-part"]),
        "value" => (
            r#"object | array | string | number | boolean | null"#,
            &["object", "array", "string", "number", "boolean", "null"],
        ),
        "object" => (
            r#""{" space ( string ":" space value ("," space string ":" space value)* )? space "}""#,
            &["string", "value"],
        ),
        "array" => (r#""[" space ( value ("," space value)* )? space "]""#, &["value"]),
        "uuid" => (
            r#""\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"""#,
            &[],
        ),
        "char" => (
            r#"[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})"#,
            &[],
        ),
        "string" => (r#""\"" char* "\"""#, &["char"]),
        "null" => (r#""null""#, &[]),
        _ => return None,
    };
    Some(rule)
}

fn string_format_rule(name: &str) -> Option<(&'static str, &'static [&'static str])> {
    let rule: (&'static str, &'static [&'static str]) = match name {
        "date" => (
            r#"[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )"#,
            &[],
        ),
        "time" => (
            r#"([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )"#,
            &[],
        ),
        "date-time" => (r#"date "T" time"#, &["date", "time"]),
        "date-string" => (r#""\"" date "\"""#, &["date"]),
        "time-string" => (r#""\"" time "\"""#, &["time"]),
        "date-time-string" => (r#""\"" date-time "\"""#, &["date-time"]),
        _ => return None,
    };
    Some(rule)
}

fn builtin_rule(name: &str) -> Option<(&'static str, &'static [&'static str])> {
    primitive_rule(name).or_else(|| string_format_rule(name))
}

fn is_reserved_name(name: &str) -> bool {
    name == "root" || builtin_rule(name).is_some()
}

fn format_literal(literal: &str) -> String {
    let mut s = String::with_capacity(literal.len() + 2);
    s.push('"');
    for c in literal.chars() {
        match c {
            '\r' => s.push_str("\\r"),
            '\n' => s.push_str("\\n"),
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            _ => s.push(c),
        }
    }
    s.push('"');
    s
}

fn sanitize_rule_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut in_run = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' {
            out.push(c);
            in_run = false;
        } else if !in_run {
            out.push('-');
            in_run = true;
        }
    }
    out
}

fn gbnf_escape_length(s: &[char], pos: usize) -> usize {
    if pos + 1 >= s.len() || s[pos] != '\\' {
        return 0;
    }
    let n_hex = match s[pos + 1] {
        'x' => 2,
        'u' => 4,
        'U' => 8,
        't' | 'r' | 'n' | '\\' | '"' | '[' | ']' | '-' => return 2,
        _ => return 0,
    };
    if pos + 2 + n_hex > s.len() {
        return 0;
    }
    for &c in &s[pos + 2..pos + 2 + n_hex] {
        if !c.is_ascii_hexdigit() {
            return 0;
        }
    }
    2 + n_hex
}

fn build_repetition(item_rule: &str, min_items: i64, max_items: i64, separator_rule: &str) -> String {
    let has_max = max_items != NO_MAX;
    if max_items == 0 {
        return String::new();
    }
    if min_items == 0 && max_items == 1 {
        return format!("{item_rule}?");
    }
    if separator_rule.is_empty() {
        if min_items == 1 && !has_max {
            return format!("{item_rule}+");
        }
        if min_items == 0 && !has_max {
            return format!("{item_rule}*");
        }
        let max = if has_max { max_items.to_string() } else { String::new() };
        return format!("{item_rule}{{{min_items},{max}}}");
    }
    let inner_item = format!("({separator_rule} {item_rule})");
    let inner_min = if min_items == 0 { 0 } else { min_items - 1 };
    let inner_max = if has_max { max_items - 1 } else { max_items };
    let result = format!("{item_rule} {}", build_repetition(&inner_item, inner_min, inner_max, ""));
    if min_items == 0 {
        format!("({result})?")
    } else {
        result
    }
}

fn to_rule(item: &(String, bool)) -> String {
    if item.1 {
        format!("\"{}\"", item.0)
    } else {
        item.0.clone()
    }
}

fn join_seq(seq: Vec<(String, bool)>) -> (String, bool) {
    let mut ret: Vec<(String, bool)> = Vec::new();
    let mut literal = String::new();
    for item in seq {
        if item.1 {
            literal.push_str(&item.0);
        } else {
            if !literal.is_empty() {
                ret.push((std::mem::take(&mut literal), true));
            }
            ret.push(item);
        }
    }
    if !literal.is_empty() {
        ret.push((literal, true));
    }
    let results: Vec<String> = ret.iter().map(to_rule).collect();
    (results.join(" "), false)
}

#[derive(Clone, Copy)]
enum Addl<'a> {
    None,
    Any,
    Schema(&'a Value),
}

#[derive(Clone, Copy, PartialEq)]
enum StringFormat {
    Date,
    Time,
    DateTime,
    Uuid,
}

fn is_any_schema(v: &Value) -> bool {
    match v {
        Value::Bool(true) => true,
        Value::Object(m) => {
            const KEYS: [&str; 12] = [
                "$ref", "oneOf", "anyOf", "type", "const", "enum", "allOf", "items",
                "prefixItems", "pattern", "minLength", "maxLength",
            ];
            if KEYS.iter().any(|k| m.contains_key(*k)) {
                return false;
            }
            if m.contains_key("format") || m.contains_key("properties") {
                return false;
            }
            if m.get("additionalProperties").is_some_and(|a| a != &Value::Bool(true)) {
                return false;
            }
            true
        }
        _ => false,
    }
}

struct Converter {
    root: Value,
    dotall: bool,
    rules: BTreeMap<String, String>,
    refs_being_resolved: HashSet<String>,
    errors: Vec<String>,
    #[allow(dead_code)]
    warnings: Vec<String>,
}

impl Converter {
    fn new(root: &Value, dotall: bool) -> Self {
        let mut rules = BTreeMap::new();
        rules.insert("space".to_string(), SPACE_RULE.to_string());
        Self {
            root: root.clone(),
            dotall,
            rules,
            refs_being_resolved: HashSet::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn add_rule(&mut self, name: &str, rule: &str) -> String {
        let esc = sanitize_rule_name(name);
        if !self.rules.contains_key(&esc) || self.rules.get(&esc).map(String::as_str) == Some(rule) {
            self.rules.insert(esc.clone(), rule.to_string());
            return esc;
        }
        let mut i = 0usize;
        loop {
            let key = format!("{esc}{i}");
            match self.rules.get(&key) {
                Some(existing) if existing != rule => i += 1,
                _ => {
                    self.rules.insert(key.clone(), rule.to_string());
                    return key;
                }
            }
        }
    }

    fn add_primitive_as(&mut self, name: &str, type_name: &str) -> String {
        let (content, deps) =
            builtin_rule(type_name).unwrap_or_else(|| panic!("unknown builtin rule {type_name}"));
        let n = self.add_rule(name, content);
        for dep in deps {
            if !self.rules.contains_key(*dep) {
                self.add_primitive_as(dep, dep);
            }
        }
        n
    }

    fn add_primitive_named(&mut self, type_name: &str) -> String {
        self.add_primitive_as(type_name, type_name)
    }

    fn visit_primitive(&mut self, rule_name: &str, type_name: &str) -> String {
        let target = if rule_name == "root" { "root" } else { type_name };
        self.add_primitive_as(target, type_name)
    }

    fn constant_rule(&self, value: &Value) -> String {
        format_literal(&value.to_string())
    }

    fn get_count(&mut self, value: Option<&Value>, key: &str, def: i64) -> i64 {
        match value {
            None => def,
            Some(v) => match v.as_i64() {
                Some(n) if n >= 0 => n,
                _ => {
                    self.errors.push(format!("{key} must be a non-negative integer"));
                    def
                }
            },
        }
    }

    fn get_format(&self, m: &Map<String, Value>) -> Option<StringFormat> {
        let f = m.get("format")?.as_str()?;
        match f {
            "date" => Some(StringFormat::Date),
            "time" => Some(StringFormat::Time),
            "date-time" => Some(StringFormat::DateTime),
            _ => {
                let b = f.as_bytes();
                if f == "uuid" || (b.len() == 5 && f.starts_with("uuid") && (b'1'..=b'5').contains(&b[4]))
                {
                    Some(StringFormat::Uuid)
                } else {
                    None
                }
            }
        }
    }

    fn resolve_pointer(&self, ref_str: &str) -> Result<Value, String> {
        // Only local refs: "#" (root) or "#/...".
        let Some(rest) = ref_str.strip_prefix('#') else {
            return Err("only local refs (#/...) are supported".into());
        };
        if !rest.is_empty() && !rest.starts_with('/') {
            return Err("only local refs (#/...) are supported".into());
        }
        let mut cur = &self.root;
        for sel in rest.split('/').skip(1) {
            match cur {
                Value::Object(map) => {
                    cur = map.get(sel).ok_or_else(|| format!("{sel} not found"))?;
                }
                Value::Array(arr) => {
                    let idx: usize = sel.parse().unwrap_or(arr.len());
                    cur = arr.get(idx).ok_or_else(|| format!("{sel} is out of range"))?;
                }
                _ => return Err(format!("{sel} not found")),
            }
        }
        Ok(cur.clone())
    }

    fn resolve_ref(&mut self, ref_str: &str) -> String {
        let fragment = match ref_str.find('#') {
            Some(i) => &ref_str[i + 1..],
            None => ref_str,
        };
        let ref_name = format!("ref{}", sanitize_rule_name(fragment));
        if !self.rules.contains_key(&ref_name) && !self.refs_being_resolved.contains(ref_str) {
            if !ref_str.starts_with("#/") {
                self.errors.push(format!(
                    "unsupported $ref {ref_str}, only references into the same document are supported"
                ));
                return ref_name;
            }
            let target = match self.resolve_pointer(ref_str) {
                Ok(t) => t,
                Err(e) => {
                    self.errors.push(format!("Unresolved $ref {ref_str}: {e}"));
                    return ref_name;
                }
            };
            self.refs_being_resolved.insert(ref_str.to_string());
            let n = self.visit(&target, &ref_name);
            self.refs_being_resolved.remove(ref_str);
            return n;
        }
        ref_name
    }

    fn generate_union_rule(&mut self, name: &str, alts: &[Value]) -> String {
        let mut rules = Vec::with_capacity(alts.len());
        for (i, alt) in alts.iter().enumerate() {
            let child = format!("{}{}{}", name, if name.is_empty() { "alternative-" } else { "-" }, i);
            rules.push(self.visit(alt, &child));
        }
        rules.join(" | ")
    }

    fn visit(&mut self, schema: &Value, name: &str) -> String {
        let rule_name = if is_reserved_name(name) {
            format!("{name}-")
        } else if name.is_empty() {
            "root".to_string()
        } else {
            name.to_string()
        };
        let sub_name = if name.is_empty() { String::new() } else { format!("{name}-") };

        if let Value::Bool(true) = schema {
            let v = self.add_primitive_named("value");
            return self.add_rule(&rule_name, &v);
        }
        let m = match schema {
            Value::Object(m) => m,
            _ => {
                self.errors.push("schema must be an object".to_string());
                return self.add_rule(&rule_name, "");
            }
        };

        if let Some(r) = m.get("$ref") {
            let target = self.resolve_ref(r.as_str().unwrap_or(""));
            return self.add_rule(&rule_name, &target);
        }
        if let Some(alts) = m.get("oneOf").or_else(|| m.get("anyOf")) {
            let alts = alts.as_array().cloned().unwrap_or_default();
            let union = self.generate_union_rule(name, &alts);
            return self.add_rule(&rule_name, &union);
        }
        if let Some(Value::Array(types)) = m.get("type") {
            if types.is_empty() {
                self.errors.push("type must not be empty".to_string());
                return self.add_rule(&rule_name, "");
            }
            let mut alts = Vec::with_capacity(types.len());
            for t in types {
                let mut alt = m.clone();
                alt.insert("type".to_string(), t.clone());
                alts.push(Value::Object(alt));
            }
            let union = self.generate_union_rule(name, &alts);
            return self.add_rule(&rule_name, &union);
        }
        if let Some(c) = m.get("const") {
            let r = self.constant_rule(c);
            return self.add_rule(&rule_name, &r);
        }
        if let Some(e) = m.get("enum") {
            let vals = e.as_array().cloned().unwrap_or_default();
            if vals.is_empty() {
                self.errors.push("enum must be a non-empty array".to_string());
                return self.add_rule(&rule_name, "");
            }
            let parts: Vec<String> = vals.iter().map(|v| self.constant_rule(v)).collect();
            return self.add_rule(&rule_name, &format!("({})", parts.join(" | ")));
        }

        let type_name = match m.get("type") {
            None => String::new(),
            Some(Value::String(s)) => s.clone(),
            Some(_) => {
                self.errors.push("type must be a string or an array of strings".to_string());
                return self.add_rule(&rule_name, "");
            }
        };
        let has_properties = m.contains_key("properties")
            || (m.contains_key("additionalProperties")
                && m.get("additionalProperties") != Some(&Value::Bool(true)));

        if type_name.is_empty() {
            if has_properties {
                return self.visit_object(m, name, &rule_name);
            }
            if m.contains_key("allOf") {
                return self.visit_all_of(m, name, &rule_name);
            }
            if m.contains_key("items") || m.contains_key("prefixItems") {
                return self.visit_array(m, name, &rule_name, &sub_name);
            }
            if m.contains_key("pattern")
                || m.contains_key("minLength")
                || m.contains_key("maxLength")
                || self.get_format(m).is_some()
            {
                return self.visit_string(m, &rule_name);
            }
            let v = self.add_primitive_named("value");
            return self.add_rule(&rule_name, &v);
        }

        match type_name.as_str() {
            "object" => {
                if !has_properties && m.contains_key("allOf") {
                    return self.visit_all_of(m, name, &rule_name);
                }
                self.visit_object(m, name, &rule_name)
            }
            "string" => {
                if m.contains_key("allOf") {
                    return self.visit_all_of(m, name, &rule_name);
                }
                self.visit_string(m, &rule_name)
            }
            "array" => self.visit_array(m, name, &rule_name, &sub_name),
            "integer" => self.visit_primitive(&rule_name, "integer"),
            "number" => self.visit_primitive(&rule_name, "number"),
            "boolean" => self.visit_primitive(&rule_name, "boolean"),
            "null" => self.visit_primitive(&rule_name, "null"),
            other => {
                self.errors.push(format!("unrecognized type {other}"));
                self.add_rule(&rule_name, "")
            }
        }
    }

    fn visit_object(&mut self, m: &Map<String, Value>, name: &str, rule_name: &str) -> String {
        let mut properties: Vec<(String, Value)> = Vec::new();
        let mut required: BTreeSet<String> = BTreeSet::new();
        if let Some(Value::Array(req)) = m.get("required") {
            for r in req {
                if let Some(s) = r.as_str() {
                    required.insert(s.to_string());
                }
            }
        }
        if let Some(Value::Object(props)) = m.get("properties") {
            for (k, v) in props {
                properties.push((k.clone(), v.clone()));
            }
        }
        let has_props = m.contains_key("properties");
        let addl = match m.get("additionalProperties") {
            Some(Value::Bool(true)) => Addl::Any,
            Some(Value::Bool(false)) => Addl::None,
            Some(v) => Addl::Schema(v),
            None if has_props => Addl::None,
            None => Addl::Any,
        };
        if properties.is_empty() && matches!(addl, Addl::Any) {
            let o = self.add_primitive_named("object");
            return self.add_rule(rule_name, &o);
        }
        let r = self.build_object_rule(&properties, &required, name, addl);
        self.add_rule(rule_name, &r)
    }

    fn visit_array(&mut self, m: &Map<String, Value>, _name: &str, rule_name: &str, sub_name: &str) -> String {
        let key = if m.contains_key("items") {
            Some("items")
        } else if m.contains_key("prefixItems") {
            Some("prefixItems")
        } else {
            None
        };
        let min = self.get_count(m.get("minItems"), "minItems", 0);
        let max = self.get_count(m.get("maxItems"), "maxItems", -1);

        if let Some(key) = key {
            let items = m.get(key).unwrap();
            if let Value::Array(arr) = items {
                let mut rule = String::from("\"[\" space ");
                for (i, it) in arr.iter().enumerate() {
                    if i > 0 {
                        rule.push_str(" \",\" space ");
                    }
                    rule.push_str(&self.visit(it, &format!("{sub_name}tuple-{i}")));
                }
                rule.push_str(" space \"]\"");
                return self.add_rule(rule_name, &rule);
            }
            if is_any_schema(items) && min == 0 && max < 0 {
                return self.visit_primitive(rule_name, "array");
            }
            let item_rule = self.visit(items, &format!("{sub_name}item"));
            let max = if max < 0 { NO_MAX } else { max };
            let rep = build_repetition(&item_rule, min, max, "\",\" space");
            return self.add_rule(rule_name, &format!("\"[\" space {rep} space \"]\""));
        }
        if min == 0 && max < 0 {
            return self.visit_primitive(rule_name, "array");
        }
        let item_rule = self.add_primitive_named("value");
        let max = if max < 0 { NO_MAX } else { max };
        let rep = build_repetition(&item_rule, min, max, "\",\" space");
        self.add_rule(rule_name, &format!("\"[\" space {rep} space \"]\""))
    }

    fn visit_string(&mut self, m: &Map<String, Value>, rule_name: &str) -> String {
        if let Some(Value::String(pat)) = m.get("pattern") {
            return self.visit_pattern(pat, rule_name);
        }
        if let Some(fmt) = self.get_format(m) {
            match fmt {
                StringFormat::Uuid => return self.visit_primitive(rule_name, "uuid"),
                StringFormat::Date => {
                    let p = self.add_primitive_named("date-string");
                    return self.add_rule(rule_name, &p);
                }
                StringFormat::Time => {
                    let p = self.add_primitive_named("time-string");
                    return self.add_rule(rule_name, &p);
                }
                StringFormat::DateTime => {
                    let p = self.add_primitive_named("date-time-string");
                    return self.add_rule(rule_name, &p);
                }
            }
        }
        let min = self.get_count(m.get("minLength"), "minLength", 0);
        let max = self.get_count(m.get("maxLength"), "maxLength", -1);
        if min > 0 || max >= 0 {
            let char_rule = self.add_primitive_named("char");
            let max = if max < 0 { NO_MAX } else { max };
            let rep = build_repetition(&char_rule, min, max, "");
            return self.add_rule(rule_name, &format!("\"\\\"\" {rep} \"\\\"\""));
        }
        self.visit_primitive(rule_name, "string")
    }

    fn visit_all_of(&mut self, m: &Map<String, Value>, name: &str, rule_name: &str) -> String {
        let children = m.get("allOf").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let mut properties: Vec<(String, Value)> = Vec::new();
        let mut required: BTreeSet<String> = BTreeSet::new();
        let mut enum_values: BTreeMap<String, usize> = BTreeMap::new();
        for child in &children {
            if let Some(alts) = child.get("anyOf").or_else(|| child.get("oneOf")).and_then(|v| v.as_array()) {
                for alt in alts {
                    self.add_component(alt, false, &mut properties, &mut required, &mut enum_values);
                }
            } else {
                self.add_component(child, true, &mut properties, &mut required, &mut enum_values);
            }
        }
        if !enum_values.is_empty() {
            let n = children.len();
            let inter: Vec<String> = enum_values
                .iter()
                .filter(|(_, c)| **c == n)
                .map(|(k, _)| k.clone())
                .collect();
            if !inter.is_empty() {
                return self.add_rule(rule_name, &format!("({})", inter.join(" | ")));
            }
        }
        let r = self.build_object_rule(&properties, &required, name, Addl::None);
        self.add_rule(rule_name, &r)
    }

    fn add_component(
        &mut self,
        comp: &Value,
        is_required: bool,
        properties: &mut Vec<(String, Value)>,
        required: &mut BTreeSet<String>,
        enum_values: &mut BTreeMap<String, usize>,
    ) {
        if let Some(r) = comp.get("$ref").and_then(|v| v.as_str()) {
            // allOf members are inlined, so a self-referencing $ref here would recurse forever.
            if !self.refs_being_resolved.insert(r.to_string()) {
                self.errors.push(format!("recursive $ref {r} inside allOf"));
                return;
            }
            match self.resolve_pointer(r) {
                Ok(target) => self.add_component(&target, is_required, properties, required, enum_values),
                Err(e) => self.errors.push(format!("Unresolved $ref {r}: {e}")),
            }
            self.refs_being_resolved.remove(r);
            return;
        }
        let is_object =
            comp.get("properties").is_some() || comp.get("type").and_then(|v| v.as_str()) == Some("object");
        if is_object {
            if let Some(Value::Object(props)) = comp.get("properties") {
                for (k, v) in props {
                    properties.push((k.clone(), v.clone()));
                    if is_required {
                        required.insert(k.clone());
                    }
                }
            }
        } else if let Some(Value::Array(vals)) = comp.get("enum") {
            for v in vals {
                let c = self.constant_rule(v);
                *enum_values.entry(c).or_insert(0) += 1;
            }
        }
    }

    fn build_object_rule(
        &mut self,
        properties: &[(String, Value)],
        required: &BTreeSet<String>,
        name: &str,
        additional_properties: Addl<'_>,
    ) -> String {
        let mut required_props: Vec<String> = Vec::new();
        let mut optional_props: Vec<String> = Vec::new();
        let mut prop_kv: BTreeMap<String, String> = BTreeMap::new();
        let mut prop_names: Vec<String> = Vec::new();

        for (prop_name, prop_schema) in properties {
            let child = format!("{}{}{}", name, if name.is_empty() { "" } else { "-" }, prop_name);
            let prop_rule_name = self.visit(prop_schema, &child);
            let kv = self.add_rule(
                &format!("{child}-kv"),
                &format!(
                    "{} space \":\" space {}",
                    format_literal(&Value::String(prop_name.clone()).to_string()),
                    prop_rule_name
                ),
            );
            prop_kv.insert(prop_name.clone(), kv);
            if required.contains(prop_name) {
                required_props.push(prop_name.clone());
            } else {
                optional_props.push(prop_name.clone());
            }
            prop_names.push(prop_name.clone());
        }

        if let Addl::Any | Addl::Schema(_) = additional_properties {
            let sub_name = format!("{}{}additional", name, if name.is_empty() { "" } else { "-" });
            let value_rule = match additional_properties {
                Addl::Any => self.add_primitive_named("value"),
                Addl::Schema(v) if is_any_schema(v) => self.add_primitive_named("value"),
                Addl::Schema(v) => self.visit(v, &format!("{sub_name}-value")),
                Addl::None => unreachable!(),
            };
            let key_rule = if prop_names.is_empty() {
                self.add_primitive_named("string")
            } else {
                let ns = self.not_strings(&prop_names);
                self.add_rule(&format!("{sub_name}-k"), &ns)
            };
            let kv_rule = self.add_rule(
                &format!("{sub_name}-kv"),
                &format!("{key_rule} space \":\" space {value_rule}"),
            );
            prop_kv.insert("*".to_string(), kv_rule);
            optional_props.push("*".to_string());
        }

        if required_props.is_empty() && optional_props.is_empty() {
            return "\"{\" space \"}\"".to_string();
        }

        let mut rule = String::from("\"{\" space ");
        for (i, p) in required_props.iter().enumerate() {
            if i > 0 {
                rule.push_str(" \",\" space ");
            }
            rule.push_str(prop_kv.get(p).unwrap());
        }

        if !optional_props.is_empty() {
            rule.push_str(" (");
            if !required_props.is_empty() {
                rule.push_str(" \",\" space ( ");
            }
            for i in 0..optional_props.len() {
                if i > 0 {
                    rule.push_str(" | ");
                }
                let res = self.get_recursive_refs(&optional_props[i..], false, name, &prop_kv);
                rule.push_str(&res);
            }
            if !required_props.is_empty() {
                rule.push_str(" )");
            }
            rule.push_str(" )?");
        }
        rule.push_str(" space \"}\"");
        rule
    }

    fn get_recursive_refs(
        &mut self,
        ks: &[String],
        first_is_optional: bool,
        name: &str,
        prop_kv: &BTreeMap<String, String>,
    ) -> String {
        if ks.is_empty() {
            return String::new();
        }
        let k = &ks[0];
        let kv = &prop_kv[k];
        let comma_ref = format!("( \",\" space {kv} )");
        let mut res = if first_is_optional {
            format!("{comma_ref}{}", if k == "*" { "*" } else { "?" })
        } else if k == "*" {
            format!("{kv} {comma_ref}*")
        } else {
            kv.clone()
        };
        if ks.len() > 1 {
            let rest = self.get_recursive_refs(&ks[1..], true, name, prop_kv);
            let rest_name = format!("{}{}{}-rest", name, if name.is_empty() { "" } else { "-" }, k);
            let r = self.add_rule(&rest_name, &rest);
            res.push(' ');
            res.push_str(&r);
        }
        res
    }

    fn not_strings(&mut self, strings: &[String]) -> String {
        let mut nodes: Vec<TrieNode> = vec![TrieNode { children: BTreeMap::new(), pattern: -1 }];
        let mut n_pat = 0i32;
        for w in strings {
            let mut cur = 0usize;
            for c in w.chars() {
                let next = match nodes[cur].children.get(&c) {
                    Some(&x) => x,
                    None => {
                        let idx = nodes.len();
                        nodes.push(TrieNode { children: BTreeMap::new(), pattern: -1 });
                        nodes[cur].children.insert(c, idx);
                        idx
                    }
                };
                cur = next;
            }
            if nodes[cur].pattern < 0 {
                nodes[cur].pattern = n_pat;
                n_pat += 1;
            }
        }

        let char_rule = self.add_primitive_named("char");
        let mut out = String::from("[\"] ( ");
        trie_visit(&nodes, 0, &mut out, &char_rule);
        out.push_str(" )");
        if nodes[0].pattern < 0 {
            out.push('?');
        }
        out.push_str(" [\"]");
        out
    }

    fn get_dot_rule(&self) -> &'static str {
        if self.dotall {
            r"[\U00000000-\U0010FFFF]"
        } else {
            r"[^\x0A\x0D]"
        }
    }

    fn visit_pattern(&mut self, pattern: &str, rule_name: &str) -> String {
        match self.pattern_to_rule(pattern, rule_name) {
            Ok(r) => r,
            Err(msg) => {
                self.errors
                    .push(format!("pattern not supported: {pattern} ({msg})"));
                let s = self.add_primitive_named("string");
                self.add_rule(rule_name, &s)
            }
        }
    }

    fn pattern_to_rule(&mut self, pattern: &str, name: &str) -> Result<String, String> {
        let chars: Vec<char> = pattern.chars().collect();
        if chars.len() < 2 || chars[0] != '^' || chars[chars.len() - 1] != '$' {
            return Err("not anchored with '^' and '$'".to_string());
        }
        let sub: Vec<char> = chars[1..chars.len() - 1].to_vec();
        let mut i = 0usize;
        let mut depth = 0usize;
        let mut sub_ids: BTreeMap<String, String> = BTreeMap::new();
        let (rule, _is_literal) = self.pattern_transform(&sub, &mut i, &mut depth, &mut sub_ids, name)?;
        if depth != 0 {
            return Err("unbalanced parentheses".to_string());
        }
        let body = if _is_literal { format!("\"{rule}\"") } else { rule };
        Ok(self.add_rule(name, &format!("\"\\\"\" ({body}) \"\\\"\"")))
    }

    fn pattern_transform(
        &mut self,
        s: &[char],
        i: &mut usize,
        depth: &mut usize,
        sub_ids: &mut BTreeMap<String, String>,
        name: &str,
    ) -> Result<(String, bool), String> {
        let length = s.len();
        let mut seq: Vec<(String, bool)> = Vec::new();

        const NON_LITERAL: [char; 13] =
            ['|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?', '^', '$'];
        const ESCAPED: [char; 13] =
            ['^', '$', '.', '[', ']', '(', ')', '|', '{', '}', '*', '+', '?'];
        let is_non_literal = |c: char| NON_LITERAL.contains(&c);

        while *i < length {
            let c = s[*i];
            if c == '.' {
                let r = self.get_dot_rule();
                let dot = self.add_rule("dot", r);
                seq.push((dot, false));
                *i += 1;
            } else if c == '(' {
                *i += 1;
                if *i < length && s[*i] == '?' {
                    if *i + 1 < length && s[*i + 1] == ':' {
                        *i += 2;
                    } else {
                        return Err("unsupported group syntax".to_string());
                    }
                }
                *depth += 1;
                if *depth > MAX_PATTERN_DEPTH {
                    return Err("pattern nesting too deep".to_string());
                }
                let inner = self.pattern_transform(s, i, depth, sub_ids, name)?;
                seq.push((format!("({})", to_rule(&inner)), false));
            } else if c == ')' {
                *i += 1;
                if *depth == 0 {
                    return Err("unbalanced parentheses".to_string());
                }
                *depth -= 1;
                return Ok(join_seq(seq));
            } else if c == '^' || c == '$' {
                return Err("anchor inside the pattern".to_string());
            } else if c == '[' {
                let mut sb = String::from(c);
                *i += 1;
                while *i < length && s[*i] != ']' {
                    if s[*i] == '\\' {
                        let el = gbnf_escape_length(s, *i);
                        if el == 0 {
                            let frag: String = s[*i..(*i + 2).min(length)].iter().collect();
                            return Err(format!("unsupported escape in character class: {frag}"));
                        }
                        sb.extend(&s[*i..*i + el]);
                        *i += el;
                    } else {
                        sb.push(s[*i]);
                        *i += 1;
                    }
                }
                if *i >= length {
                    return Err("unterminated character class".to_string());
                }
                sb.push(']');
                *i += 1;
                seq.push((sb, false));
            } else if c == '|' {
                seq.push(("|".to_string(), false));
                *i += 1;
            } else if c == '*' || c == '+' || c == '?' {
                if seq.is_empty() {
                    return Err("nothing to repeat".to_string());
                }
                let last = seq.pop().unwrap();
                seq.push((format!("{}{}", to_rule(&last), c), false));
                *i += 1;
            } else if c == '{' {
                let mut cb = String::from(c);
                *i += 1;
                while *i < length && s[*i] != '}' {
                    cb.push(s[*i]);
                    *i += 1;
                }
                if *i >= length {
                    return Err("unterminated curly brackets".to_string());
                }
                cb.push('}');
                *i += 1;
                let nums: Vec<&str> = cb[1..cb.len() - 1].split(',').collect();
                if nums.len() != 1 && nums.len() != 2 {
                    return Err("wrong number of values in curly brackets".to_string());
                }
                let mut min_times: i64 = 0;
                let mut max_times: i64 = NO_MAX;
                let bad = || "invalid number in curly brackets".to_string();
                if nums.len() == 1 {
                    let v: i64 = nums[0].parse().map_err(|_| bad())?;
                    min_times = v;
                    max_times = v;
                } else {
                    if !nums[0].is_empty() {
                        min_times = nums[0].parse().map_err(|_| bad())?;
                    }
                    if !nums[1].is_empty() {
                        max_times = nums[1].parse().map_err(|_| bad())?;
                    }
                }
                if seq.is_empty() {
                    return Err("nothing to repeat".to_string());
                }
                let (sub, sub_is_literal) = seq.pop().unwrap();
                let sub2 = if !sub_is_literal {
                    if !sub_ids.contains_key(&sub) {
                        let id = self.add_rule(&format!("{name}-{}", sub_ids.len()), &sub);
                        sub_ids.insert(sub.clone(), id);
                    }
                    sub_ids[&sub].clone()
                } else {
                    sub
                };
                let item = if sub_is_literal { format!("\"{sub2}\"") } else { sub2 };
                seq.push((build_repetition(&item, min_times, max_times, ""), false));
            } else {
                let mut literal = String::new();
                while *i < length {
                    if s[*i] == '\\' {
                        if *i == length - 1 {
                            return Err("trailing backslash".to_string());
                        }
                        let next = s[*i + 1];
                        if ESCAPED.contains(&next) {
                            *i += 1;
                            literal.push(s[*i]);
                            *i += 1;
                        } else {
                            let el = gbnf_escape_length(s, *i);
                            if el == 0 {
                                let frag: String = s[*i..(*i + 2).min(length)].iter().collect();
                                return Err(format!("unsupported escape: {frag}"));
                            }
                            literal.extend(&s[*i..*i + el]);
                            *i += el;
                        }
                    } else if s[*i] == '"' {
                        literal.push_str("\\\"");
                        *i += 1;
                    } else if !is_non_literal(s[*i])
                        && (*i == length - 1
                            || literal.is_empty()
                            || s[*i + 1] == '.'
                            || !is_non_literal(s[*i + 1]))
                    {
                        literal.push(s[*i]);
                        *i += 1;
                    } else {
                        break;
                    }
                }
                if literal.is_empty() {
                    return Err(format!("unsupported character: {c}"));
                }
                seq.push((literal, true));
            }
        }
        Ok(join_seq(seq))
    }

    fn format_grammar(&self) -> String {
        let mut out = String::new();
        if let Some(r) = self.rules.get("root") {
            out.push_str(&format!("root ::= {r}\n"));
        }
        for (k, v) in &self.rules {
            if k == "root" {
                continue;
            }
            out.push_str(&format!("{k} ::= {v}\n"));
        }
        out
    }
}

struct TrieNode {
    children: BTreeMap<char, usize>,
    pattern: i32,
}

fn trie_visit(nodes: &[TrieNode], idx: usize, out: &mut String, char_rule: &str) {
    let node = &nodes[idx];
    let mut rejects = String::new();
    let mut first = true;
    for (cpt, child) in &node.children {
        rejects.push(*cpt);
        if first {
            first = false;
        } else {
            out.push_str(" | ");
        }
        out.push('[');
        out.push(*cpt);
        out.push(']');
        if !nodes[*child].children.is_empty() {
            out.push_str(" (");
            trie_visit(nodes, *child, out, char_rule);
            out.push(')');
        } else {
            out.push(' ');
            out.push_str(char_rule);
            out.push('+');
        }
    }
    if !node.children.is_empty() {
        out.push_str(&format!(" | [^\"{rejects}] {char_rule}*"));
    }
}

/// Convert a JSON schema to a GBNF grammar.
pub(crate) fn json_schema_to_grammar(schema: &Value) -> Result<String, String> {
    let mut conv = Converter::new(schema, false);
    conv.visit(schema, "");
    if !conv.errors.is_empty() {
        return Err(format!("JSON schema conversion failed:\n{}", conv.errors.join("\n")));
    }
    Ok(conv.format_grammar())
}

/// The generic "any JSON value, root is an object" grammar (json_object mode).
pub(crate) fn any_json_grammar() -> String {
    String::from(
        r#"root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

ws ::= | " " | "\n" [ \t]{0,20}
"#,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeSet;

    fn rhs_identifiers(rhs: &str) -> Vec<String> {
        let b: Vec<char> = rhs.chars().collect();
        let mut i = 0;
        let mut out = Vec::new();
        while i < b.len() {
            match b[i] {
                '"' => {
                    i += 1;
                    while i < b.len() {
                        if b[i] == '\\' {
                            i += 2;
                        } else if b[i] == '"' {
                            i += 1;
                            break;
                        } else {
                            i += 1;
                        }
                    }
                }
                '[' => {
                    i += 1;
                    while i < b.len() {
                        if b[i] == '\\' {
                            i += 2;
                        } else if b[i] == ']' {
                            i += 1;
                            break;
                        } else {
                            i += 1;
                        }
                    }
                }
                '#' => break,
                c if c.is_ascii_alphabetic() || c == '_' => {
                    let start = i;
                    i += 1;
                    while i < b.len()
                        && (b[i].is_ascii_alphanumeric() || b[i] == '_' || b[i] == '-')
                    {
                        i += 1;
                    }
                    out.push(b[start..i].iter().collect());
                }
                _ => i += 1,
            }
        }
        out
    }

    /// Collect rule definitions and the identifiers used on right-hand sides.
    fn defined_and_used(grammar: &str) -> (BTreeSet<String>, BTreeSet<String>) {
        let mut defined = BTreeSet::new();
        let mut used = BTreeSet::new();
        let mut cur_rhs = String::new();
        let mut have = false;
        for line in grammar.lines() {
            if let Some(pos) = line.find("::=") {
                if have {
                    used.extend(rhs_identifiers(&cur_rhs));
                }
                let lhs = line[..pos].trim();
                if !lhs.is_empty() {
                    defined.insert(lhs.to_string());
                }
                cur_rhs = line[pos + 3..].to_string();
                have = true;
            } else if have {
                cur_rhs.push('\n');
                cur_rhs.push_str(line);
            }
        }
        if have {
            used.extend(rhs_identifiers(&cur_rhs));
        }
        (defined, used)
    }

    fn undefined_rules(grammar: &str) -> Vec<String> {
        let (defined, used) = defined_and_used(grammar);
        used.into_iter().filter(|u| !defined.contains(u)).collect()
    }

    fn assert_valid(grammar: &str) {
        let undef = undefined_rules(grammar);
        assert!(
            undef.is_empty(),
            "undefined rule references {undef:?} in grammar:\n{grammar}"
        );
    }

    fn conv(v: &Value) -> String {
        let g = json_schema_to_grammar(v).expect("conversion should succeed");
        assert_valid(&g);
        g
    }

    #[test]
    fn empty_schema_is_any_value() {
        let g = conv(&json!({}));
        assert!(g.contains("root ::= value"));
        assert!(g.contains("value ::= object | array | string | number | boolean | null"));
    }

    #[test]
    fn object_primitive() {
        let g = conv(&json!({"type": "object"}));
        assert!(g.contains("root ::= object"));
        assert!(g.contains("object ::= \"{\" space"));
    }

    #[test]
    fn object_properties_required_first() {
        let g = conv(&json!({
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"]
        }));
        assert!(g.contains("a-kv"));
        assert!(g.contains("b-kv"));
        let root = g.lines().find(|l| l.starts_with("root ::= ")).unwrap();
        assert!(root.contains("a-kv"));
        assert!(root.find("a-kv").unwrap() < root.find("b-kv").unwrap());
    }

    #[test]
    fn additional_properties_false() {
        let g = conv(&json!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": false
        }));
        assert!(g.contains("x-kv"));
        assert!(!g.contains("additional-kv"));
    }

    #[test]
    fn additional_properties_schema() {
        let g = conv(&json!({
            "type": "object",
            "additionalProperties": {"type": "integer"}
        }));
        assert!(g.contains("additional-kv"));
        assert!(g.contains("integer"));
    }

    #[test]
    fn additional_properties_true_with_properties() {
        let g = conv(&json!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": true
        }));
        assert!(g.contains("additional-k"));
        assert!(g.contains("additional-kv"));
    }

    #[test]
    fn array_items() {
        let g = conv(&json!({"type": "array", "items": {"type": "integer"}}));
        assert!(g.contains("root ::= \"[\" space"));
        assert!(g.contains("(integer"));
        assert!(g.contains("integral-part"));
    }

    #[test]
    fn array_prefix_items_tuple() {
        let g = conv(&json!({
            "type": "array",
            "prefixItems": [{"type": "string"}, {"type": "integer"}]
        }));
        assert!(g.contains("root ::= \"[\" space string \",\" space integer space \"]\""));
    }

    #[test]
    fn array_min_max_items() {
        let g = conv(&json!({
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3
        }));
        assert!(g.contains("{0,2}"));
    }

    #[test]
    fn string_enum() {
        let g = conv(&json!({"enum": ["a", "b"]}));
        assert!(g.contains("root ::= ("));
        assert!(g.contains(r#""\"a\"""#));
        assert!(g.contains(r#""\"b\"""#));
        assert!(g.contains(" | "));
    }

    #[test]
    fn const_object_literal() {
        let g = conv(&json!({"const": {"k": 1}}));
        assert!(g.contains(r#""{\"k\":1}""#));
    }

    #[test]
    fn nullable_type_array() {
        let g = conv(&json!({"type": ["string", "null"]}));
        assert!(g.contains("root ::= string | null"));
    }

    #[test]
    fn any_of_alternation() {
        let g = conv(&json!({"anyOf": [{"type": "integer"}, {"type": "boolean"}]}));
        assert!(g.contains("root ::= integer | boolean"));
    }

    #[test]
    fn all_of_merges_properties() {
        let g = conv(&json!({
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "integer"}}}
            ]
        }));
        assert!(g.contains("a-kv"));
        assert!(g.contains("b-kv"));
    }

    #[test]
    fn ref_to_defs() {
        let g = conv(&json!({
            "$defs": {"Foo": {"type": "object", "properties": {"x": {"type": "integer"}}}},
            "$ref": "#/$defs/Foo"
        }));
        assert!(g.contains("root ::= ref-defs-Foo"));
        assert!(g.contains("ref-defs-Foo ::= "));
    }

    #[test]
    fn recursive_ref() {
        let g = conv(&json!({
            "$defs": {"Node": {
                "type": "object",
                "properties": {"next": {"$ref": "#/$defs/Node"}}
            }},
            "$ref": "#/$defs/Node"
        }));
        assert!(g.contains("root ::= ref-defs-Node"));
        assert!(g.contains("ref-defs-Node-next ::= ref-defs-Node"));
    }

    #[test]
    fn string_format_date() {
        let g = conv(&json!({"type": "string", "format": "date"}));
        assert!(g.contains("root ::= date-string"));
        assert!(g.contains("date ::= "));
    }

    #[test]
    fn string_format_uuid() {
        let g = conv(&json!({"type": "string", "format": "uuid"}));
        assert!(g.contains("[0-9a-fA-F]{8}"));
        assert!(g.contains("[0-9a-fA-F]{12}"));
    }

    #[test]
    fn string_pattern() {
        let g = conv(&json!({"type": "string", "pattern": "^[a-z]+$"}));
        assert!(g.contains("[a-z]+"));
        assert!(g.contains("root ::= \"\\\"\""));
    }

    #[test]
    fn string_min_max_length() {
        let g = conv(&json!({"type": "string", "minLength": 2, "maxLength": 5}));
        assert!(g.contains("char{2,5}"));
    }

    #[test]
    fn scalar_primitives() {
        assert!(conv(&json!({"type": "integer"})).contains(r#"("-"? integral-part)"#));
        assert!(conv(&json!({"type": "number"})).contains("decimal-part"));
        assert!(conv(&json!({"type": "boolean"})).contains(r#"("true" | "false")"#));
        assert!(conv(&json!({"type": "null"})).contains(r#"root ::= "null""#));
    }

    #[test]
    fn any_json_grammar_is_valid() {
        let g = any_json_grammar();
        assert!(g.starts_with("root   ::= object"));
        assert_valid(&g);
    }

    #[test]
    fn root_rule_comes_first() {
        let g = conv(&json!({"type": "object", "properties": {"a": {"type": "string"}}}));
        assert!(g.lines().next().unwrap().starts_with("root ::= "));
    }

    #[test]
    fn undefined_rule_checker_detects_gaps() {
        let bad = "root ::= foo\nbar ::= \"x\"\n";
        assert_eq!(undefined_rules(bad), vec!["foo".to_string()]);
        assert_valid("root ::= bar\nbar ::= \"x\"\n");
    }

    #[test]
    fn remote_ref_is_error() {
        let err = json_schema_to_grammar(&json!({"$ref": "https://example.com/x"}))
            .expect_err("remote ref should fail");
        assert!(err.contains("unsupported $ref"), "{err}");
    }

    #[test]
    fn unsupported_pattern_is_error() {
        let err = json_schema_to_grammar(&json!({"type": "string", "pattern": "^(?=x)a$"}))
            .expect_err("lookaround should fail");
        assert!(err.contains("pattern not supported"), "{err}");
    }

    #[test]
    fn properties_follow_declared_order() {
        // Field order matters for quality (e.g. "reasoning" before "answer"); requires
        // serde_json's preserve_order feature.
        let schema: Value = serde_json::from_str(
            r#"{"type":"object","properties":{"reasoning":{"type":"string"},"answer":{"type":"string"}},"required":["reasoning","answer"]}"#,
        )
        .unwrap();
        let g = json_schema_to_grammar(&schema).unwrap();
        let root = g.lines().find(|l| l.starts_with("root ::=")).unwrap();
        let (r, a) = (root.find("reasoning-kv").unwrap(), root.find("answer-kv").unwrap());
        assert!(r < a, "{root}");
    }

    #[test]
    fn hostile_refs_error_instead_of_crashing() {
        for schema in [
            r##"{"allOf":[{"$ref":"#/$defs/a"}],"$defs":{"a":{"$ref":"#/$defs/a"}}}"##,
            r##"{"allOf":[{"$ref":"#/$defs/a"}],"$defs":{"a":{"allOf":[{"$ref":"#/$defs/b"}]},"b":{"allOf":[{"$ref":"#/$defs/a"}]}}}"##,
            r##"{"allOf":[{"$ref":""}]}"##,
            r##"{"allOf":[{"$ref":"é"}]}"##,
            r##"{"$ref":"x"}"##,
        ] {
            let v: Value = serde_json::from_str(schema).unwrap();
            let _ = json_schema_to_grammar(&v); // Err or Ok; must not overflow or panic
        }
    }
}
