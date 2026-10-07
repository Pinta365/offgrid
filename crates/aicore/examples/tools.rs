//! Tool calling with real local tools.
//! cargo run --release -p aicore --example tools -- model.gguf "What time is it in UTC, and what's 2^20?"

use aicore::*;
use serde_json::{Value, json};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: tools <model.gguf> [question]");
    let question = args.next().unwrap_or_else(|| "What time is it (UTC), and what is 1234 * 5678?".into());

    let model = Model::load(&path, &ModelParams::default())?;
    let mut chat = ChatSession::new(model, &SessionParams::default())?;
    let req = ChatRequest {
        tools: vec![
            Tool::function("utc_now", "Current UTC date and time.", json!({"type": "object", "properties": {}})),
            Tool::function(
                "multiply",
                "Multiply two integers exactly.",
                json!({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}),
            ),
        ],
        ..ChatRequest::new(vec![ChatMessage::system("Use the tools for facts and arithmetic."), ChatMessage::user(question)])
    };

    let (resp, history) = chat.run_tools(
        &req,
        4,
        |call| {
            let args: Value = serde_json::from_str(&call.function.arguments).unwrap_or_default();
            let result = match call.function.name.as_str() {
                // Return what the model needs: small models convert timestamps unreliably.
                "utc_now" => json!({"utc": iso_utc_now()}),
                "multiply" => json!({"product": args["a"].as_i64().unwrap_or(0) * args["b"].as_i64().unwrap_or(0)}),
                other => json!({"error": format!("unknown tool {other}")}),
            };
            eprintln!("[tool] {}({}) -> {result}", call.function.name, call.function.arguments);
            result.to_string()
        },
        |piece| {
            print!("{piece}");
            true
        },
    )?;
    println!();
    eprintln!("[{:?} after {} messages]", resp.finish_reason, history.len());
    Ok(())
}

/// Current time as `YYYY-MM-DDTHH:MM:SSZ` (civil-from-days, no date crate needed).
fn iso_utc_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let (days, tod) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", tod / 3600, tod / 60 % 60, tod % 60)
}
