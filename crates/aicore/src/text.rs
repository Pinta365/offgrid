/// Turns a stream of token bytes into UTF-8 text pieces, holding back
/// incomplete characters and anything that could be the start of a stop string.
pub(crate) struct TextStream {
    stops: Vec<String>,
    pending: Vec<u8>,
    held: String,
}

pub(crate) enum Push {
    /// Safe-to-emit text (may be empty).
    Text(String),
    /// A stop string was hit; text before it is returned, generation should end.
    Stop(String),
}

impl TextStream {
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            pending: Vec::new(),
            held: String::new(),
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Push {
        self.pending.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(s) => s.len(),
            Err(e) if e.error_len().is_some() => {
                // Invalid (not just incomplete) bytes: decode lossily and move on.
                let s = String::from_utf8_lossy(&self.pending).into_owned();
                self.pending.clear();
                self.held.push_str(&s);
                return self.drain();
            }
            Err(e) => e.valid_up_to(),
        };
        let s = std::str::from_utf8(&self.pending[..valid]).expect("validated");
        self.held.push_str(s);
        self.pending.drain(..valid);
        self.drain()
    }

    fn drain(&mut self) -> Push {
        if let Some(pos) = self.stops.iter().filter_map(|s| self.held.find(s.as_str())).min() {
            let out = self.held[..pos].to_string();
            self.held.clear();
            return Push::Stop(out);
        }
        // Hold back the longest suffix that is a proper prefix of some stop string.
        let mut keep = 0;
        for stop in &self.stops {
            for (i, _) in stop.char_indices().skip(1) {
                if i > keep && self.held.ends_with(&stop[..i]) {
                    keep = i;
                }
            }
        }
        let split = self.held.len() - keep;
        let rest = self.held.split_off(split);
        Push::Text(std::mem::replace(&mut self.held, rest))
    }

    /// Flush whatever is left at end of generation.
    pub fn finish(&mut self) -> String {
        let mut out = std::mem::take(&mut self.held);
        out.push_str(&String::from_utf8_lossy(&std::mem::take(&mut self.pending)));
        out
    }
}

/// Decides which streamed text is user-visible content: hides `<think>` reasoning and
/// everything from a tool-call marker on (those are returned parsed at the end).
pub(crate) struct StreamFilter {
    marker: Option<&'static str>,
    /// Llama 3 style: a reply starting with `{` is a tool call.
    json_start: bool,
    state: FilterState,
    held: String,
    started: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterState {
    Reasoning,
    Content,
    Suppressed,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

impl StreamFilter {
    pub fn new(marker: Option<&'static str>, json_start: bool, think_open_in_prompt: bool) -> Self {
        let state = if think_open_in_prompt { FilterState::Reasoning } else { FilterState::Content };
        Self { marker, json_start, state, held: String::new(), started: false }
    }

    /// Feed generated text; returns the part that is safe to show as content.
    pub fn push(&mut self, text: &str) -> String {
        self.held.push_str(text);
        let mut out = String::new();
        loop {
            match self.state {
                FilterState::Suppressed => {
                    self.held.clear();
                    return out;
                }
                FilterState::Reasoning => match self.held.find(THINK_CLOSE) {
                    Some(i) => {
                        self.held.drain(..i + THINK_CLOSE.len());
                        self.state = FilterState::Content;
                    }
                    None => {
                        // Keep a tail that might be the start of "</think>".
                        let keep = partial_suffix(&self.held, THINK_CLOSE);
                        self.held.drain(..self.held.len() - keep);
                        return out;
                    }
                },
                FilterState::Content => {
                    if !self.started {
                        let trimmed = self.held.trim_start();
                        if trimmed.is_empty() || (THINK_OPEN.starts_with(trimmed) && trimmed.len() < THINK_OPEN.len()) {
                            return out; // wait for more text
                        }
                        if let Some(rest) = trimmed.strip_prefix(THINK_OPEN) {
                            self.held = rest.to_string();
                            self.state = FilterState::Reasoning;
                            continue;
                        }
                        if self.json_start && trimmed.starts_with('{') {
                            self.state = FilterState::Suppressed;
                            continue;
                        }
                        self.held = trimmed.to_string();
                        self.started = true;
                    }
                    if let Some(marker) = self.marker {
                        if let Some(i) = self.held.find(marker) {
                            out.push_str(&self.held[..i]);
                            self.state = FilterState::Suppressed;
                            continue;
                        }
                        let keep = partial_suffix(&self.held, marker);
                        let rest = self.held.split_off(self.held.len() - keep);
                        out.push_str(&std::mem::replace(&mut self.held, rest));
                        return out;
                    }
                    out.push_str(&std::mem::take(&mut self.held));
                    return out;
                }
            }
        }
    }

    /// End of generation: release content held back for a marker that never completed.
    pub fn flush(&mut self) -> String {
        match self.state {
            FilterState::Content if self.started => std::mem::take(&mut self.held),
            _ => String::new(),
        }
    }
}

/// Length of the longest suffix of `s` that is a proper prefix of `pat` (on char boundaries).
fn partial_suffix(s: &str, pat: &str) -> usize {
    pat.char_indices()
        .skip(1)
        .map(|(i, _)| i)
        .filter(|&i| s.ends_with(&pat[..i]))
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], chunks: &[&[u8]]) -> (String, bool) {
        let mut ts = TextStream::new(&stops.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let mut out = String::new();
        for c in chunks {
            match ts.push(c) {
                Push::Text(t) => out.push_str(&t),
                Push::Stop(t) => {
                    out.push_str(&t);
                    return (out, true);
                }
            }
        }
        out.push_str(&ts.finish());
        (out, false)
    }

    #[test]
    fn split_utf8() {
        let bytes = "hé€".as_bytes();
        let (out, stopped) = run(&[], &[&bytes[..2], &bytes[2..4], &bytes[4..]]);
        assert_eq!(out, "hé€");
        assert!(!stopped);
    }

    #[test]
    fn stop_across_chunks() {
        let (out, stopped) = run(&["</end>"], &[b"hello </", b"en", b"d> world"]);
        assert_eq!(out, "hello ");
        assert!(stopped);
    }

    #[test]
    fn partial_stop_released() {
        let (out, stopped) = run(&["STOP"], &[b"ab ST", b"x"]);
        assert_eq!(out, "ab STx");
        assert!(!stopped);
    }

    fn filter(marker: Option<&'static str>, json: bool, think: bool, chunks: &[&str]) -> String {
        let mut f = StreamFilter::new(marker, json, think);
        let mut out: String = chunks.iter().map(|c| f.push(c)).collect();
        out.push_str(&f.flush());
        out
    }

    #[test]
    fn filter_hides_tool_markup() {
        assert_eq!(filter(Some("<tool_call>"), false, false, &["Let me check.", " <tool", "_call>{\"a\":1}", "</tool_call>"]), "Let me check. ");
        assert_eq!(filter(Some("<tool_call>"), false, false, &["a <to", "y"]), "a <toy");
    }

    #[test]
    fn filter_hides_reasoning() {
        assert_eq!(filter(None, false, false, &["\n<thi", "nk>hmm</th", "ink>\n\nAnswer"]), "Answer");
        assert_eq!(filter(None, false, true, &["still thinking</think>Done"]), "Done");
        assert_eq!(filter(None, false, false, &["<b>bold</b>"]), "<b>bold</b>");
    }

    #[test]
    fn filter_llama3_json_start() {
        assert_eq!(filter(None, true, false, &["  {\"name\": \"f\"", "}"]), "");
        assert_eq!(filter(None, true, false, &["Sure: {not a call}"]), "Sure: {not a call}");
    }
}
