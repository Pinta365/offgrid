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

/// Kind of streamed text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PieceKind {
    Content,
    Reasoning,
}

/// Splits streamed text into reply content and `<think>` reasoning, and hides
/// everything from a tool-call marker on (calls are returned parsed at the end).
pub(crate) struct StreamFilter {
    marker: Option<&'static str>,
    /// Llama 3 style: a reply starting with `{` is a tool call.
    json_start: bool,
    state: FilterState,
    held: String,
    started: bool,
    reasoning_started: bool,
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
        Self { marker, json_start, state, held: String::new(), started: false, reasoning_started: false }
    }

    /// Feed generated text; returns the pieces that are safe to show, in order.
    pub fn push(&mut self, text: &str) -> Vec<(PieceKind, String)> {
        self.held.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.state {
                FilterState::Suppressed => {
                    self.held.clear();
                    return out;
                }
                FilterState::Reasoning => {
                    let (emit, done) = match self.held.find(THINK_CLOSE) {
                        Some(i) => {
                            let emit = self.held[..i].to_string();
                            self.held.drain(..i + THINK_CLOSE.len());
                            (emit, true)
                        }
                        None => {
                            // Keep a tail that might be the start of "</think>".
                            let keep = partial_suffix(&self.held, THINK_CLOSE);
                            let rest = self.held.split_off(self.held.len() - keep);
                            (std::mem::replace(&mut self.held, rest), false)
                        }
                    };
                    self.emit_reasoning(emit, done, &mut out);
                    if done {
                        self.state = FilterState::Content;
                    } else {
                        return out;
                    }
                }
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
                    let visible = match self.marker {
                        Some(marker) => match self.held.find(marker) {
                            Some(i) => {
                                let v = self.held[..i].to_string();
                                self.state = FilterState::Suppressed;
                                v
                            }
                            None => {
                                let keep = partial_suffix(&self.held, marker);
                                let rest = self.held.split_off(self.held.len() - keep);
                                std::mem::replace(&mut self.held, rest)
                            }
                        },
                        None => std::mem::take(&mut self.held),
                    };
                    if !visible.is_empty() {
                        out.push((PieceKind::Content, visible));
                    }
                    if self.state != FilterState::Suppressed {
                        return out;
                    }
                }
            }
        }
    }

    /// Reasoning is reported without its leading/trailing blank lines (like the final
    /// `reasoning_content`), so strip whitespace at its start and hold trailing whitespace
    /// until more reasoning follows.
    fn emit_reasoning(&mut self, mut text: String, done: bool, out: &mut Vec<(PieceKind, String)>) {
        if !self.reasoning_started {
            text = text.trim_start().to_string();
        }
        if done {
            text.truncate(text.trim_end().len());
        } else {
            let trimmed = text.trim_end().len();
            let tail = text.split_off(trimmed);
            self.held.insert_str(0, &tail);
        }
        if !text.is_empty() {
            self.reasoning_started = true;
            out.push((PieceKind::Reasoning, text));
        }
    }

    /// End of generation: release text held back for a marker that never completed.
    pub fn flush(&mut self) -> Vec<(PieceKind, String)> {
        let held = std::mem::take(&mut self.held);
        match self.state {
            FilterState::Content if self.started && !held.is_empty() => vec![(PieceKind::Content, held)],
            FilterState::Reasoning => {
                let t = held.trim_end().to_string();
                if t.is_empty() { vec![] } else { vec![(PieceKind::Reasoning, t)] }
            }
            _ => Vec::new(),
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

    /// (content, reasoning) as streamed.
    fn filter2(marker: Option<&'static str>, json: bool, think: bool, chunks: &[&str]) -> (String, String) {
        let mut f = StreamFilter::new(marker, json, think);
        let (mut content, mut reasoning) = (String::new(), String::new());
        let mut pieces: Vec<(PieceKind, String)> = chunks.iter().flat_map(|c| f.push(c)).collect();
        pieces.extend(f.flush());
        for (kind, text) in pieces {
            match kind {
                PieceKind::Content => content.push_str(&text),
                PieceKind::Reasoning => reasoning.push_str(&text),
            }
        }
        (content, reasoning)
    }

    fn filter(marker: Option<&'static str>, json: bool, think: bool, chunks: &[&str]) -> String {
        filter2(marker, json, think, chunks).0
    }

    #[test]
    fn filter_streams_reasoning() {
        let (c, r) = filter2(None, false, false, &["<think>\nStep", " one.\n", "Step two.\n</th", "ink>\n\nDone."]);
        assert_eq!((c.as_str(), r.as_str()), ("Done.", "Step one.\nStep two."));
        // Reasoning opened by the prompt, cut off by max_tokens.
        let (c, r) = filter2(None, false, true, &["thinking about", " it\n\n"]);
        assert_eq!((c.as_str(), r.as_str()), ("", "thinking about it"));
        // Reasoning followed by a tool call.
        let (c, r) = filter2(Some("<tool_call>"), false, false, &["<think>plan</think>Calling.<tool_call>{}</tool_call>"]);
        assert_eq!((c.as_str(), r.as_str()), ("Calling.", "plan"));
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
