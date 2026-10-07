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
}
