//! Audio input end-to-end. Needs model + projector pairs and the spike/audio set:
//!   OFFGRID_TEST_AUDIO_MODEL / OFFGRID_TEST_AUDIO_MMPROJ   audio-only (e.g. Ultravox 0.5 1B; not used for transcription)
//!   OFFGRID_TEST_OMNI_MODEL / OFFGRID_TEST_OMNI_MMPROJ     audio + vision (e.g. Qwen2.5-Omni-3B)
//!   OFFGRID_TEST_AUDIO, OFFGRID_TEST_IMAGES                sample dirs
//! Optional: OFFGRID_TEST_VL_MODEL / OFFGRID_TEST_VL_MMPROJ (vision-only) for the error test.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use offgrid::*;
use serde_json::json;

/// A session with a context that fits small GPUs (the tests' prompts are short).
fn session(model: Arc<Model>) -> ChatSession {
    ChatSession::new(model, &SessionParams { n_ctx: 2048, ..Default::default() }).unwrap()
}

fn load(model_var: &str, mmproj_var: &str) -> Option<(String, Arc<Model>)> {
    let (model, mmproj) = (std::env::var(model_var).ok()?, std::env::var(mmproj_var).ok()?);
    let name = PathBuf::from(&model).file_stem()?.to_string_lossy().into_owned();
    let m = Model::load(model, &ModelParams { mmproj: Some(mmproj.into()), ..Default::default() }).expect("loads");
    Some((name, m))
}

/// Configured audio models, loaded one at a time as the caller iterates.
fn audio_models() -> impl Iterator<Item = (String, Arc<Model>)> {
    [("OFFGRID_TEST_AUDIO_MODEL", "OFFGRID_TEST_AUDIO_MMPROJ"), ("OFFGRID_TEST_OMNI_MODEL", "OFFGRID_TEST_OMNI_MMPROJ")]
        .into_iter()
        .filter_map(|(m, p)| load(m, p))
}

fn sample(var: &str, name: &str) -> Option<Media> {
    let path = Path::new(&std::env::var(var).ok()?).join(name);
    path.is_file().then(|| Media::from_path(path).unwrap())
}

fn ask(chat: &mut ChatSession, messages: Vec<ChatMessage>) -> ChatResponse {
    let req = ChatRequest {
        max_tokens: Some(80),
        sampling: SamplingParams { temperature: 0.0, ..Default::default() },
        ..ChatRequest::new(messages)
    };
    chat.complete(&req, |_| true).unwrap()
}

// Omni only: Ultravox 1B understands the clip but doesn't reliably transcribe it.
#[test]
fn transcribes_speech_in_all_formats() {
    if let Some((name, model)) = load("OFFGRID_TEST_OMNI_MODEL", "OFFGRID_TEST_OMNI_MMPROJ") {
        assert!(model.info().audio, "[{name}]");
        let mut chat = session(model);
        for file in ["jfk.wav", "jfk.mp3", "jfk.flac"] {
            let Some(clip) = sample("OFFGRID_TEST_AUDIO", file) else { continue };
            assert_eq!(clip.kind, MediaKind::Audio, "{file} detected as audio");
            let r = ask(&mut chat, vec![ChatMessage::user_with_audio("Transcribe this audio exactly.", vec![clip])]);
            let text = r.content.to_lowercase();
            println!("[{name}] {file}: {:?}", r.content);
            assert!(text.contains("ask not") && text.contains("country"), "[{name}] {file}: {}", r.content);
        }
    }
}

#[test]
fn understands_and_reuses_audio() {
    for (name, model) in audio_models() {
        let Some(clip) = sample("OFFGRID_TEST_AUDIO", "jfk.wav") else { return };
        let mut chat = session(model.clone());
        let mut msgs =
            vec![ChatMessage::user_with_audio("Who is the speaker addressing? Answer in a few words.", vec![clip])];
        let r = ask(&mut chat, msgs.clone());
        println!("[{name}] addressed: {:?}", r.content);
        assert!(r.content.to_lowercase().contains("american"), "[{name}] {}", r.content);

        // Follow-up reuses the encoded audio and matches a fresh session exactly.
        msgs.push(r.to_message());
        msgs.push(ChatMessage::user("What does the speaker ask them to do? One short sentence."));
        let warm = ask(&mut chat, msgs.clone());
        drop(chat);
        let cold = ask(&mut session(model.clone()), msgs);
        println!(
            "[{name}] follow-up cached {}/{} tokens, prompt {:.0} ms (cold {:.0} ms): {:?}",
            warm.usage.cached_tokens,
            warm.usage.prompt_tokens,
            warm.usage.prompt_ms,
            cold.usage.prompt_ms,
            warm.content
        );
        assert_eq!(warm.content, cold.content, "[{name}] reused audio KV changed the answer");
        assert!(warm.usage.cached_tokens >= r.usage.prompt_tokens - 1, "[{name}] audio not reused");
    }
}

#[test]
fn image_and_audio_together() {
    let Some((name, model)) = load("OFFGRID_TEST_OMNI_MODEL", "OFFGRID_TEST_OMNI_MMPROJ") else { return };
    let (Some(img), Some(clip)) =
        (sample("OFFGRID_TEST_IMAGES", "red_square.png"), sample("OFFGRID_TEST_AUDIO", "jfk.wav"))
    else {
        return;
    };
    assert!(model.info().vision && model.info().audio);
    let mut chat = session(model);
    let r = ask(
        &mut chat,
        vec![ChatMessage::user_with_media(
            "First: what color is the image? Second: what nationality does the speaker address? Answer both briefly.",
            vec![img, clip],
        )],
    );
    println!("[{name}] image+audio: {:?}", r.content);
    let text = r.content.to_lowercase();
    assert!(text.contains("red") && text.contains("american"), "[{name}] {}", r.content);
}

#[test]
fn capability_and_format_errors() {
    // Wire format round trip.
    let wav = sample("OFFGRID_TEST_AUDIO", "jfk.wav").unwrap_or_else(|| Media::audio(b"RIFF\0\0\0\0WAVEfmt ".to_vec()));
    let (data, format) = wav.to_input_audio();
    let msg: ChatMessage = serde_json::from_value(json!({"role": "user", "content": [
        {"type": "text", "text": "Listen:"},
        {"type": "input_audio", "input_audio": {"data": data, "format": format}}
    ]}))
    .unwrap();
    assert_eq!(msg.media, vec![wav.clone()]);
    let back = serde_json::to_value(&msg).unwrap();
    assert_eq!(back["content"][1]["type"], "input_audio");
    assert_eq!(serde_json::from_value::<ChatMessage>(back).unwrap(), msg);
    assert!(
        serde_json::from_value::<ChatMessage>(json!({"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "ogg"}}]}))
        .is_err()
    );

    // Audio-only projector refuses images; vision-only refuses audio.
    if let Some((name, model)) = load("OFFGRID_TEST_AUDIO_MODEL", "OFFGRID_TEST_AUDIO_MMPROJ") {
        assert!(model.info().audio && !model.info().vision, "[{name}]");
        let mut chat = session(model);
        if let Some(img) = sample("OFFGRID_TEST_IMAGES", "red_square.png") {
            let err = chat
                .complete(&ChatRequest::new(vec![ChatMessage::user_with_images("x", vec![img.clone()])]), |_| true)
                .unwrap_err();
            assert!(err.to_string().contains("does not accept images"), "{err}");
            // Image bytes declared as audio.
            let fake = Media::audio(img.data);
            let err = chat
                .complete(&ChatRequest::new(vec![ChatMessage::user_with_audio("x", vec![fake])]), |_| true)
                .unwrap_err();
            assert!(matches!(err, Error::Media(_)), "{err}");
        }
        let err = chat
            .complete(
                &ChatRequest::new(vec![ChatMessage::user_with_audio("x", vec![Media::audio(b"RIFFgarbage".to_vec())])]),
                |_| true,
            )
            .unwrap_err();
        assert!(matches!(err, Error::Media(_)), "{err}");
        assert!(!ask(&mut chat, vec![ChatMessage::user("Say OK.")]).content.is_empty(), "session usable after errors");
    }
    if let Some((name, model)) = load("OFFGRID_TEST_VL_MODEL", "OFFGRID_TEST_VL_MMPROJ") {
        assert!(model.info().vision && !model.info().audio, "[{name}]");
        let mut chat = session(model);
        let err =
            chat.complete(&ChatRequest::new(vec![ChatMessage::user_with_audio("x", vec![wav])]), |_| true).unwrap_err();
        assert!(err.to_string().contains("does not accept audio"), "{err}");
    }
}
