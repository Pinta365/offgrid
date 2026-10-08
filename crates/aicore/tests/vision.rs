//! Vision (image input) end-to-end. Needs model + projector pairs:
//!   AICORE_TEST_VL_MODEL / AICORE_TEST_VL_MMPROJ     (e.g. Qwen2.5-VL-3B, M-RoPE)
//!   AICORE_TEST_VL2_MODEL / AICORE_TEST_VL2_MMPROJ   (e.g. Gemma 3 4B)
//!   AICORE_TEST_IMAGES                               dir with the spike/images test set
//! Skipped when unset.

use std::path::PathBuf;
use std::sync::Arc;

use aicore::*;
use serde_json::json;

fn vision_models() -> Vec<(String, Arc<Model>)> {
    [("AICORE_TEST_VL_MODEL", "AICORE_TEST_VL_MMPROJ"), ("AICORE_TEST_VL2_MODEL", "AICORE_TEST_VL2_MMPROJ")]
        .iter()
        .filter_map(|(m, p)| {
            let (model, mmproj) = (std::env::var(m).ok()?, std::env::var(p).ok()?);
            let name = PathBuf::from(&model).file_stem()?.to_string_lossy().into_owned();
            let params = ModelParams { mmproj: Some(mmproj.into()), ..Default::default() };
            Some((name, Model::load(model, &params).expect("vision model loads")))
        })
        .collect()
}

fn image(name: &str) -> Image {
    let dir = std::env::var("AICORE_TEST_IMAGES").expect("AICORE_TEST_IMAGES");
    Image::from_path(PathBuf::from(dir).join(name)).unwrap()
}

fn ask(chat: &mut ChatSession, messages: Vec<ChatMessage>) -> ChatResponse {
    let req = ChatRequest {
        max_tokens: Some(60),
        sampling: SamplingParams { temperature: 0.0, ..Default::default() },
        ..ChatRequest::new(messages)
    };
    let mut streamed = String::new();
    let r = chat.complete(&req, |p| { streamed.push_str(p); true }).unwrap();
    assert_eq!(streamed.trim(), r.content);
    r
}

fn one(chat: &mut ChatSession, img: &str, question: &str) -> String {
    ask(chat, vec![ChatMessage::user_with_images(question, vec![image(img)])]).content.to_lowercase()
}

#[test]
fn recognizes_test_images() {
    for (name, model) in vision_models() {
        assert!(model.info().vision);
        let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
        let color = one(&mut chat, "red_square.png", "What color is this image? One word.");
        let count = one(&mut chat, "three_circles.png", "How many circles are in this image? Answer with a number.");
        let text = one(&mut chat, "text_hello.png", "What text is written in this image?");
        let jpeg = one(&mut chat, "stripes.jpg", "Which two colors are in this image?");
        println!("[{name}] color={color:?} count={count:?} text={text:?} jpeg={jpeg:?}");
        assert!(color.contains("red"), "[{name}] {color}");
        assert!(count.contains('3') || count.contains("three"), "[{name}] {count}");
        assert!(text.contains("hello") && text.contains("42"), "[{name}] {text}");
        assert!(jpeg.contains("green") && jpeg.contains("yellow"), "[{name}] {jpeg}");
    }
}

#[test]
fn multiple_images_and_followups() {
    for (name, model) in vision_models() {
        let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
        let mut messages = vec![ChatMessage::user_with_images(
            "The first image shows a single color, the second shows text. What is the color, and what is the text?",
            vec![image("red_square.png"), image("text_hello.png")],
        )];
        let r = ask(&mut chat, messages.clone());
        println!("[{name}] two images: {:?}", r.content);
        let lower = r.content.to_lowercase();
        assert!(lower.contains("red") && lower.contains("hello"), "[{name}] {}", r.content);

        // Follow-up about the earlier images (history is re-encoded each turn).
        messages.push(r.to_message());
        messages.push(ChatMessage::user("Repeat just the number from the text image."));
        let r = ask(&mut chat, messages);
        println!("[{name}] follow-up: {:?}", r.content);
        assert!(r.content.contains("42"), "[{name}] {}", r.content);

        // A text-only request on the same session after media: correct, and reuse resumes.
        let r = ask(&mut chat, vec![ChatMessage::user("What is 6 times 7? Just the number.")]);
        assert!(r.content.contains("42"), "[{name}] {}", r.content);
        let r2 = ask(&mut chat, vec![ChatMessage::user("What is 6 times 7? Just the number.")]);
        assert!(r2.usage.cached_tokens as usize >= r2.usage.prompt_tokens as usize - 1, "[{name}] text prefix reuse");
    }
}

#[test]
fn errors_and_wire_format() {
    // OpenAI parts round-trip through serde.
    let png = image_or_dummy();
    let wire = json!({"role": "user", "content": [
        {"type": "text", "text": "Describe:"},
        {"type": "image_url", "image_url": {"url": png.to_data_url()}}
    ]});
    let msg: ChatMessage = serde_json::from_value(wire).unwrap();
    assert_eq!(msg.images.len(), 1);
    assert_eq!(msg.content, format!("Describe:\n{}", media_marker()));
    let back = serde_json::to_value(&msg).unwrap();
    assert_eq!(back["content"][0]["text"], "Describe:");
    assert_eq!(back["content"][1]["type"], "image_url");
    let again: ChatMessage = serde_json::from_value(back).unwrap();
    assert_eq!(again, msg);
    // Remote URLs and unknown parts are rejected.
    assert!(serde_json::from_value::<ChatMessage>(json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]})).is_err());
    assert!(serde_json::from_value::<ChatMessage>(json!({"role": "user", "content": [{"type": "input_audio"}]})).is_err());

    // Image sent to a text-only model; corrupt image to a vision model.
    if let Ok(path) = std::env::var("AICORE_TEST_MODEL") {
        let model = Model::load(path, &ModelParams::default()).unwrap();
        let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
        let req = ChatRequest::new(vec![ChatMessage::user_with_images("hi", vec![png.clone()])]);
        let err = chat.complete(&req, |_| true).unwrap_err();
        assert!(err.to_string().contains("vision projector"), "{err}");
    }
    for (name, model) in vision_models().into_iter().take(1) {
        let mut chat = ChatSession::new(model, &SessionParams::default()).unwrap();
        let req = ChatRequest::new(vec![ChatMessage::user_with_images("hi", vec![Image::from_bytes(b"not an image".to_vec())])]);
        let err = chat.complete(&req, |_| true).unwrap_err();
        assert!(matches!(err, Error::Image(_)), "[{name}] {err}");
        // Session still works afterwards.
        let r = ask(&mut chat, vec![ChatMessage::user("Say OK.")]);
        assert!(!r.content.is_empty());
    }
}

fn image_or_dummy() -> Image {
    std::env::var("AICORE_TEST_IMAGES")
        .map(|d| Image::from_path(PathBuf::from(d).join("red_square.png")).unwrap())
        .unwrap_or_else(|_| Image::from_bytes(vec![0x89, b'P', b'N', b'G', 0, 1, 2]))
}

/// KV-cache reuse across image prompts: identical images are reused (not re-encoded),
/// changed images are not, and answers match a fresh session exactly.
#[test]
fn image_prompt_cache_reuse() {
    for (name, model) in vision_models() {
        let mut chat = ChatSession::new(model.clone(), &SessionParams::default()).unwrap();
        let q1 = vec![ChatMessage::user_with_images("What text is in this image?", vec![image("text_hello.png")])];
        let r1 = ask(&mut chat, q1.clone());

        // 1. Follow-up: the image and first turn are reused, answer identical to a fresh session.
        let mut q2 = q1.clone();
        q2.push(r1.to_message());
        q2.push(ChatMessage::user("Now write only the number from that text, twice, separated by a space."));
        let warm = ask(&mut chat, q2.clone());
        let cold = ask(&mut ChatSession::new(model.clone(), &SessionParams::default()).unwrap(), q2.clone());
        println!(
            "[{name}] follow-up: cached {}/{} tokens, prompt {:.0} ms (cold {:.0} ms): {:?}",
            warm.usage.cached_tokens, warm.usage.prompt_tokens, warm.usage.prompt_ms, cold.usage.prompt_ms, warm.content
        );
        assert_eq!(warm.content, cold.content, "[{name}] reused KV changed the answer");
        assert!(warm.usage.cached_tokens >= r1.usage.prompt_tokens - 1, "[{name}] first turn incl. image not reused");
        assert_eq!(cold.usage.cached_tokens, 0);
        assert!(warm.content.contains("42"), "[{name}] {}", warm.content);

        // 2. Same question, different image: must not reuse the old image.
        let q3 = vec![ChatMessage::user_with_images("What text is in this image?", vec![image("red_square.png")])];
        let mut chat2 = ChatSession::new(model.clone(), &SessionParams::default()).unwrap();
        ask(&mut chat2, q1.clone());
        let swapped = ask(&mut chat2, q3.clone());
        let fresh = ask(&mut ChatSession::new(model.clone(), &SessionParams::default()).unwrap(), q3);
        println!("[{name}] swapped image: cached {} tokens: {:?}", swapped.usage.cached_tokens, swapped.content);
        assert_eq!(swapped.content, fresh.content, "[{name}] stale image leaked into the answer");
        assert!(!swapped.content.to_uppercase().contains("HELLO"), "[{name}] answered about the old image");
        assert!(swapped.usage.cached_tokens < 40, "[{name}] reused past the changed image");

        // 3. Two images, swap only the second: the first is reused.
        let two = |second: &str| {
            vec![ChatMessage::user_with_images(
                "Describe each of the two images in a few words.",
                vec![image("three_circles.png"), image(second)],
            )]
        };
        let mut chat3 = ChatSession::new(model.clone(), &SessionParams::default()).unwrap();
        let a = ask(&mut chat3, two("red_square.png"));
        let b = ask(&mut chat3, two("stripes.jpg"));
        let b_fresh = ask(&mut ChatSession::new(model.clone(), &SessionParams::default()).unwrap(), two("stripes.jpg"));
        println!("[{name}] second image swapped: cached {}/{} tokens: {:?}", b.usage.cached_tokens, b.usage.prompt_tokens, b.content);
        assert!(b.usage.cached_tokens > 20, "[{name}] first image not reused");
        assert!(b.usage.cached_tokens < b.usage.prompt_tokens - 20, "[{name}] second image wrongly reused");
        assert_eq!(b.content, b_fresh.content, "[{name}] partial reuse changed the answer");
        assert_ne!(a.content, b.content);
    }
}
