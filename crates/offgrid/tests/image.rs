//! Image generation end-to-end (feature `image`). Needs a Z-Image-Turbo set:
//!   OFFGRID_TEST_IMAGE_MODEL         diffusion model (z_image_turbo-Q4_K.gguf)
//!   OFFGRID_TEST_IMAGE_VAE           VAE (ae-f16.gguf)
//!   OFFGRID_TEST_IMAGE_TEXT_ENCODER  Qwen3-4B GGUF
//! Optional: OFFGRID_TEST_MODEL (a chat model) for the concurrency test.
#![cfg(feature = "image")]

use offgrid::*;

fn params() -> Option<ImageModelParams> {
    let var = |k: &str| std::env::var(k).ok().map(Into::into);
    Some(ImageModelParams {
        diffusion_model: Some(var("OFFGRID_TEST_IMAGE_MODEL")?),
        vae: Some(var("OFFGRID_TEST_IMAGE_VAE")?),
        text_encoder: Some(var("OFFGRID_TEST_IMAGE_TEXT_ENCODER")?),
        ..Default::default()
    })
}

fn request(prompt: &str) -> ImageRequest {
    ImageRequest {
        prompt: prompt.into(),
        width: 512,
        height: 512,
        steps: Some(8),
        cfg_scale: Some(1.0),
        seed: Some(42),
        ..Default::default()
    }
}

#[test]
fn generates_png_deterministically_and_cancels() {
    let Some(params) = params() else { return };
    let mut model = ImageModel::load(&params).expect("image model loads");

    let mut events = Vec::new();
    let img = model
        .generate(&request("A red apple on a white table"), |p| {
            events.push(p);
            true
        })
        .unwrap();
    assert_eq!((img.width, img.height, img.rgb.len()), (512, 512, 512 * 512 * 3));
    let sampled: Vec<u32> = events
        .iter()
        .filter_map(|p| if let ImageProgress::Sampling { step, .. } = p { Some(*step) } else { None })
        .collect();
    assert_eq!(sampled.last(), Some(&8), "{events:?}");
    assert!(events.iter().any(|p| matches!(p, ImageProgress::Loading { .. })), "{events:?}");
    let last_sampling = events.iter().rposition(|p| matches!(p, ImageProgress::Sampling { .. })).unwrap();
    assert!(events[last_sampling + 1..].iter().all(|p| matches!(p, ImageProgress::Decoding { .. })), "{events:?}");
    // Not a flat image: a red apple on white has plenty of distinct colours.
    let distinct: std::collections::HashSet<&[u8]> = img.rgb.as_chunks::<3>().0.iter().map(|px| &px[..]).collect();
    assert!(distinct.len() > 1000, "{} distinct colours", distinct.len());

    let png = img.to_png();
    let decoder = png::Decoder::new(std::io::Cursor::new(&png));
    let reader = decoder.read_info().unwrap();
    assert_eq!((reader.info().width, reader.info().height), (512, 512));

    let again = model.generate(&request("A red apple on a white table"), |_| true).unwrap();
    assert_eq!(again, img, "same seed, same image");

    let err = model
        .generate(&request("A blue boat"), |p| !matches!(p, ImageProgress::Sampling { step: 2.., .. }))
        .unwrap_err();
    assert!(matches!(err, Error::Cancelled), "{err}");
    let after = model.generate(&request("A blue boat"), |_| true).unwrap();
    assert_eq!(after.rgb.len(), 512 * 512 * 3, "model still works after a cancel");

    let bad = ImageRequest { width: 500, ..request("x") };
    assert!(matches!(model.generate(&bad, |_| true), Err(Error::InvalidArgument(_))));
}

/// Image generation on one thread while a chat session answers on another.
#[test]
fn generates_while_chatting() {
    let (Some(params), Ok(chat_path)) = (params(), std::env::var("OFFGRID_TEST_MODEL")) else { return };
    let chat_model = Model::load(chat_path, &ModelParams::default()).unwrap();
    let mut chat = ChatSession::new(chat_model, &SessionParams { n_ctx: 2048, ..Default::default() }).unwrap();
    let mut model = ImageModel::load(&params).unwrap();

    let painter = std::thread::spawn(move || model.generate(&request("A lighthouse at dusk"), |_| true));
    let req = ChatRequest {
        max_tokens: Some(20),
        sampling: SamplingParams { temperature: 0.0, ..Default::default() },
        ..ChatRequest::new(vec![ChatMessage::user("What is the capital of France? One word.")])
    };
    let answer = chat.complete(&req, |_| true).unwrap();
    assert!(answer.content.contains("Paris"), "{}", answer.content);
    let img = painter.join().unwrap().unwrap();
    assert_eq!(img.rgb.len(), 512 * 512 * 3);
}
