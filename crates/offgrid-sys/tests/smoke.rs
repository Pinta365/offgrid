use std::ffi::{CStr, CString};

unsafe fn cstr_or_null(p: *const std::ffi::c_char) -> String {
    if p.is_null() {
        "<null>".to_string()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

/// llama_backend_init, with the same AMD switchable-graphics layer workaround as offgrid::backend
/// (the layer hangs vkCreateInstance on hybrid-GPU Windows laptops).
unsafe fn backend_init() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        #[cfg(windows)]
        if std::env::var_os("DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1").is_none() {
            unsafe { std::env::set_var("DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1", "1") };
        }
    });
    unsafe { offgrid_sys::llama_backend_init() };
}

#[test]
fn enumerate_backend_devices() {
    unsafe {
        backend_init();
        offgrid_sys::ggml_backend_load_all();

        let count = offgrid_sys::ggml_backend_dev_count();
        println!("ggml_backend_dev_count() = {count}");
        assert!(count >= 1, "expected at least one backend device");

        for i in 0..count {
            let dev = offgrid_sys::ggml_backend_dev_get(i);
            let name = cstr_or_null(offgrid_sys::ggml_backend_dev_name(dev));
            let desc = cstr_or_null(offgrid_sys::ggml_backend_dev_description(dev));
            println!("device {i}: {name} :: {desc}");
        }
    }
}

#[test]
fn tokenize_with_model() {
    let path = match std::env::var("OFFGRID_TEST_MODEL") {
        Ok(p) => p,
        Err(_) => {
            println!("OFFGRID_TEST_MODEL not set; skipping tokenize_with_model");
            return;
        }
    };

    let cpath = CString::new(path.clone()).expect("model path contains NUL");
    let text = CString::new("Hello world").expect("literal contains no NUL");

    unsafe {
        backend_init();

        let mut params = offgrid_sys::llama_model_default_params();
        params.n_gpu_layers = 0;

        let model = offgrid_sys::llama_model_load_from_file(cpath.as_ptr(), params);
        assert!(!model.is_null(), "failed to load model at {path}");

        let vocab = offgrid_sys::llama_model_get_vocab(model);
        assert!(!vocab.is_null(), "model has no vocab");

        let mut tokens = vec![0i32; 128];
        let n = offgrid_sys::llama_tokenize(
            vocab,
            text.as_ptr(),
            text.as_bytes().len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            true,
            false,
        );
        println!("llama_tokenize(\"Hello world\") = {n} tokens");
        assert!(n > 0, "expected > 0 tokens, got {n}");

        offgrid_sys::llama_model_free(model);
    }
}
