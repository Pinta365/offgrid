#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// stable-diffusion.cpp (image generation).
#[cfg(feature = "image")]
pub mod sd {
    #![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/sd_bindings.rs"));
}

/// Build-time directory of the ggml backend modules when built with `dynamic-backends`
/// (empty otherwise). Used as a development fallback only.
pub const BUILD_BACKEND_DIR: &str = env!("OFFGRID_SYS_BACKEND_DIR");
