use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    // Windows doesn't search a DLL's own folder for its dependencies, so a host that
    // loads aicore.dll by path (Deno, Python, ...) couldn't find llama/ggml. Delay-load
    // them; aicore preloads them by full path from its own folder before first use.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var_os("CARGO_FEATURE_DYNAMIC_BACKENDS").is_some()
    {
        for dll in ["libllama.dll", "ggml.dll", "ggml-base.dll"] {
            println!("cargo:rustc-cdylib-link-arg=-Wl,--delayload={dll}");
        }
        println!("cargo:rustc-cdylib-link-arg=-ldelayimp");
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        // Find libllama/libggml next to libaicore in a packaged folder.
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,$ORIGIN");
        // Never unmap on dlclose: ggml (which outlives us when backends are loaded)
        // keeps pointers to our log callback, and our init state must stay consistent.
        println!("cargo:rustc-cdylib-link-arg=-Wl,-z,nodelete");
    }

    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")).expect("cbindgen.toml");
    match cbindgen::generate_with_config(&crate_dir, config) {
        Ok(bindings) => {
            bindings.write_to_file(crate_dir.join("include/aicore.h"));
        }
        // Don't break the build on header generation hiccups; the checked-in header stays.
        Err(e) => println!("cargo:warning=cbindgen failed: {e}"),
    }
}
