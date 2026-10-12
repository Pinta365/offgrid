use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    // Windows doesn't search a DLL's own folder for its dependencies, so a host that
    // loads offgrid.dll by path (Deno, Python, ...) couldn't find llama/ggml. Delay-load
    // them; offgrid preloads them by full path from its own folder before first use.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var_os("CARGO_FEATURE_DYNAMIC_BACKENDS").is_some()
    {
        // MSVC names the DLLs without the `lib` prefix MinGW adds.
        let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
        let image = std::env::var_os("CARGO_FEATURE_IMAGE").is_some();
        let names = ["stable-diffusion", "mtmd", "llama"].into_iter().filter(|n| image || *n != "stable-diffusion");
        let dlls = names
            .map(|n| if msvc { format!("{n}.dll") } else { format!("lib{n}.dll") })
            .chain(["ggml.dll".to_string(), "ggml-base.dll".to_string()]);
        for dll in dlls {
            if msvc {
                println!("cargo:rustc-cdylib-link-arg=/DELAYLOAD:{dll}");
            } else {
                println!("cargo:rustc-cdylib-link-arg=-Wl,--delayload={dll}");
            }
        }
        println!("cargo:rustc-cdylib-link-arg={}", if msvc { "delayimp.lib" } else { "-ldelayimp" });
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        // Find libllama/libggml next to liboffgrid in a packaged folder.
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,$ORIGIN");
        // Never unmap on dlclose: ggml (which outlives us when backends are loaded)
        // keeps pointers to our log callback, and our init state must stay consistent.
        println!("cargo:rustc-cdylib-link-arg=-Wl,-z,nodelete");
    }

    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")).expect("cbindgen.toml");
    match cbindgen::generate_with_config(&crate_dir, config) {
        Ok(bindings) => {
            bindings.write_to_file(crate_dir.join("include/offgrid.h"));
        }
        // Don't break the build on header generation hiccups; the checked-in header stays.
        Err(e) => println!("cargo:warning=cbindgen failed: {e}"),
    }
}
