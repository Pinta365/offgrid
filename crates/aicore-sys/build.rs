use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=AICORE_NATIVE");
    println!("cargo:rerun-if-env-changed=AICORE_TEST_MODEL");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let llama_dir = {
        let p = manifest_dir.join("../../vendor/llama.cpp");
        p.canonicalize().unwrap_or(p)
    };

    println!(
        "cargo:rerun-if-changed={}",
        llama_dir.join("include/llama.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        llama_dir.join("ggml/include/ggml-backend.h").display()
    );

    let vulkan = env::var_os("CARGO_FEATURE_VULKAN").is_some();
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let native = env::var("AICORE_NATIVE").map(|v| v == "1").unwrap_or(false);

    let mut cfg = cmake::Config::new(&llama_dir);

    // Always build llama.cpp in Release, even for debug cargo builds, so that
    // incremental development stays fast.
    cfg.profile("Release");

    // Static-only, minimal feature surface.
    cfg.define("BUILD_SHARED_LIBS", "OFF");
    cfg.define("LLAMA_BUILD_TESTS", "OFF");
    cfg.define("LLAMA_BUILD_EXAMPLES", "OFF");
    cfg.define("LLAMA_BUILD_TOOLS", "OFF");
    cfg.define("LLAMA_BUILD_SERVER", "OFF");
    // `app` (the unified `llama` binary) is a separate target added in this
    // commit; it defaults ON for standalone builds and pulls in build-info.h.
    cfg.define("LLAMA_BUILD_APP", "OFF");
    // `common` is only used by tools/examples/app; not needed for the libraries.
    cfg.define("LLAMA_BUILD_COMMON", "OFF");
    cfg.define("LLAMA_CURL", "OFF");
    cfg.define("GGML_OPENMP", "OFF");
    cfg.define("GGML_NATIVE", if native { "ON" } else { "OFF" });
    cfg.define("GGML_VULKAN", if vulkan { "ON" } else { "OFF" });
    cfg.define("GGML_CUDA", if cuda { "ON" } else { "OFF" });

    if has_ninja() {
        cfg.generator("Ninja");
    }

    let dst = cfg.build();

    // Dependency order (left-to-right): consumers before providers.
    let mut static_libs: Vec<&str> = vec!["llama", "ggml"];
    if vulkan {
        static_libs.push("ggml-vulkan");
    }
    if cuda {
        static_libs.push("ggml-cuda");
    }
    static_libs.push("ggml-cpu");
    // ggml-base is the leaf every other ggml archive depends on; keep it last.
    static_libs.push("ggml-base");

    let mut search_dirs: Vec<PathBuf> = Vec::new();
    for name in &static_libs {
        match find_static_lib(&dst, name) {
            Some(path) => {
                if let Some(dir) = path.parent()
                    && !search_dirs.iter().any(|d| d == dir) {
                        search_dirs.push(dir.to_path_buf());
                    }
            }
            None => {
                // CUDA may be requested on a machine without the toolkit; let the
                // linker report the failure rather than silently succeeding.
                println!("cargo:warning=aicore-sys: static library for `{name}` not found under {}", dst.display());
            }
        }
    }

    for dir in &search_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for name in &static_libs {
        println!("cargo:rustc-link-lib=static={name}");
    }

    // System dependencies.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" | "freebsd" | "netbsd" | "openbsd" | "dragonfly" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=dl");
            if vulkan {
                println!("cargo:rustc-link-lib=vulkan");
            }
        }
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=c++");
            if vulkan {
                println!("cargo:rustc-link-lib=vulkan");
            }
        }
        "windows" => {
            if vulkan {
                println!("cargo:rustc-link-lib=vulkan-1");
            }
            println!("cargo:rustc-link-lib=userenv");
            println!("cargo:rustc-link-lib=ws2_32");
            println!("cargo:rustc-link-lib=advapi32");
        }
        _ => {}
    }

    let bindings = bindgen::Builder::default()
        .header(llama_dir.join("include/llama.h").to_string_lossy())
        .clang_arg(format!("-I{}", llama_dir.join("include").display()))
        .clang_arg(format!("-I{}", llama_dir.join("ggml/include").display()))
        .allowlist_function("llama_.*|ggml_backend_.*|ggml_log_.*")
        .allowlist_type("llama_.*|ggml_.*")
        .allowlist_var("LLAMA_.*")
        .generate()
        .expect("aicore-sys: bindgen failed to generate bindings");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("aicore-sys: failed to write bindings.rs");
}

fn has_ninja() -> bool {
    Command::new("ninja")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Recursively look for a static archive for `name` (e.g. "ggml-base") underneath
/// the cmake output directory, accepting MSVC (`.lib`), MinGW/MSYS (`lib*.a`),
/// and Unix (`lib*.a`) naming.
fn find_static_lib(root: &Path, name: &str) -> Option<PathBuf> {
    let candidates = [
        format!("lib{name}.a"),
        format!("{name}.lib"),
        format!("lib{name}.lib"),
        format!("{name}.a"),
    ];
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if let Some(file_name) = path.file_name().and_then(|s| s.to_str())
                && candidates.iter().any(|c| c == file_name) {
                    return Some(path);
                }
        }
    }
    None
}
