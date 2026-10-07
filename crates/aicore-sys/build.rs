use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=AICORE_NATIVE");

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
    // Portable mode: shared libs + backends as runtime-loaded modules + every CPU variant.
    let dynamic = env::var_os("CARGO_FEATURE_DYNAMIC_BACKENDS").is_some();
    let native = env::var("AICORE_NATIVE").map(|v| v == "1").unwrap_or(false);
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let unix_like = matches!(target_os.as_str(), "linux" | "freebsd" | "netbsd" | "openbsd" | "dragonfly");

    let mut cfg = cmake::Config::new(&llama_dir);

    // Always build llama.cpp in Release, even for debug cargo builds, so that
    // incremental development stays fast.
    cfg.profile("Release");

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
    cfg.define("GGML_VULKAN", if vulkan { "ON" } else { "OFF" });
    cfg.define("GGML_CUDA", if cuda { "ON" } else { "OFF" });

    if dynamic {
        cfg.define("BUILD_SHARED_LIBS", "ON");
        cfg.define("GGML_BACKEND_DL", "ON");
        cfg.define("GGML_NATIVE", "OFF");
        // Builds libggml-cpu-{x64,sse42,sandybridge,haswell,skylakex,zen4,...};
        // the best one for the running CPU is picked at load time.
        cfg.define("GGML_CPU_ALL_VARIANTS", "ON");
        if unix_like {
            // Find sibling libs next to each other; don't depend on the host libstdc++ version.
            cfg.define("CMAKE_INSTALL_RPATH", "$ORIGIN");
            cfg.define("CMAKE_BUILD_WITH_INSTALL_RPATH", "ON");
            // Each lib gets a private copy of libstdc++; --exclude-libs hides those symbols
            // so copies in different libs can't interpose each other (that crashed in
            // std::locale when loaded into Deno). Backends are MODULE targets, which use
            // a separate flags variable.
            // zig (cargo-zigbuild) always links its own libc++ statically and its
            // linker rejects these GNU flags, so only pass them to gcc/clang toolchains.
            // (Always set explicitly: CMakeCache would otherwise keep stale values.)
            let flags = if using_zig() { "" } else { "-static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL" };
            cfg.define("CMAKE_SHARED_LINKER_FLAGS", flags);
            // ggml dlopens every backend module to probe it and dlcloses the unused ones
            // (other CPU variants; Vulkan when no GPU). With zig's static libc++ those
            // modules leave exit-time handlers behind, so unmapping them crashes the
            // process at exit. nodelete keeps them mapped (cheap: mostly untouched pages).
            cfg.define("CMAKE_MODULE_LINKER_FLAGS", format!("{flags} -Wl,-z,nodelete").trim());
        }
    } else {
        cfg.define("BUILD_SHARED_LIBS", "OFF");
        cfg.define("GGML_NATIVE", if native { "ON" } else { "OFF" });
    }

    if has_ninja() {
        cfg.generator("Ninja");
    }

    let dst = cfg.build();

    if dynamic {
        // Only the core libs are linked; backends (ggml-cpu-*, ggml-vulkan, ...) are
        // dlopen'ed at runtime by ggml_backend_load_all_from_path().
        let lib_dir = find_lib(&dst, "llama", true)
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .expect("aicore-sys: libllama shared library not found");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        for name in ["llama", "ggml", "ggml-base"] {
            println!("cargo:rustc-link-lib=dylib={name}");
        }
        // cmake installs backend modules into bin/ (lib/ on some layouts).
        let backend_dir = find_lib(&dst, "ggml-cpu-x64", true)
            .or_else(|| find_lib(&dst, "ggml-cpu", true))
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| lib_dir.clone());
        // Dev fallback for locating backend modules, and metadata
        // (DEP_LLAMA_LIB_DIR / DEP_LLAMA_BACKEND_DIR) for packaging scripts.
        println!("cargo:rustc-env=AICORE_SYS_BACKEND_DIR={}", backend_dir.display());
        println!("cargo:lib_dir={}", lib_dir.display());
        println!("cargo:backend_dir={}", backend_dir.display());
    } else {
        println!("cargo:rustc-env=AICORE_SYS_BACKEND_DIR=");
        link_static(&dst, vulkan, cuda, &target_os);
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

/// True when the C/C++ compiler is zig's (e.g. under `cargo zigbuild`).
fn using_zig() -> bool {
    let target = env::var("TARGET").unwrap_or_default().replace('-', "_");
    ["CXX", &format!("CXX_{target}"), "CC", &format!("CC_{target}")]
        .iter()
        .filter_map(|k| env::var(k).ok())
        .any(|v| v.contains("zig"))
}

fn has_ninja() -> bool {
    Command::new("ninja")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Recursively look for library `name` (e.g. "ggml-base") under the cmake output dir.
/// `shared`: `.so`/`.dylib`/import `.lib`; otherwise a static archive (`.a`/`.lib`).
fn find_lib(root: &Path, name: &str, shared: bool) -> Option<PathBuf> {
    let candidates = if shared {
        vec![format!("lib{name}.so"), format!("lib{name}.dylib"), format!("{name}.lib")]
    } else {
        vec![format!("lib{name}.a"), format!("{name}.lib"), format!("lib{name}.lib"), format!("{name}.a")]
    };
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Some(file_name) = path.file_name().and_then(|s| s.to_str())
                && candidates.iter().any(|c| c == file_name)
            {
                return Some(path);
            }
        }
    }
    None
}

fn link_static(dst: &Path, vulkan: bool, cuda: bool, target_os: &str) {
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
        match find_lib(dst, name, false) {
            Some(path) => {
                if let Some(dir) = path.parent()
                    && !search_dirs.iter().any(|d| d == dir)
                {
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
    match target_os {
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
}
