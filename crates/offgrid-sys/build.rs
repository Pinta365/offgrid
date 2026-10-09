use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=OFFGRID_NATIVE");

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
    // Other headers bindgen reads, and the pinned submodule commit (a bump changes sources
    // cmake must rebuild even when these headers don't change).
    for header in ["tools/mtmd/mtmd.h", "tools/mtmd/mtmd-helper.h", "ggml/include/ggml.h"] {
        println!("cargo:rerun-if-changed={}", llama_dir.join(header).display());
    }
    let submodule_head = manifest_dir.join("../../.git/modules/vendor/llama.cpp/HEAD");
    if submodule_head.exists() {
        println!("cargo:rerun-if-changed={}", submodule_head.display());
    }

    let vulkan = env::var_os("CARGO_FEATURE_VULKAN").is_some();
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    // Portable mode: shared libs + backends as runtime-loaded modules + every CPU variant.
    let dynamic = env::var_os("CARGO_FEATURE_DYNAMIC_BACKENDS").is_some();
    let native = env::var("OFFGRID_NATIVE").map(|v| v == "1").unwrap_or(false);
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
    // Multimodal (vision) library, built standalone without the rest of tools/.
    cfg.define("LLAMA_BUILD_MTMD", "ON");
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
        } else if target_os == "windows" {
            // Note: build Windows with a real MinGW toolchain (llvm-mingw, see
            // scripts/env-llvm-mingw.sh). ggml relies on MinGW auto-export, and zig's
            // CRT object naming makes lld also export atexit/_CRT_INIT -> duplicate symbols.
            // (Set explicitly so CMakeCache can't keep stale flags.)
            // -static: bake libc++/libunwind into each DLL (only the OS UCRT stays dynamic).
            cfg.define("CMAKE_SHARED_LINKER_FLAGS", "-static");
            cfg.define("CMAKE_MODULE_LINKER_FLAGS", "-static");
        }
    } else {
        cfg.define("BUILD_SHARED_LIBS", "OFF");
        cfg.define("GGML_NATIVE", if native { "ON" } else { "OFF" });
    }

    if vulkan && target_os == "windows" && !cfg!(windows) {
        cross_windows_vulkan(&mut cfg, &manifest_dir);
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
            .expect("offgrid-sys: libllama shared library not found");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        for name in ["mtmd", "llama", "ggml", "ggml-base"] {
            println!("cargo:rustc-link-lib=dylib={name}");
        }
        // cmake installs backend modules into bin/ (lib/ on some layouts).
        let backend_dir = find_lib(&dst, "ggml-cpu-x64", true)
            .or_else(|| find_lib(&dst, "ggml-cpu", true))
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| lib_dir.clone());
        // Dev fallback for locating backend modules, and metadata
        // (DEP_LLAMA_LIB_DIR / DEP_LLAMA_BACKEND_DIR) for packaging scripts.
        println!("cargo:rustc-env=OFFGRID_SYS_BACKEND_DIR={}", backend_dir.display());
        println!("cargo:lib_dir={}", lib_dir.display());
        println!("cargo:backend_dir={}", backend_dir.display());
    } else {
        println!("cargo:rustc-env=OFFGRID_SYS_BACKEND_DIR=");
        link_static(&dst, vulkan, cuda, &target_os);
    }

    let bindings = bindgen::Builder::default()
        .header_contents(
            "offgrid_sys.h",
            "#include \"llama.h\"\n#include \"mtmd.h\"\n#include \"mtmd-helper.h\"\n",
        )
        .clang_arg(format!("-I{}", llama_dir.join("include").display()))
        .clang_arg(format!("-I{}", llama_dir.join("ggml/include").display()))
        .clang_arg(format!("-I{}", llama_dir.join("tools/mtmd").display()))
        .allowlist_function("llama_.*|ggml_backend_.*|ggml_log_.*|mtmd_.*")
        .allowlist_type("llama_.*|ggml_.*|mtmd_.*")
        .allowlist_var("LLAMA_.*")
        .generate()
        .expect("offgrid-sys: bindgen failed to generate bindings");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("offgrid-sys: failed to write bindings.rs");
}

/// Cross-compiling ggml-vulkan for Windows from Linux without a Vulkan SDK: use the
/// host's (platform-neutral) Vulkan headers and glslc, and generate a MinGW import
/// library for vulkan-1.dll from windows/vulkan-1.def with llvm-dlltool.
/// Override with OFFGRID_VULKAN_INCLUDE / OFFGRID_GLSLC.
fn cross_windows_vulkan(cfg: &mut cmake::Config, manifest_dir: &Path) {
    println!("cargo:rerun-if-env-changed=OFFGRID_VULKAN_INCLUDE");
    println!("cargo:rerun-if-env-changed=OFFGRID_GLSLC");
    let def = manifest_dir.join("windows/vulkan-1.def");
    println!("cargo:rerun-if-changed={}", def.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let implib = out_dir.join("libvulkan-1.a");
    let status = Command::new("llvm-dlltool")
        .args(["-m", "i386:x86-64", "-d"])
        .arg(&def)
        .arg("-l")
        .arg(&implib)
        .status()
        .expect("offgrid-sys: llvm-dlltool not found (source scripts/env-llvm-mingw.sh)");
    assert!(status.success(), "offgrid-sys: llvm-dlltool failed");

    // Copy just the Vulkan header dirs: adding all of /usr/include would pull host libc
    // headers into the Windows compile.
    let include = env::var("OFFGRID_VULKAN_INCLUDE").map(PathBuf::from).unwrap_or_else(|_| {
        let dst = out_dir.join("vulkan-include");
        for sub in ["vulkan", "vk_video", "spirv"] {
            let src = Path::new("/usr/include").join(sub);
            if src.is_dir() {
                copy_dir(&src, &dst.join(sub));
            }
        }
        dst
    });
    let glslc = env::var("OFFGRID_GLSLC").unwrap_or_else(|_| "glslc".into());
    cfg.define("Vulkan_INCLUDE_DIR", include);
    cfg.define("Vulkan_LIBRARY", &implib);
    cfg.define("Vulkan_GLSLC_EXECUTABLE", which(&glslc).unwrap_or_else(|| glslc.into()));
    println!("cargo:rustc-link-search=native={}", out_dir.display());
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("create dir");
    for entry in std::fs::read_dir(src).expect("read dir").flatten() {
        let (from, to) = (entry.path(), dst.join(entry.file_name()));
        if from.is_dir() {
            copy_dir(&from, &to);
        } else {
            std::fs::copy(&from, &to).expect("copy header");
        }
    }
}

fn which(prog: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|paths| env::split_paths(&paths).map(|d| d.join(prog)).find(|p| p.is_file()))
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
        // .so / .dylib / MinGW import lib / MSVC import lib
        vec![format!("lib{name}.so"), format!("lib{name}.dylib"), format!("lib{name}.dll.a"), format!("{name}.lib")]
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

/// `libvendor-*.a` archives in the cmake build tree (MSVC: `vendor-*.lib`).
fn find_vendor_archives(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if path.is_dir() {
                stack.push(path);
            } else if (name.starts_with("libvendor-") && name.ends_with(".a"))
                || (name.starts_with("vendor-") && name.ends_with(".lib"))
            {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

fn link_static(dst: &Path, vulkan: bool, cuda: bool, target_os: &str) {
    // Dependency order (left-to-right): consumers before providers.
    let mut static_libs: Vec<&str> = vec!["mtmd", "llama", "ggml"];
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
                println!("cargo:warning=offgrid-sys: static library for `{name}` not found under {}", dst.display());
            }
        }
    }

    for dir in &search_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for name in &static_libs {
        println!("cargo:rustc-link-lib=static={name}");
    }
    // mtmd's private helper archives (vendor/hash, ...) are not installed; take them from
    // the build tree. Linked after mtmd, which depends on them.
    for archive in find_vendor_archives(&dst.join("build")) {
        let stem = archive.file_stem().unwrap().to_string_lossy();
        println!("cargo:rustc-link-search=native={}", archive.parent().unwrap().display());
        println!("cargo:rustc-link-lib=static={}", stem.trim_start_matches("lib"));
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
