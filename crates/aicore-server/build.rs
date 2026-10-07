fn main() {
    // Packaged layout: bin/aicore-server + lib/{libllama,libggml*,backends}.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN/../lib:$ORIGIN");
    }
}
