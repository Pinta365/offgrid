# Source me: environment for cross-compiling to Windows (x86_64-pc-windows-gnullvm)
# with llvm-mingw (https://github.com/mstorsjo/llvm-mingw).
# LLVM_MINGW defaults to the newest ~/.local/share/llvm-mingw-* directory.
: "${LLVM_MINGW:=$(ls -d "$HOME"/.local/share/llvm-mingw-* 2>/dev/null | sort | tail -1)}"
if [ ! -x "$LLVM_MINGW/bin/x86_64-w64-mingw32-clang" ]; then
    echo "llvm-mingw not found (set LLVM_MINGW)" >&2
    return 1 2>/dev/null || exit 1
fi
export PATH="$LLVM_MINGW/bin:$PATH"
export CC_x86_64_pc_windows_gnullvm=x86_64-w64-mingw32-clang
export CXX_x86_64_pc_windows_gnullvm=x86_64-w64-mingw32-clang++
export AR_x86_64_pc_windows_gnullvm=llvm-ar
export CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_LINKER=x86_64-w64-mingw32-clang
# bindgen parses llama.h for the Windows target; point it at the mingw headers.
export BINDGEN_EXTRA_CLANG_ARGS_x86_64_pc_windows_gnullvm="--target=x86_64-w64-mingw32 --sysroot=$LLVM_MINGW/x86_64-w64-mingw32"
