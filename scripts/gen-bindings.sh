#!/usr/bin/env bash
#
# gen-bindings.sh [--check] [linux|windows]...
#
# Regenerate the committed bindings in crates/offgrid-sys/src/bindings (default: both targets):
# <os>.rs for llama.cpp and mtmd, sd_<os>.rs for stable-diffusion.cpp.
# --check compares instead of writing and fails if they are out of date.
# Run after bumping llama.cpp. Windows needs llvm-mingw (scripts/env-llvm-mingw.sh).
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DIR=crates/offgrid-sys/src/bindings

check=0
if [ "${1:-}" = "--check" ]; then
    check=1
    shift
fi
[ "$#" -gt 0 ] || set -- linux windows

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
fail=0
for os in "$@"; do
    case "$os" in
        linux) args=() ;;
        windows) args=(--target x86_64-pc-windows-gnullvm) ;;
        *) echo "unknown target: $os" >&2; exit 2 ;;
    esac
    out="$tmp/$os"
    mkdir -p "$out"
    (
        [ "$os" = windows ] && source "$ROOT/scripts/env-llvm-mingw.sh"
        OFFGRID_SYS_WRITE_BINDINGS="$out" cargo check -q -p offgrid-sys --features bindgen,image "${args[@]}"
    )
    for file in "$os.rs" "sd_$os.rs"; do
        if [ "$check" = 1 ]; then
            if diff -u "$DIR/$file" "$out/$file" >"$tmp/diff"; then
                echo "$file: up to date"
            else
                echo "$file: out of date, run scripts/gen-bindings.sh" >&2
                head -40 "$tmp/diff" >&2
                fail=1
            fi
        else
            cp "$out/$file" "$DIR/$file"
            echo "wrote $DIR/$file"
        fi
    done
done
exit "$fail"
