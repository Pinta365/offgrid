#!/usr/bin/env bash
#
# deno-release.sh <version> [asset-dir]
#
# Write bindings/deno/release.ts for <version>. With [asset-dir] (holding the release's
# offgrid-<version>-{linux,windows}-x86_64.tar.gz), fill in their SHA-256 checksums so the
# published binding can download them; without it the checksums stay empty (repo builds).
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$1"
ASSETS="${2:-}"

sum() {
    local file="$ASSETS/offgrid-$VERSION-$1-x86_64.tar.gz"
    if [ -z "$ASSETS" ]; then
        echo ""
    elif [ -f "$file" ]; then
        sha256sum "$file" | cut -d' ' -f1
    else
        echo "missing $file" >&2
        exit 1
    fi
}
LINUX_SHA="$(sum linux)"
WINDOWS_SHA="$(sum windows)"

cat >"$ROOT/bindings/deno/release.ts" <<EOF
// Release this version of the bindings downloads the native library from. The publish workflow
// fills in the checksums (scripts/deno-release.sh) before publishing to JSR; with empty checksums
// (a repository checkout) nothing is downloaded.

/** A prebuilt native package attached to the GitHub release. */
export interface ReleaseAsset {
  /** Archive file name in the release. */
  file: string;
  /** Hex SHA-256 of the archive. */
  sha256: string;
  /** Path of the library inside the extracted archive. */
  lib: string;
}

export const RELEASE = {
  version: "$VERSION",
  baseUrl: "https://github.com/Pinta365/offgrid/releases/download/v$VERSION",
  assets: {
    "linux-x86_64": {
      file: "offgrid-$VERSION-linux-x86_64.tar.gz",
      sha256: "$LINUX_SHA",
      lib: "offgrid-$VERSION-linux-x86_64/lib/liboffgrid.so",
    },
    "windows-x86_64": {
      file: "offgrid-$VERSION-windows-x86_64.tar.gz",
      sha256: "$WINDOWS_SHA",
      lib: "offgrid-$VERSION-windows-x86_64/bin/offgrid.dll",
    },
  } as Record<string, ReleaseAsset>,
};
EOF
echo "wrote bindings/deno/release.ts for $VERSION${ASSETS:+ with checksums}"
