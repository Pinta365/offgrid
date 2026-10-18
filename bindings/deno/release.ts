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
  version: "0.4.4",
  baseUrl: "https://github.com/Pinta365/offgrid/releases/download/v0.4.4",
  assets: {
    "linux-x86_64": {
      file: "offgrid-0.4.4-linux-x86_64.tar.gz",
      sha256: "",
      lib: "offgrid-0.4.4-linux-x86_64/lib/liboffgrid.so",
    },
    "windows-x86_64": {
      file: "offgrid-0.4.4-windows-x86_64.tar.gz",
      sha256: "",
      lib: "offgrid-0.4.4-windows-x86_64/bin/offgrid.dll",
    },
  } as Record<string, ReleaseAsset>,
};
