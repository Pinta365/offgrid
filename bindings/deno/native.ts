// Locating the native library: OFFGRID_LIB, a repository build, or the release package
// downloaded once into a per-user cache.

import { UntarStream } from "@std/tar/untar-stream";
import { dirname, isAbsolute, join, normalize } from "@std/path";
import { RELEASE, type ReleaseAsset } from "./release.ts";

const LIB_FILE = Deno.build.os === "windows"
  ? "offgrid.dll"
  : Deno.build.os === "darwin"
  ? "liboffgrid.dylib"
  : "liboffgrid.so";

/** `linux-x86_64`, `windows-x86_64`, ... */
export function platform(): string {
  return `${Deno.build.os}-${Deno.build.arch}`;
}

/** Library from `$OFFGRID_LIB` or this repository's target/ dir, without any download. */
export function localLibPath(): string | undefined {
  const env = Deno.env.get("OFFGRID_LIB");
  if (env) return env;
  for (const profile of ["release", "debug"]) {
    const url = new URL(`../../target/${profile}/${LIB_FILE}`, import.meta.url);
    if (url.protocol !== "file:") break;
    try {
      Deno.statSync(url);
      return fromFileUrl(url);
    } catch { /* try next */ }
  }
  return undefined;
}

/** Per-user cache root: `$OFFGRID_CACHE`, else the platform's cache dir + `/offgrid`. */
export function cacheRoot(): string {
  const env = Deno.env.get("OFFGRID_CACHE");
  if (env) return env;
  const home = Deno.env.get("HOME") ?? Deno.env.get("USERPROFILE") ?? ".";
  switch (Deno.build.os) {
    case "windows":
      return join(Deno.env.get("LOCALAPPDATA") ?? join(home, "AppData", "Local"), "offgrid");
    case "darwin":
      return join(home, "Library", "Caches", "offgrid");
    default:
      return join(Deno.env.get("XDG_CACHE_HOME") ?? join(home, ".cache"), "offgrid");
  }
}

/** The release package's library if it is already in the cache. */
export function cachedLibPath(root: string = cacheRoot()): string | undefined {
  const asset = RELEASE.assets[platform()];
  if (!asset) return undefined;
  const lib = join(root, RELEASE.version, platform(), asset.lib);
  try {
    Deno.statSync(lib);
    return lib;
  } catch {
    return undefined;
  }
}

/** Download this version's release package for the current platform into the cache (once). */
export async function downloadLibrary(root: string = cacheRoot()): Promise<string> {
  const asset = RELEASE.assets[platform()];
  if (!asset) throw new Error(`offgrid: no prebuilt library for ${platform()}; set OFFGRID_LIB`);
  if (!asset.sha256) {
    throw new Error("offgrid: these bindings are not a published release; build the library or set OFFGRID_LIB");
  }
  return await installAsset(asset, `${RELEASE.baseUrl}/${asset.file}`, join(root, RELEASE.version, platform()));
}

/** @internal Fetch `url`, verify it against `asset.sha256` and extract it to `dir`; returns the library path. */
export async function installAsset(asset: ReleaseAsset, url: string, dir: string): Promise<string> {
  const lib = join(dir, asset.lib);
  try {
    Deno.statSync(lib);
    return lib;
  } catch { /* not installed yet */ }

  const res = await fetch(url);
  if (!res.ok) {
    await res.body?.cancel();
    throw new Error(`offgrid: download ${url} failed: HTTP ${res.status}`);
  }
  const data = new Uint8Array(await res.arrayBuffer());
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", data));
  const hex = Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
  if (hex !== asset.sha256) {
    throw new Error(`offgrid: checksum mismatch for ${asset.file} (expected ${asset.sha256}, got ${hex})`);
  }

  // Extract next to the final dir, then rename, so a partial extraction is never used.
  await Deno.mkdir(dirname(dir), { recursive: true });
  const tmp = await Deno.makeTempDir({ dir: dirname(dir), prefix: ".offgrid-" });
  try {
    const entries = ReadableStream.from([data])
      .pipeThrough(new DecompressionStream("gzip"))
      .pipeThrough(new UntarStream());
    for await (const entry of entries) {
      const path = normalize(entry.path);
      if (isAbsolute(path) || path.startsWith("..")) throw new Error(`offgrid: bad path in archive: ${entry.path}`);
      const dest = join(tmp, path);
      if (entry.header.typeflag === "5") {
        await Deno.mkdir(dest, { recursive: true });
        await entry.readable?.cancel();
      } else if (entry.readable) {
        await Deno.mkdir(dirname(dest), { recursive: true });
        await Deno.writeFile(dest, entry.readable, { mode: entry.header.mode & 0o777 || 0o644 });
      }
    }
    try {
      await Deno.rename(tmp, dir);
    } catch (e) {
      // Fine if another process installed it first.
      try {
        Deno.statSync(lib);
      } catch {
        throw e;
      }
    }
  } finally {
    await Deno.remove(tmp, { recursive: true }).catch(() => {});
  }
  Deno.statSync(lib);
  return lib;
}

function fromFileUrl(url: URL): string {
  const path = decodeURIComponent(url.pathname);
  return Deno.build.os === "windows" ? path.replace(/^\/([A-Za-z]:)/, "$1").replaceAll("/", "\\") : path;
}
