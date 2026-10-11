// Download/verify/extract of the release package, against a local server.
import { assertEquals, assertRejects } from "@std/assert";
import { join } from "@std/path";
import { TarStream, type TarStreamInput } from "@std/tar/tar-stream";
import { installAsset } from "./native.ts";

const LIB = "pkg/lib/liboffgrid.so";
const CONTENT = new TextEncoder().encode("not really a library");

async function tarGz(): Promise<Uint8Array<ArrayBuffer>> {
  const entries: TarStreamInput[] = [
    { type: "directory", path: "pkg" },
    { type: "directory", path: "pkg/lib" },
    { type: "file", path: LIB, size: CONTENT.length, readable: ReadableStream.from([CONTENT]) },
  ];
  const gz = ReadableStream.from(entries).pipeThrough(new TarStream()).pipeThrough(new CompressionStream("gzip"));
  return new Uint8Array(await new Response(gz).arrayBuffer());
}

async function sha256(data: Uint8Array<ArrayBuffer>): Promise<string> {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", data));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

Deno.test("download: verifies, extracts, caches", async () => {
  const archive = await tarGz();
  let hits = 0;
  await using server = Deno.serve({ hostname: "127.0.0.1", port: 0, onListen() {} }, (req) => {
    hits++;
    return new URL(req.url).pathname === "/pkg.tar.gz" ? new Response(archive) : new Response("no", { status: 404 });
  });
  const base = `http://127.0.0.1:${server.addr.port}`;
  const root = await Deno.makeTempDir();
  try {
    const asset = { file: "pkg.tar.gz", sha256: await sha256(archive), lib: LIB };
    const lib = await installAsset(asset, `${base}/pkg.tar.gz`, join(root, "good"));
    assertEquals(await Deno.readFile(lib), CONTENT);
    assertEquals(await installAsset(asset, `${base}/pkg.tar.gz`, join(root, "good")), lib);
    assertEquals(hits, 1, "second call must use the cache");

    const bad = { ...asset, sha256: "0".repeat(64) };
    await assertRejects(() => installAsset(bad, `${base}/pkg.tar.gz`, join(root, "bad")), Error, "checksum mismatch");
    await assertRejects(() => Deno.stat(join(root, "bad")), Deno.errors.NotFound);
    assertEquals([...Deno.readDirSync(root)].map((e) => e.name), ["good"], "no temp dirs left behind");

    await assertRejects(() => installAsset(asset, `${base}/missing.tar.gz`, join(root, "404")), Error, "HTTP 404");
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});
