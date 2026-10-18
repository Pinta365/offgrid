#!/usr/bin/env node
// stamp-windows-version.mjs <resedit-modules-dir> <bin-dir> <version>
//
// Writes a version resource (product name, version, description) into every .exe and .dll in
// <bin-dir>. Code signing requires consistent product metadata across the signed files, and
// Windows shows it in a file's properties. Uses resedit (installed by package-windows.sh into
// <resedit-modules-dir>/node_modules), so it runs on Linux next to the cross-compiled build.

import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { basename, join } from "node:path";

const [modules, bin, version] = process.argv.slice(2);
if (!modules || !bin || !/^\d+\.\d+\.\d+$/.test(version ?? "")) {
  console.error("usage: stamp-windows-version.mjs <resedit-modules-dir> <bin-dir> <x.y.z>");
  process.exit(2);
}
const require = createRequire(join(modules, "node_modules", "_.js"));
const PE = require("pe-library");
const ResEdit = require("resedit");

const [major, minor, patch] = version.split(".").map(Number);
const LANG = 1033; // en-US
const CODEPAGE = 1200; // UTF-16

/** What each file is: offgrid's own, or a library it builds from source. */
function describe(file) {
  const name = basename(file).toLowerCase();
  const cpu = /^ggml-cpu-(.+)\.dll$/.exec(name);
  if (name === "offgrid-server.exe") return "offgrid OpenAI-compatible server";
  if (name === "offgrid.dll") return "offgrid C library";
  if (cpu) return `ggml CPU backend (${cpu[1]}), built for offgrid`;
  if (name === "ggml-vulkan.dll") return "ggml Vulkan backend, built for offgrid";
  if (name.startsWith("ggml")) return "ggml, built for offgrid";
  if (name === "libllama.dll") return "llama.cpp, built for offgrid";
  if (name === "libmtmd.dll") return "llama.cpp multimodal (mtmd), built for offgrid";
  if (name === "libstable-diffusion.dll") return "stable-diffusion.cpp, built for offgrid";
  throw new Error(`no description for ${name}: add it to stamp-windows-version.mjs`);
}

const files = readdirSync(bin).filter((f) => /\.(exe|dll)$/i.test(f));
for (const f of files) {
  const path = join(bin, f);
  const exe = PE.NtExecutable.from(readFileSync(path));
  const res = PE.NtExecutableResource.from(exe);
  const vi = ResEdit.Resource.VersionInfo.fromEntries(res.entries)[0] ?? ResEdit.Resource.VersionInfo.createEmpty();
  vi.lang = LANG;
  vi.setFileVersion(major, minor, patch, 0, LANG);
  vi.setProductVersion(major, minor, patch, 0, LANG);
  vi.setStringValues(
    { lang: LANG, codepage: CODEPAGE },
    {
      ProductName: "offgrid",
      ProductVersion: version,
      FileVersion: version,
      FileDescription: describe(f),
      OriginalFilename: f,
      InternalName: f.replace(/\.(exe|dll)$/i, ""),
      CompanyName: "offgrid (github.com/Pinta365/offgrid)",
      LegalCopyright: "MIT License. Includes llama.cpp, ggml and stable-diffusion.cpp (MIT).",
    },
  );
  vi.outputToResourceEntries(res.entries);
  res.outputResource(exe);
  writeFileSync(path, Buffer.from(exe.generate()));
  console.log(`stamped ${f}: offgrid ${version} (${describe(f)})`);
}
if (files.length === 0) {
  console.error(`no .exe/.dll in ${bin}`);
  process.exit(1);
}
