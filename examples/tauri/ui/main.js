// No bundler: the Tauri API comes from `window.__TAURI__` (`withGlobalTauri` in tauri.conf.json).
const { invoke, Channel } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const SYSTEM = { role: "system", content: "You are a helpful assistant." };

const statusEl = document.getElementById("status");
const messagesEl = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const newChatBtn = document.getElementById("new-chat");

// The whole conversation, in OpenAI format. offgrid is stateless per request, so every turn
// sends all of it; the session's KV cache makes that cheap.
let history = [SYSTEM];
let ready = false;
let busy = false;

function showStatus(s) {
  ready = s.state === "ready";
  statusEl.className = `status ${s.state}`;
  statusEl.textContent =
    s.state === "ready" ? `${s.model} · ${s.device} · ${s.n_ctx} ctx`
    : s.state === "failed" ? `Failed to load the model: ${s.error}`
    : s.path ? `Loading ${s.path}…` : "Starting…";
  updateControls();
}

function updateControls() {
  input.disabled = !ready;
  sendBtn.disabled = !ready;
  sendBtn.textContent = busy ? "Stop" : "Send";
  sendBtn.classList.toggle("stop", busy);
  newChatBtn.disabled = busy;
}

function addMessage(role, text = "") {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  const body = document.createElement("div");
  body.textContent = text;
  el.append(body);
  messagesEl.append(el);
  scrollDown();
  return { el, body };
}

function scrollDown() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function send(text) {
  history.push({ role: "user", content: text });
  addMessage("user", text);
  const { el, body } = addMessage("assistant");
  let reasoning = null;

  busy = true;
  updateControls();

  const onDelta = new Channel();
  onDelta.onmessage = (delta) => {
    if (delta.type === "reasoning") {
      if (!reasoning) {
        const details = document.createElement("details");
        details.innerHTML = "<summary>Thinking</summary>";
        reasoning = document.createElement("div");
        details.append(reasoning);
        el.prepend(details);
      }
      reasoning.textContent += delta.text;
    } else {
      body.textContent += delta.text;
    }
    scrollDown();
  };

  try {
    const res = await invoke("chat", { messages: history, onDelta });
    // The final response carries the parsed text, so it replaces what was streamed.
    body.textContent = res.content;
    history.push({ role: "assistant", content: res.content });

    const u = res.usage;
    const tps = u.completion_tokens / Math.max(u.generation_ms / 1000, 1e-9);
    const stats = document.createElement("div");
    stats.className = "stats";
    stats.textContent =
      `${u.completion_tokens} tokens · ${tps.toFixed(1)} tok/s · prompt ${u.prompt_tokens} (${u.cached_tokens} cached)` +
      (res.finish_reason === "stop" ? "" : ` · ${res.finish_reason}`);
    el.append(stats);
  } catch (err) {
    el.remove();
    history.pop();
    addMessage("error", String(err));
  } finally {
    busy = false;
    updateControls();
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (busy) {
    invoke("cancel");
    return;
  }
  const text = input.value.trim();
  if (!text || !ready) return;
  input.value = "";
  send(text);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!busy) form.requestSubmit();
  }
});

newChatBtn.addEventListener("click", () => {
  history = [SYSTEM];
  messagesEl.replaceChildren();
  input.focus();
});

// Listen first, then fetch the current state, so a change in between isn't missed. An event
// that arrives before the reply is newer than it, so the reply is dropped then.
let gotEvent = false;
listen("status", (e) => {
  gotEvent = true;
  showStatus(e.payload);
}).then(() => invoke("status")).then((s) => gotEvent || showStatus(s));
