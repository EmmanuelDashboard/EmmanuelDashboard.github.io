"use strict";
/* Shared "Ask" tab for both dashboards (season simulation and live monitor).
   1) An instant answer is computed from the data by the page (no AI, under a second).
   2) If a free Google Gemini key is saved on this device, Gemini rewords that answer in natural
      language, using only the facts the page gives it (2-second limit; otherwise the instant answer
      stays). The key is stored only in this browser and sent only to Google.
   Expected element ids: key, remember, ai-status, check, forget, check-msg, examples, chat,
   ask-form, question, clear. */
window.GreenhouseAI = (() => {
  const SKIP = ["image", "tts", "audio", "live", "embedding", "robotics", "computer", "native", "learnlm", "gemma"];
  const $ = (id) => document.getElementById(id);
  let models = null;

  const system = (dataNote) => "You answer greenhouse growers in one or two short sentences (at most 40 words). " +
    "Use only the facts given in the message; keep every number and unit exactly as given; never add numbers. " +
    "Use plain words, no jargon. If the facts do not answer the question, say briefly what the data shows instead. " +
    "Say 'indoor' and 'outdoor' (not inside/outside). Answer in the language of the question. " + dataNote;

  function rankModels(names) {
    return names.map((n) => n.split("/").pop()).filter((n) => n.startsWith("gemini") && !SKIP.some((s) => n.includes(s)))
      .map((n) => {
        const v = parseFloat((n.match(/gemini-(\d+(?:\.\d+)?)/) || [0, 0])[1]);
        const tier = n.includes("flash") && !n.includes("lite") ? 0 : n.includes("flash") ? 1 : 2;
        return { n, key: [tier, /preview|exp/.test(n) ? 1 : 0, -v, n.length] };
      })
      .sort((a, b) => { for (let i = 0; i < 4; i++) if (a.key[i] !== b.key[i]) return a.key[i] - b.key[i]; return 0; })
      .map((x) => x.n);
  }

  async function gemini(path, body, key, ms) {
    const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), ms);
    try {
      const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/${path}`, {
        method: body ? "POST" : "GET", headers: { "Content-Type": "application/json", "x-goog-api-key": key },
        body: body ? JSON.stringify(body) : undefined, signal: ctl.signal });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) { const e = new Error((data.error && data.error.message) || r.statusText); e.code = r.status; throw e; }
      return data;
    } finally { clearTimeout(timer); }
  }

  async function loadModels(key) {
    const d = await gemini("models?pageSize=200", null, key, 10000);
    models = rankModels((d.models || []).filter((m) => !m.supportedGenerationMethods ||
      m.supportedGenerationMethods.includes("generateContent")).map((m) => m.name));
    return models;
  }

  async function askFast(context, q, history, key, dataNote, budgetMs = 2000) {
    const t0 = performance.now();
    if (!models) await loadModels(key);
    const lite = models.filter((n) => n.includes("flash-lite")).slice(0, 1);
    const flash = models.filter((n) => n.includes("flash") && !n.includes("lite")).slice(0, 2);
    const cands = [...lite, ...flash].length ? [...lite, ...flash] : models.slice(0, 3);
    const hist = history.slice(-4).map((m) => ({ role: m.role === "user" ? "user" : "model", parts: [{ text: m.text }] }));
    const contents = [...hist, { role: "user", parts: [{ text: `${context}\n\nGrower's question: ${q}` }] }];
    let last;
    for (const name of cands) {
      for (const thinking of [true, false]) {
        const left = budgetMs - (performance.now() - t0);
        if (left < 400) throw last || new Error("too slow");
        const body = { system_instruction: { parts: [{ text: system(dataNote) }] }, contents,
          generationConfig: { maxOutputTokens: 200, ...(thinking ? { thinkingConfig: { thinkingLevel: "MINIMAL" } } : {}) } };
        try {
          const d = await gemini(`models/${name}:generateContent`, body, key, left);
          const text = ((d.candidates || [])[0]?.content?.parts || []).map((p) => p.text || "").join("").trim();
          if (text) return { text, model: name };
          break;
        } catch (e) {
          last = e;
          if (thinking && e.code === 400 && /think/i.test(e.message)) continue;
          break;
        }
      }
    }
    throw last || new Error("no answer");
  }

  function storedKey() {
    try { return localStorage.getItem("ghKey") || sessionStorage.getItem("ghKey") || ""; } catch { return ""; }
  }

  /* answer(q) -> instant answer text; context(q) -> facts for the AI; dataNote -> one sentence about the data */
  function setup({ answer, context, dataNote }) {
    const chat = [];
    const key = storedKey();
    if (key) { $("key").value = key; try { $("remember").checked = !!localStorage.getItem("ghKey"); } catch { /* blocked */ } }
    const save = () => {
      const k = $("key").value.trim();
      try {
        localStorage.removeItem("ghKey"); sessionStorage.removeItem("ghKey");
        if (k) ($("remember").checked ? localStorage : sessionStorage).setItem("ghKey", k);
      } catch { /* storage blocked: the key lives only in the box */ }
      models = null;
      $("ai-status").textContent = k ? "· key entered" : "· not connected";
    };
    $("key").addEventListener("change", save); $("remember").addEventListener("change", save);
    $("ai-status").textContent = key ? "· key entered" : "· not connected";
    $("check").addEventListener("click", async () => {
      save(); const k = $("key").value.trim();
      if (!k) { $("check-msg").textContent = "Paste a key first."; return; }
      $("check-msg").textContent = "Checking…";
      try {
        const m = await loadModels(k);
        $("check-msg").textContent = m.length ? `Connected to Gemini. It will use ${m.filter((n) => n.includes("flash"))[0] || m[0]}.` : "The key works, but no text model is available.";
      } catch (e) {
        $("check-msg").textContent = /API key not valid|API_KEY_INVALID|permission/i.test(e.message) ? "The key was not accepted. Copy it again from AI Studio." : `Could not connect: ${e.message}`;
      }
    });
    $("forget").addEventListener("click", () => { $("key").value = ""; save(); $("check-msg").textContent = "Key removed from this device."; });

    function bubble(role, text, foot) {
      const div = document.createElement("div"); div.className = `msg ${role === "user" ? "user" : "bot"}`;
      div.textContent = text;
      if (foot) { const s = document.createElement("span"); s.className = "foot"; s.textContent = foot; div.appendChild(s); }
      $("chat").appendChild(div); div.scrollIntoView({ block: "nearest" }); return div;
    }

    async function ask(q) {
      bubble("user", q); $("clear").hidden = false;
      const quick = answer(q), k = $("key").value.trim() || storedKey();
      if (!k) {
        bubble("bot", quick, "Instant answer from the data. Add a free Gemini key in Connect AI for answers in natural words.");
        chat.push({ role: "user", text: q }, { role: "bot", text: quick }); return;
      }
      const b = bubble("bot", quick, "⏳ Improving the wording with AI…");
      try {
        const r = await askFast(context(q), q, chat, k, dataNote, 2000);
        b.textContent = r.text;
        const s = document.createElement("span"); s.className = "foot"; s.textContent = `AI: Gemini · ${r.model}`; b.appendChild(s);
        chat.push({ role: "user", text: q }, { role: "bot", text: r.text });
      } catch {
        b.querySelector(".foot").textContent = "Instant answer from the data (the AI did not reply within 2 seconds).";
        chat.push({ role: "user", text: q }, { role: "bot", text: quick });
      }
    }

    $("ask-form").addEventListener("submit", (ev) => { ev.preventDefault(); const q = $("question").value.trim(); if (q) { $("question").value = ""; ask(q); } });
    $("clear").addEventListener("click", () => { chat.length = 0; $("chat").innerHTML = ""; $("clear").hidden = true; });

    function setExamples(list) {
      $("examples").innerHTML = "";
      list.forEach((q) => {
        const b = document.createElement("button"); b.className = "btn example"; b.type = "button"; b.textContent = q;
        b.onclick = () => ask(q); $("examples").appendChild(b);
      });
    }
    return { ask, setExamples };
  }

  return { setup };
})();
