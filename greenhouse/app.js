"use strict";
/* AI-Assisted Modern Greenhouse - static dashboard (GitHub Pages).
   Data: data/bundle.bin, written by ghnl.webexport and ENCRYPTED with the dashboard password
   (AES-256-GCM, key from PBKDF2-SHA256). It is decrypted here in the browser, so the public files
   reveal nothing without the password. */

const DT = 900, PER_DAY = 96;
const S = { charts: {}, chat: [], models: null };
const $ = (id) => document.getElementById(id);
const pad = (n) => String(n).padStart(2, "0");
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const COL = { blue: "#0072B2", grey: "#999999", green: "#009E73", red: "#D55E00", sky: "#56B4E9", gold: "#E69F00", pink: "#CC79A7" };

// ---------------- time helpers (all times are local standard time, stored as UTC to avoid DST shifts)
const tAt = (i) => new Date(S.t0ms + i * DT * 1000);
const ymd = (d) => `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
const hm = (d) => `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
const dm = (d) => `${pad(d.getUTCDate())} ${MON[d.getUTCMonth()]}`;
const dayIndex = (s) => Math.round((Date.parse(`${s}T00:00:00Z`) - S.t0ms) / (DT * 1000));
const factMs = (s) => Date.parse(s.replace(" ", "T") + ":00Z");
const sum = (a) => a.reduce((x, y) => x + y, 0);
const r0 = (x) => Math.round(x), f1 = (x) => x.toFixed(1), f2 = (x) => x.toFixed(2);
const span = (a, b) => (Math.round(a) < 0 || Math.round(b) < 0 ? `${r0(a)} to ${r0(b)}` : `${r0(a)}–${r0(b)}`);

// ---------------- password lock
const saved = {
  get() { try { return sessionStorage.getItem("ghPw") || localStorage.getItem("ghPw") || ""; } catch { return ""; } },
  kept() { try { return !!localStorage.getItem("ghPw"); } catch { return false; } },
  set(pw, keep) { try { sessionStorage.setItem("ghPw", pw); if (keep) localStorage.setItem("ghPw", pw); else localStorage.removeItem("ghPw"); } catch { /* storage blocked */ } },
  clear() { try { sessionStorage.removeItem("ghPw"); localStorage.removeItem("ghPw"); } catch { /* storage blocked */ } },
};
let bundle = null;

async function decryptData(pw) {
  if (!bundle) {
    const r = await fetch("data/bundle.bin", { cache: "no-cache" });
    if (!r.ok) throw new Error(`data file missing (${r.status})`);
    bundle = new Uint8Array(await r.arrayBuffer());
  }
  if (String.fromCharCode(...bundle.subarray(0, 4)) !== "GHE1") throw new Error("unknown data file");
  const rounds = new DataView(bundle.buffer, bundle.byteOffset).getUint32(4);
  const salt = bundle.slice(8, 24), iv = bundle.slice(24, 36), ct = bundle.slice(36);
  const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(pw), "PBKDF2", false, ["deriveKey"]);
  const key = await crypto.subtle.deriveKey({ name: "PBKDF2", hash: "SHA-256", salt, iterations: rounds },
    base, { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
  let plain;
  try { plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ct); }
  catch { const e = new Error("wrong password"); e.wrong = true; throw e; }
  const text = await new Response(new Blob([plain]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
  return JSON.parse(text);
}

async function tryUnlock(pw, keep) {
  const msg = $("lock-msg"), btn = $("unlock");
  if (!(window.crypto && crypto.subtle && window.DecompressionStream)) {
    msg.textContent = "This browser is too old to open the dashboard. Please update it (or use Chrome, Edge, Safari or Firefox).";
    return;
  }
  btn.disabled = true; msg.textContent = "Opening…";
  try {
    const data = await decryptData(pw);
    saved.set(pw, keep);
    $("lock").hidden = true; $("app").hidden = false; $("lock-btn").hidden = false;
    start(data);
  } catch (e) {
    if (e.wrong) { saved.clear(); msg.textContent = "Wrong password. Please try again."; $("pw").select(); }
    else msg.textContent = `Could not open the data (${e.message}). Check the internet connection and try again.`;
  } finally { btn.disabled = false; }
}

function boot() {
  $("lock-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const pw = $("pw").value.trim();
    if (pw) tryUnlock(pw, $("pw-remember").checked); else $("lock-msg").textContent = "Type the password first.";
  });
  $("lock-btn").addEventListener("click", () => { saved.clear(); location.reload(); });
  const pw = saved.get();
  if (pw) { $("pw-remember").checked = saved.kept(); tryUnlock(pw, saved.kept()); }
}

// ---------------- start (after the data is unlocked)
function start({ series, facts, meta }) {
  S.d = series; S.n = series.n; S.t0ms = Date.parse(series.t0); S.meta = meta;
  S.facts = facts.map((f) => ({ ...f, t0: factMs(f.s), t1: factMs(f.e) }));
  S.first = ymd(tAt(0)); S.last = ymd(tAt(S.n - 1)); S.totalDays = Math.round(S.n / PER_DAY);
  $("note").textContent = `Lettuce greenhouse simulation · ${meta.weather_note} · the controller decides heating, vents and CO₂ every 15 minutes.`;
  const sd = $("start"); sd.min = S.first; sd.max = S.last; sd.value = S.first;
  const nd = $("ndays"); nd.max = S.totalDays;
  sd.addEventListener("change", update); nd.addEventListener("input", update);
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("topic").addEventListener("change", renderFacts);
  S.ai = GreenhouseAI.setup({ answer: quickAnswer, context: quickContext,
    dataNote: "The data comes from a greenhouse simulation." });
  update();
}

// ---------------- period
function update() {
  const startStr = $("start").value || S.first;
  let i0 = Math.max(0, Math.min(dayIndex(startStr), S.n - PER_DAY));
  const left = Math.round((S.n - i0) / PER_DAY);
  const nd = $("ndays"); nd.max = Math.max(left, 1);
  const n = Math.min(+nd.value, left);
  S.i0 = i0; S.i1 = Math.min(S.n, i0 + n * PER_DAY); S.nd = n;
  S.startStr = ymd(tAt(S.i0)); S.endStr = ymd(tAt(S.i1 - 1));
  S.label = n === 1 ? S.startStr : `${S.startStr} to ${S.endStr}`;
  $("nlabel").textContent = n;
  $("showing").innerHTML = `Showing <b>${S.label}</b> · ${n} day${n > 1 ? "s" : ""}` +
    (n >= 45 ? ` (about ${(n / 30.4).toFixed(1)} months)` : "") + ` · data from ${S.first} to ${S.last}`;
  renderSummary();
  if (S.tab === "charts") renderCharts();
  if (S.tab === "facts") renderFacts();
  renderExamples();
}

function showTab(t) {
  S.tab = t;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === t)));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${t}`));
  if (t === "charts") renderCharts();
  if (t === "facts") renderFacts();
}

const range = (a, b) => { const out = []; for (let i = a; i < b; i++) out.push(i); return out; };

// ---------------- summary
function metric(label, value, delta) {
  return `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div>` +
    (delta ? `<div class="delta">${delta}</div>` : "") + `</div>`;
}

function minmax(key, idx) {
  let lo = Infinity, hi = -Infinity;
  for (const i of idx) { const v = S.d[key][i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

function renderSummary() {
  const d = S.d, m = S.meta, idx = range(S.i0, S.i1), last = S.i1 - 1;
  const tot = (k) => { let s = 0; for (const i of idx) s += d[k][i]; return s; };
  const kwh = tot("H") * DT / 3.6e6, co2g = tot("C") * DT / 1000;
  const ventH = idx.filter((i) => d.V[i] > 0.1).length * 0.25;
  const crop = d.K[last]; let cs = last; while (cs > 0 && d.K[cs - 1] === crop) cs--;
  const age = Math.floor((last - cs) / PER_DAY) + 1;
  const crops = new Set(idx.map((i) => d.K[i])).size;
  const row = (name, indoor, outdoor) => `<tr><th scope="row">${name}</th><td>${indoor}</td><td>${outdoor}</td></tr>`;
  $("sum-title").textContent = `Summary · ${S.label}`;
  $("compare").innerHTML = `<table class="cmp"><thead><tr><th></th><th scope="col">Indoor</th><th scope="col">Outdoor</th></tr></thead><tbody>` +
    row("Temperature", `${span(...minmax("T", idx))} °C`, `${span(...minmax("To", idx))} °C`) +
    row("Humidity", `${span(...minmax("RH", idx))} %`, `${span(...minmax("RHo", idx))} %`) +
    row("CO₂", `${span(...minmax("CO2", idx))} ppm`, `${span(...minmax("CO2o", idx))} ppm`) +
    row("Sunlight", `<span class="muted">uses outdoor</span>`, `${f1(tot("R") * 0.25 / 1000)} kWh/m²`) + `</tbody></table>`;
  $("compare-note").textContent = "Lowest–highest for the period. " + m.sources;
  $("metrics").innerHTML =
    metric("Heating used", `${f2(kwh)} kWh/m²`, `${f2(kwh * m.prices.heat_cad_per_kwh)} CAD/m²`) +
    metric("CO₂ added", `${r0(co2g)} g/m²`, `${(co2g / 1000 * m.prices.co2_cad_per_kg).toFixed(3)} CAD/m²`) +
    metric("Vents open", `${f1(ventH)} h`, `of ${f1(idx.length * 0.25)} h`) +
    metric(`Lettuce now · crop ${crop}, day ${age}`, `${r0(d.DW[last])} g/m²`,
      `grew ${r0(tot("G"))} g/m² in this period${crops > 1 ? ` (${crops} crops)` : ""}`);
  const a = tAt(S.i0).getTime(), b = tAt(S.i1 - 1).getTime();
  const daily = S.facts.filter((f) => f.k === "daily" && f.t0 >= a - 1 && f.t0 <= b).sort((x, y) => x.t0 - y.t0);
  $("daily").innerHTML = daily.slice(0, 7).map((f) => `<div class="fact">${esc(f.x)}</div>`).join("") +
    (daily.length > 7 ? `<p class="muted">Showing the first 7 of ${daily.length} days; see Facts for more.</p>` : "");
  $("season").innerHTML = S.facts.filter((f) => f.k === "overall" || f.k === "crop")
    .map((f) => `<div class="fact">${esc(f.x)}</div>`).join("");
}

// ---------------- charts
function bucket() { const n = S.nd; return n <= 1 ? 1 : n <= 3 ? 2 : n <= 14 ? 4 : n <= 60 ? 24 : 96; }

function series(key) {
  const k = bucket(), out = [];
  for (let i = S.i0; i < S.i1; i += k) {
    const j = Math.min(i + k, S.i1); let s = 0;
    for (let x = i; x < j; x++) s += S.d[key][x];
    out.push(s / (j - i));
  }
  return out;
}

function labels() {
  const k = bucket(), out = [];
  for (let i = S.i0; i < S.i1; i += k) {
    const t = tAt(i);
    out.push(S.nd <= 1 ? hm(t) : k >= 24 ? dm(t) : `${dm(t)} ${hm(t)}`);
  }
  return out;
}

function chart(id, datasets, opts = {}) {
  const css = getComputedStyle(document.documentElement);
  const ink = css.getPropertyValue("--muted").trim() || "#5b6670", grid = css.getPropertyValue("--line").trim() || "#dde3e8";
  const cfg = {
    type: "line", data: { labels: labels(), datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: datasets.filter((d) => d.label && !d.hideLegend).length > 1,
        labels: { color: ink, boxWidth: 12, filter: (it, data) => !data.datasets[it.datasetIndex].hideLegend } },
        tooltip: { callbacks: { label: (c) => c.dataset.hideLegend ? null : `${c.dataset.label}: ${c.parsed.y.toFixed(1)}` } } },
      elements: { point: { radius: 0 }, line: { borderWidth: 2.2, tension: 0.2 } },
      scales: { x: { ticks: { color: ink, maxTicksLimit: 6, maxRotation: 0 }, grid: { color: grid } },
        y: { ticks: { color: ink }, grid: { color: grid }, ...opts.y } },
    },
  };
  if (S.charts[id]) S.charts[id].destroy();
  S.charts[id] = new Chart($(id), cfg);
}

function renderCharts() {
  if (typeof Chart === "undefined") { $("chart-note").textContent = "Charts need an internet connection to load."; return; }
  const k = bucket();
  $("chart-note").textContent = `Grey line = outdoor, coloured line = indoor. Tap or hover on a chart to see the values · ${k === 1 ? "every 15 min" : k === 2 ? "30-min averages" : k === 4 ? "hourly averages" : k === 24 ? "6-hour averages" : "daily averages"}.`;
  const flat = (v) => labels().map(() => v);
  chart("c-temp", [
    { label: "min", data: series("Tmin"), borderColor: "transparent", pointRadius: 0, hideLegend: true },
    { label: "max", data: series("Tmax"), borderColor: "transparent", backgroundColor: "rgba(0,114,178,.13)", fill: "-1", hideLegend: true },
    { label: "Indoor", data: series("T"), borderColor: COL.blue },
    { label: "Outdoor", data: series("To"), borderColor: COL.grey },
  ]);
  chart("c-rh", [{ label: "Indoor", data: series("RH"), borderColor: COL.green },
    { label: "Outdoor", data: series("RHo"), borderColor: COL.grey },
    { label: "max", data: flat(S.meta.limits.rh_max), borderColor: COL.red, borderDash: [5, 4], borderWidth: 1.5, hideLegend: true }]);
  chart("c-co2", [{ label: "Indoor", data: series("CO2"), borderColor: COL.pink },
    { label: "Outdoor", data: series("CO2o"), borderColor: COL.grey }]);
  chart("c-co2o", [{ label: "Outdoor", data: series("CO2o"), borderColor: "#6f6f6f" }]);
  chart("c-sun", [{ label: "Outdoor", data: series("R"), borderColor: COL.gold, backgroundColor: "rgba(230,159,0,.3)", fill: true }]);
  chart("c-heat", [{ label: "Heating", data: series("H"), borderColor: COL.red, backgroundColor: "rgba(213,94,0,.25)", fill: true }]);
  chart("c-vent", [{ label: "Ventilation", data: series("V"), borderColor: COL.sky, backgroundColor: "rgba(86,180,233,.3)", fill: true }]);
}

// ---------------- facts
function overlapping(a, b) { return S.facts.filter((f) => f.k === "segment" && f.t1 > a && f.t0 <= b); }

function renderFacts() {
  const topic = $("topic").value, a = tAt(S.i0).getTime(), b = tAt(S.i1 - 1).getTime();
  const list = overlapping(a, b).filter((f) => topic === "all" || f.v === topic).sort((x, y) => x.t0 - y.t0);
  $("facts-note").textContent = `${list.length} facts for ${S.label}.` + (list.length > 150 ? " Showing the first 150; choose fewer days or one topic." : "");
  $("facts").innerHTML = list.slice(0, 150).map((f) => `<div class="fact"><b>${f.v}</b> · ${esc(f.x)}</div>`).join("");
}

// ---------------- instant answers (no AI)
function windowFor(q) {
  const ql = q.toLowerCase() + " ";
  const m = q.match(/(\d{4}-\d{2}-\d{2})/);
  let a, b, label;
  if (m) { a = dayIndex(m[1]); b = a + PER_DAY; label = m[1]; } else { a = S.i0; b = S.i1; label = S.label; }
  let idx = range(Math.max(0, a), Math.min(S.n, b));
  if (ql.includes("night")) { idx = idx.filter((i) => !S.d.D[i]); label = `the night of ${label}`; }
  else if (/(day |daytime|noon|afternoon)/.test(ql)) { idx = idx.filter((i) => S.d.D[i]); label = `daytime on ${label}`; }
  return { idx, label };
}

function quickAnswer(q) {
  const d = S.d, L = S.meta.limits, ql = q.toLowerCase() + " ";
  const { idx, label: when } = windowFor(q);
  if (!idx.length) return `There is no simulated data for that date; pick a date between ${S.first} and ${S.last}.`;
  const T = idx.map((i) => d.T[i]), lo = Math.min(...T), hi = Math.max(...T);
  const oLo = Math.min(...idx.map((i) => d.To[i])), oHi = Math.max(...idx.map((i) => d.To[i]));
  const above = idx.filter((i) => d.T[i] > d.Tmax[i] + 0.3), below = idx.filter((i) => d.T[i] < d.Tmin[i] - 0.3);
  const ventH = idx.filter((i) => d.V[i] > 0.5).length * 0.25;
  const rhH = idx.filter((i) => d.RH[i] >= L.rh_max - 0.5).length * 0.25;
  const kwh = sum(idx.map((i) => d.H[i])) * DT / 3.6e6, cad = kwh * S.meta.prices.heat_cad_per_kwh;
  const multi = idx[idx.length - 1] - idx[0] > PER_DAY;
  const tf = (i) => (multi ? `${dm(tAt(i))} ${hm(tAt(i))}` : hm(tAt(i)));
  const iHi = idx[T.indexOf(hi)];
  const has = (...w) => w.some((x) => ql.includes(x));

  if (has("vent", "window", "open", "cool")) {
    if (above.length) return `Yes — on ${when} the greenhouse got too warm (up to ${f1(hi)} °C around ${tf(iHi)}), so the controller opened the vents for ${f1(ventH)} h to cool it.`;
    return `No need — on ${when} the greenhouse never got too warm (indoor highest ${f1(hi)} °C while it was ${r0(oLo)} to ${r0(oHi)} °C outdoors); the vents opened only slightly, to keep humidity at the ${r0(L.rh_max)} % limit.`;
  }
  if (has("heat")) {
    const a = tAt(idx[0]).getTime(), b = tAt(idx[idx.length - 1]).getTime();
    const ups = overlapping(a, b).filter((f) => f.v === "heating" && f.tr === "rose");
    if (has("why", "up", "rise", "increase", "high") && ups.length) {
      const f = ups.reduce((x, y) => (y.ch > x.ch ? y : x));
      const tmin = Math.min(...idx.map((i) => d.Tmin[i]));
      return `Heating rose from ${r0(f.a)} to ${r0(f.b)} W/m² between ${f.s.slice(11)} and ${f.e.slice(11)}${f.df < 0.5 ? " overnight" : ""} because it was ${r0(f.oe)} °C outdoors and the controller had to keep the greenhouse at its ${r0(tmin)} °C minimum.`;
    }
    return `On ${when} heating used ${f2(kwh)} kWh/m² (${f2(cad)} CAD/m²), averaging ${r0(sum(idx.map((i) => d.H[i])) / idx.length)} W/m² while it was ${r0(oLo)} to ${r0(oHi)} °C outdoors.`;
  }
  if (has("humid", "moist")) return `On ${when} indoor humidity reached ${r0(Math.max(...idx.map((i) => d.RH[i])))} % (outdoor ${span(...minmax("RHo", idx))} %); it sat at the ${r0(L.rh_max)} % limit for ${f1(rhH)} h and the controller held it there with small vent openings.`;
  if (has("co2", "co₂", "carbon")) {
    const c = idx.map((i) => d.CO2[i]);
    return `On ${when} indoor CO₂ ranged ${r0(Math.min(...c))}–${r0(Math.max(...c))} ppm (outdoor ${span(...minmax("CO2o", idx))} ppm): the controller added ${r0(sum(idx.map((i) => d.C[i])) * DT / 1000)} g/m² in daylight, when it helps the lettuce grow.`;
  }
  if (has("grow", "harvest", "yield", "weight", "lettuce")) {
    const last = idx[idx.length - 1];
    return `On ${when} the lettuce (crop ${d.K[last]}) grew ${f1(sum(idx.map((i) => d.G[i])))} g dry weight/m², reaching ${r0(d.DW[last])} g/m².`;
  }
  if (has("cost", "money", "price", "energy", "gas")) return `On ${when} heating cost ${f2(cad)} CAD/m² (${f2(kwh)} kWh/m²) and CO₂ ${(sum(idx.map((i) => d.C[i])) * DT * 1e-6 * S.meta.prices.co2_cad_per_kg).toFixed(3)} CAD/m².`;
  if (!above.length && !below.length)
    return `Yes — on ${when} the greenhouse stayed within its limits at ${r0(lo)}–${r0(hi)} °C indoors (allowed ${L.t_night[0]}–${L.t_night[1]} °C at night, ${L.t_day[0]}–${L.t_day[1]} °C by day) while it was ${r0(oLo)} to ${r0(oHi)} °C outdoors.`;
  const parts = [];
  if (above.length) {
    const ex = above.map((i) => d.T[i] - d.Tmax[i]), w = above[ex.indexOf(Math.max(...ex))];
    const warmOutside = above.filter((i) => d.To[i] > d.Tmax[i]).length > above.length / 2;
    parts.push(`too warm for ${f1(above.length * 0.25)} h (worst ${f1(Math.max(...ex))} °C over the limit around ${tf(w)})` +
      (warmOutside ? ", mostly because it was warmer outdoors than the limit, which vents cannot fix" : ", even with the vents open"));
  }
  if (below.length) {
    const de = below.map((i) => d.Tmin[i] - d.T[i]), w = below[de.indexOf(Math.max(...de))];
    const sunrise = below.filter((i) => d.D[i]).length > below.length / 2 && tAt(w).getUTCHours() < 11;
    parts.push(`too cool for ${f1(below.length * 0.25)} h (worst ${f1(Math.max(...de))} °C under the limit around ${tf(w)}` +
      (sunrise ? " at sunrise, when the higher daytime minimum starts and heating is still catching up" : "") + ")");
  }
  return `Mostly — on ${when} the greenhouse was ${r0(lo)}–${r0(hi)} °C but ${parts.join(". It was also ")}.`;
}

function quickContext(q) {
  const { idx } = windowFor(q);
  const lines = [`Quick answer from the data: ${quickAnswer(q)}`];
  if (idx.length) {
    const a = tAt(idx[0]).getTime(), b = tAt(idx[idx.length - 1]).getTime(), ql = q.toLowerCase();
    const topics = { heating: ["heat"], temperature: ["temperat", "warm", "cold", "hot"], humidity: ["humid"],
      ventilation: ["vent", "window", "open"], co2: ["co2", "co₂"] };
    const named = Object.keys(topics).filter((k) => topics[k].some((w) => ql.includes(w)));
    overlapping(a, b).filter((f) => !named.length || named.includes(f.v) || (named.includes("co2") && f.v === "co2_supply"))
      .sort((x, y) => Math.abs(y.ch) - Math.abs(x.ch)).slice(0, 4).forEach((f) => lines.push(`- ${f.x}`));
  }
  return lines.join("\n");
}

// ---------------- ask tab (shared code in ai.js: instant answer + optional Gemini wording)
function renderExamples() {
  S.ai.setExamples([`Why did heating go up on the night of ${S.startStr}?`, `Is the temperature okay on ${S.startStr}?`,
    `Should I open the vents on ${S.startStr}?`]);
}

function esc(s) { return String(s).replace(/–-/g, " to -").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

boot();
