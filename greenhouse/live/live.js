"use strict";
/* Live greenhouse monitor. Signs in to the live database (Supabase Auth, e-mail + password) and shows
   the newest readings, sensor checks and charts. Plain fetch calls: no library besides Chart.js.
   Only e-mail addresses on the database's viewers list can read the data (row-level security). */

const CFG = window.LIVE_CONFIG || {};
const LIM = { rh_max: 80, co2_max: 1600, ...(CFG.limits || {}) };
const RANGES = { "6h": { h: 6, b: 5 }, "24h": { h: 24, b: 15 }, "3d": { h: 72, b: 30 }, "7d": { h: 168, b: 60 } };
const COL = { blue: "#0072B2", grey: "#999999", dark: "#6f6f6f", green: "#009E73", red: "#D55E00",
  sky: "#56B4E9", gold: "#E69F00", pink: "#CC79A7" };
const S = { range: "24h", rows: [], latest: null, charts: {}, timer: null, sess: null };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

// ---------------- time formatting in the greenhouse's time zone
const tz = CFG.tz || undefined;
const fTime = new Intl.DateTimeFormat("en-CA", { timeZone: tz, hour: "2-digit", minute: "2-digit", hour12: false });
const fDay = new Intl.DateTimeFormat("en-GB", { timeZone: tz, day: "2-digit", month: "short" });
const fFull = new Intl.DateTimeFormat("en-CA", { timeZone: tz, year: "numeric", month: "2-digit", day: "2-digit",
  hour: "2-digit", minute: "2-digit", hour12: false });
function ago(ms) {
  const m = Math.round(ms / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m} min ago`;
  const h = m / 60;
  return h < 48 ? `${h.toFixed(h < 10 ? 1 : 0)} h ago` : `${Math.round(h / 24)} days ago`;
}

// ---------------- signed-in session (kept per tab, or on the device if 'keep me signed in')
const store = {
  get() { try { return JSON.parse(sessionStorage.getItem("ghLive") || localStorage.getItem("ghLive") || "null"); } catch { return null; } },
  kept() { try { return !!localStorage.getItem("ghLive"); } catch { return false; } },
  set(s, keep) {
    try { const v = JSON.stringify(s); sessionStorage.setItem("ghLive", v); if (keep) localStorage.setItem("ghLive", v); else localStorage.removeItem("ghLive"); } catch { /* storage blocked */ }
  },
  clear() { try { sessionStorage.removeItem("ghLive"); localStorage.removeItem("ghLive"); } catch { /* storage blocked */ } },
};

// ---------------- database calls
function headers(token) {
  const h = { apikey: CFG.key, "Content-Type": "application/json" };
  if (token) h.Authorization = `Bearer ${token}`;
  else if (!String(CFG.key).startsWith("sb_")) h.Authorization = `Bearer ${CFG.key}`;   // older anon JWT keys
  return h;
}

async function api(path, { method = "GET", body, token } = {}) {
  const r = await fetch(CFG.url + path, { method, headers: headers(token), body: body ? JSON.stringify(body) : undefined });
  const text = await r.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!r.ok) {
    const e = new Error((data && (data.msg || data.message || data.error_description || data.error)) || r.statusText);
    e.status = r.status; throw e;
  }
  return data;
}

const session = (d, email) => ({ access: d.access_token, refresh: d.refresh_token,
  exp: Date.now() + (d.expires_in || 3600) * 1000, email: (d.user && d.user.email) || email });

async function renew() {
  const d = await api("/auth/v1/token?grant_type=refresh_token", { method: "POST", body: { refresh_token: S.sess.refresh } });
  S.sess = session(d, S.sess.email);
  store.set(S.sess, store.kept());
}

async function authed(path, opts = {}) {
  if (S.sess.exp - Date.now() < 60000) await renew();
  try { return await api(path, { ...opts, token: S.sess.access }); }
  catch (e) {
    if (e.status !== 401) throw e;
    await renew();
    return api(path, { ...opts, token: S.sess.access });
  }
}

async function loadRows(hours) {
  const since = new Date(Date.now() - hours * 3600e3).toISOString(), rows = [], page = 1000;
  for (let off = 0; ; off += page) {
    const part = await authed(`/rest/v1/readings?select=*&source=eq.${encodeURIComponent(CFG.source)}` +
      `&ts=gte.${encodeURIComponent(since)}&order=ts.asc&limit=${page}&offset=${off}`);
    rows.push(...part);
    if (part.length < page) break;
  }
  return rows.map((r) => ({ ...r, t: Date.parse(r.ts) }));
}

async function loadLatest() {
  const r = await authed(`/rest/v1/readings?select=*&source=eq.${encodeURIComponent(CFG.source)}&order=ts.desc&limit=1`);
  return r.length ? { ...r[0], t: Date.parse(r[0].ts) } : null;
}

// ---------------- sign in / out
function show(which) {
  $("live-off").hidden = which !== "off"; $("signin").hidden = which !== "signin"; $("view").hidden = which !== "view";
}

async function openView() {
  const ok = await authed("/rest/v1/rpc/is_viewer", { method: "POST", body: {} });
  if (ok !== true) { const e = new Error("not on the viewers list"); e.notViewer = true; throw e; }
  show("view");
  $("pretend-note").hidden = CFG.source !== "pretend";
  await refresh();
  clearInterval(S.timer);
  S.timer = setInterval(() => { if (document.visibilityState === "visible") refresh(); }, 60000);
}

function signOut(msg = "") {
  clearInterval(S.timer);
  if (S.sess) api("/auth/v1/logout", { method: "POST", token: S.sess.access }).catch(() => {});
  S.sess = null; store.clear(); show("signin");
  $("signin-msg").textContent = msg;
}

async function signIn(ev) {
  ev.preventDefault();
  const email = $("email").value.trim(), pw = $("password").value, btn = $("signin-btn"), msg = $("signin-msg");
  if (!email || !pw) { msg.textContent = "Type your e-mail and password."; return; }
  btn.disabled = true; msg.textContent = "Signing in…";
  try {
    const d = await api("/auth/v1/token?grant_type=password", { method: "POST", body: { email, password: pw } });
    S.sess = session(d, email);
    store.set(S.sess, $("keep").checked);
    $("password").value = "";
    msg.textContent = "";
    await openView();
  } catch (e) {
    if (e.notViewer) signOut("This account is not on the viewers list. Ask the dashboard owner to add it.");
    else msg.textContent = e.status === 400 ? "Wrong e-mail or password." :
      e.status ? `Could not sign in (${e.message}).` : "Could not reach the live database. Check the internet connection.";
  } finally { btn.disabled = false; }
}

// ---------------- refresh and render
async function refresh() {
  try {
    const [rows, latest] = await Promise.all([loadRows(RANGES[S.range].h), loadLatest()]);
    S.rows = rows; S.latest = latest;
    renderStatus(); renderAlerts(); renderNow(); renderCharts();
  } catch (e) {
    if (e.status === 401 || e.status === 400) signOut("Your sign-in expired. Please sign in again.");
    else { $("status-text").textContent = `Could not load new readings (${e.message || "no connection"}). Retrying every minute.`; $("dot").className = "dot warn"; }
  }
}

function renderStatus() {
  const L = S.latest;
  if (!L) { $("dot").className = "dot bad"; $("status-text").textContent = "No readings yet."; return; }
  const age = Date.now() - L.t;
  $("dot").className = `dot ${age > 3600e3 ? "bad" : age > 900e3 ? "warn" : "ok"}`;
  $("status-text").textContent = `Last reading ${fFull.format(L.t)} · ${ago(age)} · updates every minute`;
}

function alertBox(kind, text) { return `<div class="alert ${kind}">${esc(text)}</div>`; }

function sensorChecks() {
  const out = [], L = S.latest, rows = S.rows;
  if (!L) return [["bad", "No readings yet. Start the sensor box (or the pretend sensor)."]];
  const age = Date.now() - L.t;
  if (age > 3600e3) out.push(["bad", `No readings for ${ago(age).replace(" ago", "")}. Check that the sensor box has power and internet.`]);
  else if (age > 900e3) out.push(["warn", `No new reading for ${ago(age).replace(" ago", "")} (expected every 5 minutes).`]);
  const NAMES = { temp_c: "Indoor temperature", rh_pct: "Indoor humidity", co2_ppm: "Indoor CO₂",
    temp_out_c: "Outdoor temperature", rh_out_pct: "Outdoor humidity" };
  const RANGE = { temp_c: [-40, 60], temp_out_c: [-50, 50], rh_pct: [0, 100.5], rh_out_pct: [0, 100.5], co2_ppm: [250, 5000] };
  const UNIT = { temp_c: "°C", temp_out_c: "°C", rh_pct: "%", rh_out_pct: "%", co2_ppm: "ppm" };
  for (const [k, [lo, hi]] of Object.entries(RANGE)) {
    const bad = rows.find((r) => r[k] !== null && (r[k] < lo || r[k] > hi));
    if (bad) out.push(["bad", `${NAMES[k]} sensor sent an impossible value (${fmt(bad[k], 1)} ${UNIT[k]} at ${fTime.format(bad.t)}).`]);
  }
  const recent = rows.filter((r) => r.t >= L.t - 3 * 3600e3);
  const EPS = { temp_c: 0.05, rh_pct: 0.2, co2_ppm: 2, temp_out_c: 0.05, rh_out_pct: 0.2 };
  if (recent.length >= 12 && L.t - recent[0].t >= 2.9 * 3600e3) {
    for (const [k, eps] of Object.entries(EPS)) {
      const v = recent.map((r) => r[k]).filter((x) => x !== null);
      if (v.length >= 12 && Math.max(...v) - Math.min(...v) < eps) out.push(["warn", `${NAMES[k]} has not changed for 3 hours: the sensor may be stuck.`]);
    }
  }
  for (const k of ["rh_pct", "rh_out_pct"]) {             // a wet or failing humidity sensor reads ~100 % for hours
    let i = rows.length - 1;
    while (i >= 0 && rows[i][k] >= 99.5) i--;
    const span = i < rows.length - 1 ? L.t - rows[i + 1].t : 0;
    if (span >= 12 * 3600e3) out.push(["warn", `${NAMES[k]} has read about 100 % for ${Math.round(span / 3600e3)} hours: the sensor may be wet or stuck.`]);
  }
  let gaps = 0, longest = 0;
  for (let i = 1; i < rows.length; i++) {
    const d = rows[i].t - rows[i - 1].t;
    if (d > 15 * 60e3) { gaps++; longest = Math.max(longest, d); }
  }
  if (gaps) out.push(["info", `${gaps} gap${gaps > 1 ? "s" : ""} in the readings during this period (longest ${ago(longest).replace(" ago", "")}).`]);
  if (!out.length) out.push(["ok", `Sensors look fine: ${rows.length} readings in this period, no gaps.`]);
  return out;
}

function renderAlerts() { $("alerts").innerHTML = sensorChecks().map(([k, t]) => alertBox(k, t)).join(""); }

function chip(text, kind) { return ` <span class="chip ${kind}">${text}</span>`; }

function renderNow() {
  const r = S.latest;
  if (!r) { $("now").innerHTML = `<p class="muted">No readings yet.</p>`; $("equip").innerHTML = ""; return; }
  const tChip = r.temp_c < r.limit_temp_min - 0.3 ? chip("too cold", "warn") : r.temp_c > r.limit_temp_max + 0.3 ? chip("too warm", "warn") : chip("OK", "ok");
  const hChip = r.rh_pct > LIM.rh_max + 1 ? chip("too humid", "warn") : chip("OK", "ok");
  const cChip = r.co2_ppm > LIM.co2_max + 50 ? chip("high", "warn") : chip("OK", "ok");
  const row = (n, a, b) => `<tr><th scope="row">${n}</th><td>${a}</td><td>${b}</td></tr>`;
  $("now").innerHTML = `<table class="cmp"><thead><tr><th></th><th scope="col">Indoor</th><th scope="col">Outdoor</th></tr></thead><tbody>` +
    row("Temperature", `${fmt(r.temp_c, 1)} °C${tChip}`, `${fmt(r.temp_out_c, 1)} °C`) +
    row("Humidity", `${fmt(r.rh_pct)} %${hChip}`, `${fmt(r.rh_out_pct)} %`) +
    row("CO₂", `${fmt(r.co2_ppm)} ppm${cChip}`, `${fmt(r.co2_out_ppm)} ppm`) +
    row("Sunlight", `<span class="muted">uses outdoor</span>`, `${fmt(r.radiation_w_m2)} W/m²`) + `</tbody></table>` +
    `<p class="muted tiny">Allowed indoor now: ${fmt(r.limit_temp_min)}–${fmt(r.limit_temp_max)} °C, humidity up to ${LIM.rh_max} %, CO₂ up to ${LIM.co2_max} ppm.</p>`;
  const m = (label, value, delta) => `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div>${delta ? `<div class="delta">${delta}</div>` : ""}</div>`;
  $("equip").innerHTML =
    m("Heating", `${fmt(r.heating_w_m2)} W/m²`, r.heating_w_m2 > 1 ? "on" : "off") +
    m("Vents", `${fmt(r.ventilation_mm_s, 1)} mm/s`, r.ventilation_mm_s > 0.1 ? "open" : "closed") +
    m("CO₂ dosing", `${fmt(r.co2_supply_mg_m2_s * 3.6, 1)} g/m²/h`, r.co2_supply_mg_m2_s > 0.001 ? "on" : "off") +
    m(`Lettuce · crop ${r.crop_cycle ?? "–"}`, `${fmt(r.dry_weight_g_m2)} g/m²`, "dry weight");
}

// ---------------- charts (averaged into time buckets; empty buckets break the line = gaps)
function buckets() {
  const { h, b } = RANGES[S.range], step = b * 60e3, end = Math.ceil(Date.now() / step) * step, start = end - h * 3600e3;
  const n = Math.round((end - start) / step), sums = new Map();
  for (const r of S.rows) {
    const i = Math.floor((r.t - start) / step);
    if (i < 0 || i >= n) continue;
    if (!sums.has(i)) sums.set(i, {});
    const s = sums.get(i);
    for (const [k, v] of Object.entries(r)) {
      if (typeof v === "number" && k !== "t") { s[k] = s[k] || [0, 0]; s[k][0] += v; s[k][1]++; }
    }
  }
  const times = Array.from({ length: n }, (_, i) => start + i * step);
  const series = (k) => times.map((_, i) => { const s = sums.get(i); return s && s[k] ? s[k][0] / s[k][1] : null; });
  const labels = times.map((t) => (h <= 24 ? fTime.format(t) : `${fDay.format(t)} ${fTime.format(t)}`));
  return { labels, series };
}

function chart(id, labels, datasets) {
  const css = getComputedStyle(document.documentElement);
  const ink = css.getPropertyValue("--muted").trim() || "#5b6670", grid = css.getPropertyValue("--line").trim() || "#dde3e8";
  const cfg = {
    type: "line", data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false, spanGaps: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: datasets.filter((d) => !d.hideLegend).length > 1,
        labels: { color: ink, boxWidth: 12, filter: (it, data) => !data.datasets[it.datasetIndex].hideLegend } },
        tooltip: { callbacks: { label: (c) => (c.dataset.hideLegend || c.parsed.y === null ? null : `${c.dataset.label}: ${c.parsed.y.toFixed(1)}`) } } },
      elements: { point: { radius: 0 }, line: { borderWidth: 2.2, tension: 0.2 } },
      scales: { x: { ticks: { color: ink, maxTicksLimit: 6, maxRotation: 0 }, grid: { color: grid } },
        y: { ticks: { color: ink }, grid: { color: grid } } },
    },
  };
  if (S.charts[id]) S.charts[id].destroy();
  S.charts[id] = new Chart($(id), cfg);
}

function renderCharts() {
  if (typeof Chart === "undefined") { $("chart-note").textContent = "Charts need an internet connection to load."; return; }
  const { labels, series } = buckets(), b = RANGES[S.range].b;
  $("chart-note").textContent = `Grey line = outdoor, coloured line = indoor. ${b === 5 ? "Every reading (5 min)" : `${b}-min averages`}; a break in a line means no readings.`;
  chart("l-temp", labels, [
    { label: "min", data: series("limit_temp_min"), borderColor: "transparent", hideLegend: true },
    { label: "max", data: series("limit_temp_max"), borderColor: "transparent", backgroundColor: "rgba(0,114,178,.13)", fill: "-1", hideLegend: true },
    { label: "Indoor", data: series("temp_c"), borderColor: COL.blue },
    { label: "Outdoor", data: series("temp_out_c"), borderColor: COL.grey }]);
  chart("l-rh", labels, [{ label: "Indoor", data: series("rh_pct"), borderColor: COL.green },
    { label: "Outdoor", data: series("rh_out_pct"), borderColor: COL.grey },
    { label: "max", data: labels.map(() => LIM.rh_max), borderColor: COL.red, borderDash: [5, 4], borderWidth: 1.5, hideLegend: true }]);
  chart("l-co2", labels, [{ label: "Indoor", data: series("co2_ppm"), borderColor: COL.pink },
    { label: "Outdoor", data: series("co2_out_ppm"), borderColor: COL.grey }]);
  chart("l-co2o", labels, [{ label: "Outdoor", data: series("co2_out_ppm"), borderColor: COL.dark }]);
  chart("l-sun", labels, [{ label: "Outdoor", data: series("radiation_w_m2"), borderColor: COL.gold, backgroundColor: "rgba(230,159,0,.3)", fill: true }]);
  chart("l-heat", labels, [{ label: "Heating", data: series("heating_w_m2"), borderColor: COL.red, backgroundColor: "rgba(213,94,0,.25)", fill: true }]);
  chart("l-vent", labels, [{ label: "Vents", data: series("ventilation_mm_s"), borderColor: COL.sky, backgroundColor: "rgba(86,180,233,.3)", fill: true }]);
}

// ---------------- start
function boot() {
  if (!CFG.url || !CFG.key) { show("off"); return; }
  $("signin-form").addEventListener("submit", signIn);
  $("signout").addEventListener("click", () => signOut());
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    S.range = b.dataset.range;
    document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    refresh();
  }));
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible" && S.sess && !$("view").hidden) refresh(); });
  const saved = store.get();
  if (saved && saved.refresh) {
    S.sess = saved; $("keep").checked = store.kept();
    openView().catch((e) => signOut(e.notViewer ? "This account is not on the viewers list." : "Please sign in again."));
  } else show("signin");
}

boot();
