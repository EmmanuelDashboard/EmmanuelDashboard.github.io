"use strict";
/* Live greenhouse monitor. Signs in to the live database (Supabase Auth, e-mail + password) and shows
   the newest readings, sensor checks and charts. Plain fetch calls: no library besides Chart.js.
   Only e-mail addresses on the database's viewers list can read the data (row-level security). */

const CFG = window.LIVE_CONFIG || {};
const LIM = { rh_max: 80, co2_max: 1600, ...(CFG.limits || {}) };
// The sensor sends a reading every 20 s ('raw') and a 5-minute average ('sum', source name + ':5min').
// 6 h charts use the raw readings; longer periods use the averages, so phones download little data.
const EVERY_S = 20, REFRESH_MS = 20000;
const RANGES = { "30m": { h: 0.5, b: 1 / 3, src: "raw" }, "6h": { h: 6, b: 1, src: "raw" }, "24h": { h: 24, b: 5, src: "sum" },
  "3d": { h: 72, b: 30, src: "sum" }, "7d": { h: 168, b: 60, src: "sum" } };
const GAP_MS = { raw: 2 * 60e3, sum: 15 * 60e3 };
const COL = { blue: "#0072B2", grey: "#999999", dark: "#6f6f6f", green: "#009E73", red: "#D55E00",
  sky: "#56B4E9", gold: "#E69F00", pink: "#CC79A7" };
const S = { range: "30m", cache: { raw: { rows: [], from: null }, sum: { rows: [], from: null } },
  latest: null, charts: {}, timer: null, sess: null, busy: false, tab: "summary", ai: null,
  fc: { rows: [], fetched: 0 } };       // the controller's (MPC) 6-hour forecast, refreshed every 2 minutes
const PRICES = { heat_cad_per_kwh: 0.0374, co2_cad_per_kg: 0.30, ...(CFG.prices || {}) };   // same assumptions as the season page
const RANGE_LABEL = { "30m": "the last 30 minutes", "6h": "the last 6 hours", "24h": "the last 24 hours", "3d": "the last 3 days", "7d": "the last 7 days" };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

// ---------------- time formatting in the greenhouse's time zone
const tz = CFG.tz || undefined;
const fTime = new Intl.DateTimeFormat("en-CA", { timeZone: tz, hour: "2-digit", minute: "2-digit", hour12: false });
const fDay = new Intl.DateTimeFormat("en-GB", { timeZone: tz, day: "2-digit", month: "short" });
const fClock = new Intl.DateTimeFormat("en-CA", { timeZone: tz, hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
const fFull = new Intl.DateTimeFormat("en-CA", { timeZone: tz, year: "numeric", month: "2-digit", day: "2-digit",
  hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
function ago(ms) {
  if (ms < 60000) return `${Math.max(0, Math.round(ms / 1000))} s ago`;
  const m = Math.round(ms / 60000);
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

const srcName = (kind) => (kind === "raw" ? CFG.source : kind === "fc" ? `${CFG.source}:forecast` : `${CFG.source}:5min`);
const iso = (ms) => encodeURIComponent(new Date(ms).toISOString());

async function fetchRows(kind, { gte, gt, lt } = {}) {
  let q = `/rest/v1/readings?select=*&source=eq.${encodeURIComponent(srcName(kind))}`;
  if (gte !== undefined) q += `&ts=gte.${iso(gte)}`;
  if (gt !== undefined) q += `&ts=gt.${iso(gt)}`;
  if (lt !== undefined) q += `&ts=lt.${iso(lt)}`;
  const rows = [], page = 1000;
  for (let off = 0; ; off += page) {
    const part = await authed(`${q}&order=ts.asc&limit=${page}&offset=${off}`);
    rows.push(...part);
    if (part.length < page) break;
  }
  return rows.map((r) => ({ ...r, t: Date.parse(r.ts) }));
}

// Keep what was already downloaded; only ask for older rows (when the period grows) and newer rows.
async function sync(kind, hours) {
  const c = S.cache[kind], since = Date.now() - hours * 3600e3;
  if (c.from === null || since < c.from) {
    const older = await fetchRows(kind, { gte: since, lt: c.from === null ? undefined : c.from });
    c.rows = older.concat(c.rows);
    c.from = since;
  }
  if (c.rows.length) c.rows = c.rows.concat(await fetchRows(kind, { gt: c.rows[c.rows.length - 1].t }));
  const keep = Date.now() - 7.2 * 24 * 3600e3;
  if (c.rows.length && c.rows[0].t < keep) { c.rows = c.rows.filter((r) => r.t >= keep); c.from = Math.max(c.from, keep); }
}

function rowsInRange() {
  const R = RANGES[S.range], since = Date.now() - R.h * 3600e3;
  return S.cache[R.src].rows.filter((r) => r.t >= since);
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
  S.timer = setInterval(() => { if (document.visibilityState === "visible") refresh(); }, REFRESH_MS);
}

function signOut(msg = "") {
  clearInterval(S.timer);
  if (S.sess) api("/auth/v1/logout", { method: "POST", token: S.sess.access }).catch(() => {});
  S.sess = null; store.clear(); show("signin");
  S.cache = { raw: { rows: [], from: null }, sum: { rows: [], from: null } };
  S.fc = { rows: [], fetched: 0 };
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
  if (S.busy) return;
  S.busy = true;
  try {
    await sync("raw", 6);
    await sync("sum", Math.max(24, RANGES[S.range].h));     // at least 24 h for the humidity-sensor check
    if (Date.now() - S.fc.fetched > 120e3) {
      S.fc.rows = await fetchRows("fc", { gt: Date.now() });
      S.fc.fetched = Date.now();
    }
    const raw = S.cache.raw.rows;
    S.latest = raw.length ? raw[raw.length - 1] : await loadLatest();
    renderStatus(); renderAlerts(); renderNow(); renderForecast(); renderPeriod();
    if (S.tab === "charts") renderCharts();
    if (S.tab === "facts") renderFacts();
  } catch (e) {
    if (e.status === 401 || e.status === 400) signOut("Your sign-in expired. Please sign in again.");
    else { $("status-text").textContent = `Could not load new readings (${e.message || "no connection"}). Retrying every 20 seconds.`; $("dot").className = "dot warn"; }
  } finally { S.busy = false; }
}

function renderStatus() {
  const L = S.latest;
  if (!L) { $("dot").className = "dot bad"; $("status-text").textContent = "No readings yet."; return; }
  const age = Date.now() - L.t;
  $("dot").className = `dot ${age > 15 * 60e3 ? "bad" : age > 2 * 60e3 ? "warn" : "ok"}`;
  $("status-text").textContent = `Last reading ${fFull.format(L.t)} · ${ago(age)} · updates every 20 seconds`;
}

function alertBox(kind, text) { return `<div class="alert ${kind}">${esc(text)}</div>`; }

function sensorChecks() {
  const out = [], L = S.latest, rows = S.cache.raw.rows, sums = S.cache.sum.rows;
  if (!L) return [["bad", "No readings yet. Start the sensor box (or the pretend sensor)."]];
  const age = Date.now() - L.t;
  if (age > 15 * 60e3) out.push(["bad", `No readings for ${ago(age).replace(" ago", "")}. Check that the sensor box has power and internet.`]);
  else if (age > 2 * 60e3) out.push(["warn", `No new reading for ${ago(age).replace(" ago", "")} (expected every ${EVERY_S} seconds).`]);
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
    let i = sums.length - 1;
    while (i >= 0 && sums[i][k] >= 99.5) i--;
    const span = i < sums.length - 1 ? sums[sums.length - 1].t - sums[i + 1].t + 5 * 60e3 : 0;
    if (span >= 12 * 3600e3) out.push(["warn", `${NAMES[k]} has read about 100 % for ${Math.round(span / 3600e3)} hours: the sensor may be wet or stuck.`]);
  }
  const shown = rowsInRange(), gapMs = GAP_MS[RANGES[S.range].src];
  let gaps = 0, longest = 0;
  for (let i = 1; i < shown.length; i++) {
    const d = shown[i].t - shown[i - 1].t;
    if (d > gapMs) { gaps++; longest = Math.max(longest, d); }
  }
  if (gaps) out.push(["info", `${gaps} gap${gaps > 1 ? "s" : ""} in the readings during this period (longest ${ago(longest).replace(" ago", "")}).`]);
  if (!out.length) {
    const n = RANGES[S.range].src === "raw" ? `${shown.length} readings` : `${shown.length} five-minute averages`;
    out.push(["ok", `Sensors look fine: ${n} in this period, no gaps.`]);
  }
  return out;
}

function renderAlerts() { $("alerts").innerHTML = sensorChecks().map(([k, t]) => alertBox(k, t)).join(""); }

function chip(text, kind) { return ` <span class="chip ${kind}">${text}</span>`; }

// a small line of the last 30 minutes (own scale, so small changes show) and the change over 5 minutes
function spark(key, color) {
  const since = S.latest.t - 30 * 60e3;
  const pts = S.cache.raw.rows.filter((r) => r.t >= since && r[key] !== null && r[key] !== undefined);
  if (pts.length < 3) return "";
  const W = 120, H = 24, vals = pts.map((r) => r[key]), lo = Math.min(...vals), hi = Math.max(...vals), rng = hi - lo || 1;
  const t0 = pts[0].t, dt = (pts[pts.length - 1].t - t0) || 1;
  const xy = pts.map((r) => `${((r.t - t0) / dt * W).toFixed(1)},${(H - 2 - (r[key] - lo) / rng * (H - 4)).toFixed(1)}`);
  const [lx, ly] = xy[xy.length - 1].split(",");
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">` +
    `<polyline points="${xy.join(" ")}" fill="none" stroke="${color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>` +
    `<circle cx="${lx}" cy="${ly}" r="2.2" fill="${color}"/></svg>`;
}

function trend(key, digits) {
  const L = S.latest, past = S.cache.raw.rows.filter((r) => r.t <= L.t - 5 * 60e3 && r.t >= L.t - 7 * 60e3);
  if (!past.length || L[key] === null || past[past.length - 1][key] === null) return "";
  const d = L[key] - past[past.length - 1][key], small = Math.abs(d) < (digits ? 0.05 : 0.5);
  return `<span class="trend" title="change in the last 5 minutes">${small ? "→ steady" : `${d > 0 ? "▲" : "▼"} ${Math.abs(d).toFixed(digits)}`}</span>`;
}

function renderNow() {
  const r = S.latest;
  if (!r) { $("now").innerHTML = `<p class="muted">No readings yet.</p>`; $("equip").innerHTML = ""; return; }
  if (S.shownT && S.shownT !== r.t) {                       // a new reading arrived: flash the panel
    const card = $("now-card"); card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash");
  }
  S.shownT = r.t;
  const tChip = r.temp_c < r.limit_temp_min - 0.3 ? chip("too cold", "warn") : r.temp_c > r.limit_temp_max + 0.3 ? chip("too warm", "warn") : chip("OK", "ok");
  const hChip = r.rh_pct > LIM.rh_max + 1 ? chip("too humid", "warn") : chip("OK", "ok");
  const cChip = r.co2_ppm > LIM.co2_max + 50 ? chip("high", "warn") : chip("OK", "ok");
  const row = (n, a, b) => `<tr><th scope="row">${n}</th><td>${a}</td><td>${b}</td></tr>`;
  $("now").innerHTML = `<table class="cmp"><thead><tr><th></th><th scope="col">Indoor</th><th scope="col">Outdoor</th></tr></thead><tbody>` +
    row("Temperature", `${fmt(r.temp_c, 1)} °C${tChip}${trend("temp_c", 1)}${spark("temp_c", COL.blue)}`,
      `${fmt(r.temp_out_c, 1)} °C${trend("temp_out_c", 1)}${spark("temp_out_c", COL.grey)}`) +
    row("Humidity", `${fmt(r.rh_pct)} %${hChip}${trend("rh_pct", 0)}${spark("rh_pct", COL.green)}`,
      `${fmt(r.rh_out_pct)} %${trend("rh_out_pct", 0)}${spark("rh_out_pct", COL.grey)}`) +
    row("CO₂", `${fmt(r.co2_ppm)} ppm${cChip}${trend("co2_ppm", 0)}${spark("co2_ppm", COL.pink)}`,
      `${fmt(r.co2_out_ppm)} ppm${trend("co2_out_ppm", 0)}${spark("co2_out_ppm", COL.grey)}`) +
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
  for (const r of rowsInRange()) {
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
  const labels = times.map((t) => (h < 1 ? fClock.format(t) : h <= 24 ? fTime.format(t) : `${fDay.format(t)} ${fTime.format(t)}`));
  return { labels, series };
}

function chart(id, labels, datasets) {
  for (const d of datasets) {                                // a dot on the newest value of each line
    if (d.hideLegend || d.fill || d.noDot) continue;
    let last = -1;
    d.data.forEach((v, i) => { if (v !== null) last = i; });
    d.pointRadius = d.data.map((_, i) => (i === last ? 3.5 : 0));
    d.pointBackgroundColor = d.borderColor;
  }
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

// ---------------- the controller's 6-hour forecast
const fcOld = () => !S.fc.rows.length || Date.now() - (S.fc.rows[0].t - 15 * 60e3) > 45 * 60e3;
const fcMade = () => (S.fc.rows.length ? S.fc.rows[0].t - 15 * 60e3 : null);

// the most likely reason for a planned change in heating, from what the controller knows; 'up' = heating rises
function why(r, L, up = r.heating_w_m2 >= L.heating_w_m2) {
  const reasons = [];
  if (up) {
    if (r.limit_temp_min > L.limit_temp_min + 0.5) reasons.push(`the higher daytime minimum (${r0(r.limit_temp_min)} °C) starts`);
    if (r.radiation_w_m2 < L.radiation_w_m2 - 50) reasons.push("the sun goes down");
    if (r.temp_out_c < L.temp_out_c - 1) reasons.push(`it gets colder outdoors (${f1(r.temp_out_c)} °C)`);
  } else {
    if (r.limit_temp_min < L.limit_temp_min - 0.5) reasons.push(`the lower night minimum (${r0(r.limit_temp_min)} °C) starts`);
    if (r.radiation_w_m2 > L.radiation_w_m2 + 50) reasons.push("the sun warms the greenhouse");
    if (r.temp_out_c > L.temp_out_c + 1) reasons.push(`it gets milder outdoors (${f1(r.temp_out_c)} °C)`);
  }
  return reasons.length ? `because ${reasons.slice(0, 2).join(" and ")}` : "";
}

function forecastEvents() {
  const F = S.fc.rows, L = S.latest, ev = [];
  if (!F.length || !L) return ev;
  const lim = F.find((r) => Math.abs(r.limit_temp_min - L.limit_temp_min) > 0.5);
  if (lim) ev.push({ t: lim.t, x: `Around ${fTime.format(lim.t - 15 * 60e3)}–${fTime.format(lim.t)} the ${lim.limit_temp_min > L.limit_temp_min ? "daytime" : "night"} temperature limits start (${r0(lim.limit_temp_min)}–${r0(lim.limit_temp_max)} °C).` });
  const hi = F.reduce((a, r) => (r.heating_w_m2 > a.heating_w_m2 ? r : a)), lo = F.reduce((a, r) => (r.heating_w_m2 < a.heating_w_m2 ? r : a));
  if (hi.heating_w_m2 - L.heating_w_m2 >= 15) ev.push({ t: hi.t, x: `Heating will rise from ${r0(L.heating_w_m2)} to about ${r0(hi.heating_w_m2)} W/m² by ${fTime.format(hi.t)} ${why(hi, L)}`.trim() + "." });
  if (L.heating_w_m2 - lo.heating_w_m2 >= 15) ev.push({ t: lo.t, x: `Heating will fall from ${r0(L.heating_w_m2)} to about ${r0(lo.heating_w_m2)} W/m² by ${fTime.format(lo.t)} ${why(lo, L)}`.trim() + "." });
  const vent = runs(F, (r) => r.ventilation_mm_s > 0.5, 0, 16 * 60e3)[0];
  if (vent) {
    const rs = F.slice(vent[0], vent[1] + 1);
    ev.push({ t: F[vent[0]].t, x: `The vents should open more (up to ${f1(Math.max(...rs.map((r) => r.ventilation_mm_s)))} mm/s) from about ${fTime.format(F[vent[0]].t)} to ${fTime.format(F[vent[1]].t)}, when indoor reaches ${f1(Math.max(...rs.map((r) => r.temp_c)))} °C.` });
  }
  const dosing = (r) => r.co2_supply_mg_m2_s > 0.01, sw = F.find((r) => dosing(r) !== dosing(L));
  if (sw) ev.push({ t: sw.t, x: `CO₂ dosing should ${dosing(sw) ? "start" : "stop"} around ${fTime.format(sw.t)} ${dosing(sw) ? "(daylight: it helps the lettuce grow)" : "(less light, so extra CO₂ helps less)"}.` });
  const cold = F.find((r) => r.temp_c < r.limit_temp_min - 0.3), warm = F.find((r) => r.temp_c > r.limit_temp_max + 0.3);
  if (cold) ev.push({ t: cold.t, x: `The greenhouse may get a little too cold around ${fTime.format(cold.t)} (${f1(cold.temp_c)} °C, minimum ${r0(cold.limit_temp_min)} °C).` });
  if (warm) ev.push({ t: warm.t, x: `The greenhouse may get too warm around ${fTime.format(warm.t)} (${f1(warm.temp_c)} °C, maximum ${r0(warm.limit_temp_max)} °C).` });
  if (!ev.length) ev.push({ t: F[0].t, x: `No big changes expected: indoor ${span(Math.min(...F.map((r) => r.temp_c)), Math.max(...F.map((r) => r.temp_c)), 1)} °C, heating ${span(lo.heating_w_m2, hi.heating_w_m2)} W/m².` });
  return ev.sort((a, b) => a.t - b.t);
}

function renderForecast() {
  const F = S.fc.rows, L = S.latest;
  if (!F.length || !L) {
    $("fc-table").innerHTML = `<p class="muted">No forecast yet. The controller makes a new 6-hour forecast every 15 minutes while the sensor runs.</p>`;
    $("fc-events").innerHTML = ""; $("fc-note").textContent = ""; return;
  }
  const pick = (h) => F.reduce((a, r) => (Math.abs(r.t - (fcMade() + h * 3600e3)) < Math.abs(a.t - (fcMade() + h * 3600e3)) ? r : a));
  const row = (label, r) => `<tr><th scope="row">${label}<br><span class="muted tiny">${fTime.format(r.t)}</span></th><td>${f1(r.temp_c)} °C</td><td>${r0(r.rh_pct)} %</td><td>${r0(r.co2_ppm)} ppm</td><td>${r0(r.heating_w_m2)} W/m²</td></tr>`;
  $("fc-table").innerHTML = `<table class="cmp fc"><thead><tr><th></th><th>Indoor</th><th>Humidity</th><th>CO₂</th><th>Heating</th></tr></thead><tbody>` +
    row("Now", L) + row("In 1 h", pick(1)) + row("In 3 h", pick(3)) + row("In 6 h", pick(6)) + `</tbody></table>`;
  $("fc-events").innerHTML = forecastEvents().map((e) => `<li>${esc(e.x)}</li>`).join("");
  $("fc-note").textContent = `Forecast made at ${fTime.format(fcMade())} by the controller (MPC): its own predictions and plan. ` +
    (CFG.source === "pretend" ? "With the pretend sensor this is the forecast the simulation's controller made at that moment. " : "") +
    "It assumes the weather forecast is right." + (fcOld() ? " This forecast is old: the sensor may have stopped." : "");
}

function renderForecastCharts() {
  const F = S.fc.rows, L = S.latest;
  if (!F.length || !L) { $("fc-chart-note").textContent = "No forecast yet."; return; }
  const step = 5 * 60e3, now = Date.now(), start = Math.floor((now - 3600e3) / step) * step, end = F[F.length - 1].t;
  const n = Math.round((end - start) / step) + 1, times = Array.from({ length: n }, (_, i) => start + i * step);
  const nowIdx = Math.min(n - 1, Math.floor((L.t - start) / step));
  const measured = (k) => {
    const s = new Map();
    for (const r of S.cache.raw.rows) {
      const i = Math.floor((r.t - start) / step);
      if (i < 0 || i >= n || r[k] === null) continue;
      const v = s.get(i) || [0, 0]; v[0] += r[k]; v[1]++; s.set(i, v);
    }
    return times.map((_, i) => (s.has(i) ? s.get(i)[0] / s.get(i)[1] : null));
  };
  const predicted = (k) => {
    const out = times.map(() => null);
    out[nowIdx] = L[k];                                        // join the dashed line to the latest reading
    for (const r of F) { const i = Math.round((r.t - start) / step); if (i >= 0 && i < n) out[i] = r[k]; }
    return out;
  };
  const both = (k) => { const m = measured(k), p = predicted(k); return m.map((v, i) => (i <= nowIdx && v !== null ? v : p[i])); };
  const labels = times.map((t, i) => (i === nowIdx ? "now" : fTime.format(t)));
  const fc = (label, k, color) => ({ label: `${label} forecast`, data: predicted(k), borderColor: color, borderDash: [6, 4], spanGaps: true, noDot: true });
  $("fc-chart-note").textContent = `Forecast made at ${fTime.format(fcMade())}; it reaches ${fTime.format(end)}.` + (fcOld() ? " It is old: the sensor may have stopped." : "");
  chart("f-temp", labels, [
    { label: "min", data: both("limit_temp_min"), borderColor: "transparent", hideLegend: true, spanGaps: true },
    { label: "max", data: both("limit_temp_max"), borderColor: "transparent", backgroundColor: "rgba(0,114,178,.13)", fill: "-1", hideLegend: true, spanGaps: true },
    { label: "Indoor", data: measured("temp_c"), borderColor: COL.blue }, fc("Indoor", "temp_c", COL.blue),
    { label: "Outdoor", data: measured("temp_out_c"), borderColor: COL.grey }, fc("Outdoor", "temp_out_c", COL.grey)]);
  chart("f-rh", labels, [
    { label: "Indoor", data: measured("rh_pct"), borderColor: COL.green }, fc("Indoor", "rh_pct", COL.green),
    { label: "Outdoor", data: measured("rh_out_pct"), borderColor: COL.grey }, fc("Outdoor", "rh_out_pct", COL.grey)]);
  chart("f-heat", labels, [{ label: "Heating", data: measured("heating_w_m2"), borderColor: COL.red },
    fc("Heating", "heating_w_m2", COL.red)]);
  chart("f-co2", labels, [{ label: "Indoor", data: measured("co2_ppm"), borderColor: COL.pink }, fc("Indoor", "co2_ppm", COL.pink)]);
}

function renderCharts() {
  if (typeof Chart === "undefined") { $("chart-note").textContent = "Charts need an internet connection to load."; return; }
  renderForecastCharts();
  $("hist-label").textContent = `· ${RANGE_LABEL[S.range]}`;
  const { labels, series } = buckets(), R = RANGES[S.range];
  $("chart-note").textContent = `Grey line = outdoor, coloured line = indoor. ` +
    (R.b < 1 ? `Every reading (every ${EVERY_S} s)` : R.src === "raw" ? `Readings every ${EVERY_S} s, shown as 1-min averages` : `${R.b}-min averages`) +
    "; a break in a line means no readings. Updates every 20 seconds.";
  chart("l-temp", labels, [
    { label: "min", data: series("limit_temp_min"), borderColor: "transparent", hideLegend: true },
    { label: "max", data: series("limit_temp_max"), borderColor: "transparent", backgroundColor: "rgba(0,114,178,.13)", fill: "-1", hideLegend: true },
    { label: "Indoor", data: series("temp_c"), borderColor: COL.blue },
    { label: "Outdoor", data: series("temp_out_c"), borderColor: COL.grey }]);
  chart("l-tin", labels, [{ label: "Indoor", data: series("temp_c"), borderColor: COL.blue }]);
  chart("l-rhin", labels, [{ label: "Indoor", data: series("rh_pct"), borderColor: COL.green }]);
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

// ---------------- tabs
function showTab(t) {
  S.tab = t;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === t)));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${t}`));
  if (t === "charts") renderCharts();
  if (t === "facts") renderFacts();
}

// ---------------- helpers for the period summary, facts and answers
const r0 = (x) => Math.round(x), f1 = (x) => Number(x).toFixed(1);
const span = (a, b, d = 0) => (a === null ? "–" : (Math.round(a) < 0 || Math.round(b) < 0)
  ? `${Number(a).toFixed(d)} to ${Number(b).toFixed(d)}` : `${Number(a).toFixed(d)}–${Number(b).toFixed(d)}`);
const fDate = new Intl.DateTimeFormat("en-CA", { timeZone: tz, year: "numeric", month: "2-digit", day: "2-digit" });
const longRange = () => RANGES[S.range].h > 24;
const tStr = (t, withDay) => (withDay ? `${fDay.format(t)} ${fTime.format(t)}` : fTime.format(t));

function stats(rows, dt) {
  const mm = (k) => {
    let lo = Infinity, hi = -Infinity;
    for (const r of rows) { const v = r[k]; if (v === null || v === undefined) continue; if (v < lo) lo = v; if (v > hi) hi = v; }
    return lo === Infinity ? [null, null] : [lo, hi];
  };
  const tot = (k) => rows.reduce((s, r) => s + (r[k] || 0), 0);
  const hours = (f) => rows.filter(f).length * dt / 3600;
  const first = rows[0], last = rows[rows.length - 1];
  return {
    n: rows.length, T: mm("temp_c"), To: mm("temp_out_c"), RH: mm("rh_pct"), RHo: mm("rh_out_pct"),
    C: mm("co2_ppm"), Co: mm("co2_out_ppm"),
    kwh: tot("heating_w_m2") * dt / 3.6e6, co2g: tot("co2_supply_mg_m2_s") * dt / 1000,
    sun: tot("radiation_w_m2") * dt / 3.6e6, heatAvg: tot("heating_w_m2") / Math.max(1, rows.length),
    ventH: hours((r) => r.ventilation_mm_s > 0.1), coldH: hours((r) => r.temp_c < r.limit_temp_min - 0.3),
    warmH: hours((r) => r.temp_c > r.limit_temp_max + 0.3), humidH: hours((r) => r.rh_pct >= LIM.rh_max - 0.5),
    growth: first && last && first.crop_cycle === last.crop_cycle ? last.dry_weight_g_m2 - first.dry_weight_g_m2 : null,
    crop: last ? last.crop_cycle : null, dw: last ? last.dry_weight_g_m2 : null,
  };
}
const stepS = (src) => (src === "raw" ? EVERY_S : 300);

// ---------------- summary of the chosen period
function renderPeriod() {
  const R = RANGES[S.range], rows = rowsInRange();
  $("per-title").textContent = `This period · ${RANGE_LABEL[S.range]}`;
  if (!rows.length) { $("per-compare").innerHTML = `<p class="muted">No readings in this period.</p>`; $("per-metrics").innerHTML = ""; $("per-note").textContent = ""; return; }
  const s = stats(rows, stepS(R.src));
  const row = (n, a, b) => `<tr><th scope="row">${n}</th><td>${a}</td><td>${b}</td></tr>`;
  $("per-compare").innerHTML = `<table class="cmp"><thead><tr><th></th><th scope="col">Indoor</th><th scope="col">Outdoor</th></tr></thead><tbody>` +
    row("Temperature", `${span(...s.T)} °C`, `${span(...s.To)} °C`) +
    row("Humidity", `${span(...s.RH)} %`, `${span(...s.RHo)} %`) +
    row("CO₂", `${span(...s.C)} ppm`, `${span(...s.Co)} ppm`) +
    row("Sunlight", `<span class="muted">uses outdoor</span>`, `${f1(s.sun)} kWh/m²`) + `</tbody></table>`;
  $("per-note").textContent = `Lowest–highest ${R.src === "raw" ? "of the 20-second readings" : "of the 5-minute averages"}. ` +
    `Too cold ${f1(s.coldH)} h, too warm ${f1(s.warmH)} h, humidity at its ${LIM.rh_max} % limit ${f1(s.humidH)} h.`;
  const m = (label, value, delta) => `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div>${delta ? `<div class="delta">${delta}</div>` : ""}</div>`;
  $("per-metrics").innerHTML =
    m("Heating used", `${s.kwh.toFixed(2)} kWh/m²`, `${(s.kwh * PRICES.heat_cad_per_kwh).toFixed(2)} CAD/m² (assumed price)`) +
    m("CO₂ added", `${r0(s.co2g)} g/m²`, `${(s.co2g / 1000 * PRICES.co2_cad_per_kg).toFixed(3)} CAD/m²`) +
    m("Vents open", `${f1(s.ventH)} h`, `of ${f1(s.n * stepS(R.src) / 3600)} h`) +
    m(`Lettuce · crop ${s.crop ?? "–"}`, `${r0(s.dw)} g/m²`, s.growth !== null ? `grew ${f1(s.growth)} g/m² in this period` : "a harvest happened in this period");
}

// ---------------- facts written from the live readings
function runs(rows, pred, minMs, joinMs) {
  const out = [];
  let a = null, b = null;
  for (let i = 0; i < rows.length; i++) {
    if (!pred(rows[i])) continue;
    if (a !== null && rows[i].t - rows[b].t <= joinMs) b = i;
    else { if (a !== null) out.push([a, b]); a = b = i; }
  }
  if (a !== null) out.push([a, b]);
  return out.filter(([i, j]) => rows[j].t - rows[i].t >= minMs);
}

function makeFacts() {
  const R = RANGES[S.range], rows = rowsInRange(), dt = stepS(R.src), wd = longRange(), facts = [];
  if (!rows.length) return facts;
  const step = dt * 1000, join = Math.max(3 * step, 5 * 60e3), when = (i, j) => `From ${tStr(rows[i].t, wd)} to ${tStr(rows[j].t + step, wd)}`;
  const seg = (i, j) => rows.slice(i, j + 1);
  const maxOf = (rs, k) => Math.max(...rs.map((r) => r[k])), minOf = (rs, k) => Math.min(...rs.map((r) => r[k]));
  // daily summaries
  const days = new Map();
  for (const r of rows) { const d = fDate.format(r.t); if (!days.has(d)) days.set(d, []); days.get(d).push(r); }
  const today = fDate.format(Date.now());
  for (const [d, rs] of days) {
    const s = stats(rs, dt);
    facts.push({ v: "day", t: rs[0].t, x: `${d}${d === today ? " (so far)" : ""}: indoor ${span(...s.T, 1)} °C, outdoor ${span(...s.To, 1)} °C; ` +
      `humidity indoor up to ${r0(s.RH[1])} %, outdoor ${span(...s.RHo)} %; CO₂ indoor up to ${r0(s.C[1])} ppm, outdoor ${span(...s.Co)} ppm; ` +
      `heating ${s.kwh.toFixed(2)} kWh/m²; vents open ${f1(s.ventH)} h; CO₂ added ${r0(s.co2g)} g/m²` +
      (s.coldH || s.warmH ? `; too cold ${f1(s.coldH)} h, too warm ${f1(s.warmH)} h` : "") + "." });
  }
  // too cold / too warm
  for (const [i, j] of runs(rows, (r) => r.temp_c < r.limit_temp_min - 0.3, 5 * 60e3, join)) {
    const rs = seg(i, j), lo = minOf(rs, "temp_c"), sunrise = rs[0].radiation_w_m2 > 5 && rs[0].radiation_w_m2 < 150;
    facts.push({ v: "temperature", t: rows[i].t, x: `${when(i, j)} the greenhouse was too cold: lowest ${f1(lo)} °C (minimum ${r0(rs[0].limit_temp_min)} °C) while it was ${f1(rs[0].temp_out_c)} °C outdoors` +
      (sunrise ? ", at sunrise when the higher daytime minimum starts and heating is catching up." : "; heating was " + r0(maxOf(rs, "heating_w_m2")) + " W/m² at most.") });
  }
  for (const [i, j] of runs(rows, (r) => r.temp_c > r.limit_temp_max + 0.3, 5 * 60e3, join)) {
    const rs = seg(i, j), hi = maxOf(rs, "temp_c"), warmOut = rs.filter((r) => r.temp_out_c > r.limit_temp_max).length > rs.length / 2;
    facts.push({ v: "temperature", t: rows[i].t, x: `${when(i, j)} the greenhouse was too warm: highest ${f1(hi)} °C (maximum ${r0(rs[0].limit_temp_max)} °C)` +
      (warmOut ? ", because it was warmer outdoors than the limit." : `; the vents opened up to ${f1(maxOf(rs, "ventilation_mm_s"))} mm/s to cool it.`) });
  }
  // humidity held at its limit
  for (const [i, j] of runs(rows, (r) => r.rh_pct >= LIM.rh_max - 0.5, 60 * 60e3, join)) {
    const rs = seg(i, j);
    facts.push({ v: "humidity", t: rows[i].t, x: `${when(i, j)} indoor humidity stayed at the ${LIM.rh_max} % limit (outdoor ${span(minOf(rs, "rh_out_pct"), maxOf(rs, "rh_out_pct"))} %); ` +
      `the vents opened slightly (average ${(rs.reduce((a, r) => a + r.ventilation_mm_s, 0) / rs.length).toFixed(2)} mm/s) to hold it there.` });
  }
  // vents clearly open
  for (const [i, j] of runs(rows, (r) => r.ventilation_mm_s > 0.5, 15 * 60e3, join)) {
    const rs = seg(i, j);
    facts.push({ v: "ventilation", t: rows[i].t, x: `${when(i, j)} the vents were open (up to ${f1(maxOf(rs, "ventilation_mm_s"))} mm/s) while indoor reached ${f1(maxOf(rs, "temp_c"))} °C and ${r0(maxOf(rs, "rh_pct"))} % humidity, and it was ${f1(rs[0].temp_out_c)} °C outdoors.` });
  }
  // CO2 dosing
  for (const [i, j] of runs(rows, (r) => r.co2_supply_mg_m2_s > 0.01, 30 * 60e3, join)) {
    const rs = seg(i, j), g = rs.reduce((a, r) => a + r.co2_supply_mg_m2_s, 0) * dt / 1000;
    facts.push({ v: "co2", t: rows[i].t, x: `${when(i, j)} CO₂ dosing was on (${r0(g)} g/m² added): indoor CO₂ rose to ${r0(maxOf(rs, "co2_ppm"))} ppm while outdoor was about ${r0(rs[0].co2_out_ppm)} ppm. Dosing runs in daylight, when it helps the lettuce grow.` });
  }
  // heating: big rises and falls of the hourly average
  const hourly = [];
  for (const r of rows) {
    const h = Math.floor(r.t / 3600e3);
    if (!hourly.length || hourly[hourly.length - 1].h !== h) hourly.push({ h, t: h * 3600e3, n: 0, H: 0, To: 0, Mn: 0, Sun: 0 });
    const x = hourly[hourly.length - 1]; x.n++; x.H += r.heating_w_m2; x.To += r.temp_out_c; x.Mn += r.limit_temp_min; x.Sun += r.radiation_w_m2;
  }
  hourly.forEach((x) => { x.H /= x.n; x.To /= x.n; x.Mn /= x.n; x.Sun /= x.n; });
  const cause = (s0, s1, up, next) => {
    const dMin = s1.Mn - s0.Mn, dOut = s1.To - s0.To, dSun = s1.Sun - s0.Sun;
    if (up && dMin >= 1) return `the higher daytime minimum (${r0(s1.Mn)} °C) started at sunrise`;
    const ahead = next.find((x) => x.Mn - s1.Mn >= 1);
    if (up && ahead) return `the controller warmed the greenhouse ahead of the higher daytime minimum (${r0(Math.max(...next.map((x) => x.Mn)))} °C) that starts at sunrise`;
    if (!up && dMin <= -1) return `the lower night minimum (${r0(s1.Mn)} °C) started`;
    if (up && dOut <= -1) return "it got colder outdoors";
    if (!up && dOut >= 1) return "it got milder outdoors";
    if (up && dSun <= -50) return "the sun went down";
    if (!up && dSun >= 50) return "the sun warmed the greenhouse";
    return null;
  };
  let a = 0;
  for (let k = 1; k <= hourly.length; k++) {
    const dirPrev = k >= 2 ? Math.sign(hourly[k - 1].H - hourly[k - 2].H) : 0;
    const dirNow = k < hourly.length ? Math.sign(hourly[k].H - hourly[k - 1].H) : 0;
    if (k === hourly.length || (dirNow !== dirPrev && dirPrev !== 0 && k - 1 > a)) {
      const s0 = hourly[a], s1 = hourly[k - 1], ch = s1.H - s0.H;
      if (Math.abs(ch) >= 25) {
        const why = cause(s0, s1, ch > 0, hourly.slice(k, k + 2));
        facts.push({ v: "heating", t: s0.t, x: `Heating ${ch > 0 ? "rose" : "fell"} from ${r0(s0.H)} to ${r0(s1.H)} W/m² between ${tStr(s0.t, wd)} and ${tStr(s1.t + 3600e3, wd)}` +
          (why ? ` because ${why}` : "") + ` (outdoor ${f1(s0.To)} → ${f1(s1.To)} °C).` });
      }
      a = k - 1;
    }
  }
  // sensors
  const gapMs = GAP_MS[R.src];
  for (let k = 1; k < rows.length; k++) {
    const d = rows[k].t - rows[k - 1].t;
    if (d > gapMs) facts.push({ v: "sensors", t: rows[k - 1].t, x: `No readings from ${tStr(rows[k - 1].t, wd)} to ${tStr(rows[k].t, wd)} (${ago(d).replace(" ago", "")}).` });
  }
  for (const [kind, text] of sensorChecks()) if (kind === "bad" || kind === "warn") facts.push({ v: "sensors", t: Date.now(), x: text });
  for (const e of forecastEvents()) facts.push({ v: "forecast", t: e.t, x: `Forecast: ${e.x}` });
  return facts.sort((p, q) => q.t - p.t);
}

function renderFacts() {
  const topic = $("topic").value, all = makeFacts();
  const list = all.filter((f) => topic === "all" || f.v === topic);
  $("facts-note").textContent = `${list.length} fact${list.length === 1 ? "" : "s"} for ${RANGE_LABEL[S.range]}, newest first` +
    (list.length > 150 ? "; showing the first 150." : ".");
  const NAME = { forecast: "forecast", day: "day", temperature: "temperature", humidity: "humidity", heating: "heating", ventilation: "vents", co2: "CO₂", sensors: "sensors" };
  $("facts").innerHTML = list.slice(0, 150).map((f) => `<div class="fact"><b>${NAME[f.v]}</b> · ${esc(f.x)}</div>`).join("") ||
    `<p class="muted">Nothing to report for this topic in this period.</p>`;
}

// ---------------- instant answers from the live data (the Ask tab)
function windowFor(ql) {
  const now = Date.now();
  if (/last hour|past hour|this hour/.test(ql)) return { rows: S.cache.raw.rows.filter((r) => r.t >= now - 3600e3), dt: EVERY_S, label: "in the last hour" };
  if (/today|this morning|this afternoon/.test(ql)) {        // the 5-minute averages always cover the whole day
    const d = fDate.format(now);
    return { rows: S.cache.sum.rows.filter((r) => fDate.format(r.t) === d), dt: 300, label: "today" };
  }
  const m = ql.match(/(?:last|past) (\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|h|days?)\b/) ||
    (/(?:last|past|this) week/.test(ql) ? [null, "7", "days"] : null);
  if (m) {                                                   // the question names its own time window
    const hours = parseFloat(m[1]) * (m[2].startsWith("d") ? 24 : m[2].startsWith("m") ? 1 / 60 : 1);
    const src = hours <= 6 ? "raw" : "sum", since = now - hours * 3600e3, c = S.cache[src];
    const note = c.from !== null && c.from > since + 600e3 ? ` (only ${Math.round((now - c.from) / 3600e3)} h loaded: choose a longer period at the top for more)` : "";
    return { rows: c.rows.filter((r) => r.t >= since), dt: stepS(src), label: `in the last ${m[1]} ${m[2].replace(/^hrs?$|^h$/, "hours")}${note}` };
  }
  let rows = rowsInRange(), label = `in ${RANGE_LABEL[S.range]}`;
  if (/night|tonight/.test(ql)) { rows = rows.filter((r) => r.radiation_w_m2 < 10); label = `at night (${RANGE_LABEL[S.range]})`; }
  return { rows, dt: stepS(RANGES[S.range].src), label };
}

// questions about the future are answered from the controller's forecast
const fHM = new Intl.DateTimeFormat("en-GB", { timeZone: tz, hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
function isFuture(ql) {
  if (/\b(was|were|did|had|yesterday|last)\b/.test(ql)) return false;
  return /\b(will|going to|forecast|predict|expect|expected|later|tonight|this evening|soon|upcoming|plan|plans|planned|tomorrow|next \d|next hour|next few)\b|\bin \d+(\.\d+)?\s*(h|hr|hrs|hour|hours|min|mins|minute|minutes)\b|\bat \d{1,2}(:\d{2})?\s*(am|pm)?\b/.test(ql);
}
function targetTime(ql) {
  const now = Date.now();
  let m = ql.match(/\bin (\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|min|mins|minute|minutes)\b/);
  if (m) return now + parseFloat(m[1]) * (m[2].startsWith("h") ? 3600e3 : 60e3);
  m = ql.match(/\bat (\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b/);
  if (m) {
    let h = parseInt(m[1], 10) % 24; const mi = parseInt(m[2] || "0", 10);
    if (m[3] === "pm" && h < 12) h += 12;
    if (m[3] === "am" && h === 12) h = 0;
    const [nh, nm] = fHM.format(now).split(":").map(Number);
    let diff = h * 60 + mi - (nh * 60 + nm);
    if (diff < 0 && !m[3] && h < 12 && diff + 720 > 0) diff += 720;   // "at 4" in the afternoon means 16:00
    if (diff < 0) diff += 1440;
    return now + diff * 60e3;
  }
  if (/tonight|this evening/.test(ql)) return "night";
  return null;
}

function futureAnswer(ql) {
  const F = S.fc.rows, L = S.latest;
  if (!F.length) return "There is no forecast yet. The controller makes a new 6-hour forecast every 15 minutes while the sensor runs.";
  const end = F[F.length - 1].t;
  const made = ` (forecast by the controller, made at ${fTime.format(fcMade())}${fcOld() ? "; it is old, the sensor may have stopped" : ""}; it assumes the weather forecast is right)`;
  if (/tomorrow|next week|next day/.test(ql)) return `The forecast only covers the next 6 hours, until ${fTime.format(end)}.`;
  const tgt = targetTime(ql);
  if (typeof tgt === "number" && tgt > end + 8 * 60e3) return `The forecast only reaches ${fTime.format(end)} (6 hours ahead). Ask about a time before then.`;
  const at = typeof tgt === "number" ? F.reduce((a, r) => (Math.abs(r.t - tgt) < Math.abs(a.t - tgt) ? r : a)) : null;
  const night = F.filter((r) => r.radiation_w_m2 < 10), rows = tgt === "night" && night.length ? night : F;
  const when = at ? `At ${fTime.format(at.t)}` : tgt === "night" ? `Tonight (until ${fTime.format(end)})` : `In the next 6 hours (until ${fTime.format(end)})`;
  const has = (...w) => w.some((x) => ql.includes(x));
  const ext = (k) => [Math.min(...rows.map((r) => r[k])), Math.max(...rows.map((r) => r[k]))];
  const hiR = (k) => rows.reduce((a, r) => (r[k] > a[k] ? r : a)), loR = (k) => rows.reduce((a, r) => (r[k] < a[k] ? r : a));
  if (has("heat")) {
    if (at) return `${when} the controller plans heating at ${r0(at.heating_w_m2)} W/m² (now ${r0(L.heating_w_m2)}) ${why(at, L)}`.trim() + "." + made;
    const hi = hiR("heating_w_m2"), lo = loR("heating_w_m2");
    if (hi.heating_w_m2 - L.heating_w_m2 >= 15) return `Yes — heating will rise from ${r0(L.heating_w_m2)} W/m² now to about ${r0(hi.heating_w_m2)} W/m² by ${fTime.format(hi.t)} ${why(hi, L)}`.trim() + "." + made;
    if (L.heating_w_m2 - lo.heating_w_m2 >= 15) return `No — heating will fall from ${r0(L.heating_w_m2)} W/m² now to about ${r0(lo.heating_w_m2)} W/m² by ${fTime.format(lo.t)} ${why(lo, L)}`.trim() + "." + made;
    return `${when} heating stays about ${span(lo.heating_w_m2, hi.heating_w_m2)} W/m² (now ${r0(L.heating_w_m2)}).` + made;
  }
  if (has("vent", "window", "open")) {
    if (at) return `${when} the vents should be at ${f1(at.ventilation_mm_s)} mm/s (indoor ${f1(at.temp_c)} °C, ${r0(at.rh_pct)} % humidity).` + made;
    const v = runs(rows, (r) => r.ventilation_mm_s > 0.5, 0, 16 * 60e3)[0];
    if (v) return `Yes — the controller plans to open the vents more (up to ${f1(Math.max(...rows.slice(v[0], v[1] + 1).map((r) => r.ventilation_mm_s)))} mm/s) from about ${fTime.format(rows[v[0]].t)} to ${fTime.format(rows[v[1]].t)}, when indoor reaches ${f1(Math.max(...rows.slice(v[0], v[1] + 1).map((r) => r.temp_c)))} °C.` + made;
    return `No — no big vent openings are planned ${when.toLowerCase()}; they stay at about ${span(...ext("ventilation_mm_s"), 2)} mm/s to hold humidity at ${LIM.rh_max} %.` + made;
  }
  if (has("humid", "moist", "wet")) {
    if (at) return `${when} indoor humidity should be ${r0(at.rh_pct)} % (limit ${LIM.rh_max} %) with ${r0(at.rh_out_pct)} % outdoors.` + made;
    return `${when} indoor humidity should stay at ${span(...ext("rh_pct"))} % (limit ${LIM.rh_max} %) while outdoor humidity is ${span(...ext("rh_out_pct"))} %.` + made;
  }
  if (has("co2", "co₂", "carbon")) {
    const dosing = (r) => r.co2_supply_mg_m2_s > 0.01, sw = rows.find((r) => dosing(r) !== dosing(L));
    if (at) return `${when} indoor CO₂ should be ${r0(at.co2_ppm)} ppm with dosing ${dosing(at) ? "on" : "off"} (outdoor ${r0(at.co2_out_ppm)} ppm).` + made;
    return `${when} indoor CO₂ should be ${span(...ext("co2_ppm"))} ppm` + (sw ? `; dosing will ${dosing(sw) ? "start" : "stop"} around ${fTime.format(sw.t)}.` : ".") + made;
  }
  if (has("grow", "harvest", "yield", "weight", "lettuce")) return "The forecast covers the climate and the equipment, not lettuce growth. Ask about temperature, humidity, CO₂, heating or vents.";
  // temperature or general
  if (at) return `${when} the greenhouse should be ${f1(at.temp_c)} °C (allowed ${r0(at.limit_temp_min)}–${r0(at.limit_temp_max)} °C) with ${f1(at.temp_out_c)} °C outdoors and heating at ${r0(at.heating_w_m2)} W/m².` + made;
  const lim = rows.find((r) => Math.abs(r.limit_temp_min - L.limit_temp_min) > 0.5);
  const cold = rows.find((r) => r.temp_c < r.limit_temp_min - 0.3), warm = rows.find((r) => r.temp_c > r.limit_temp_max + 0.3);
  return `${when} the greenhouse should stay at ${span(...ext("temp_c"), 1)} °C while it is ${span(...ext("temp_out_c"), 1)} °C outdoors` +
    (lim ? `; the ${lim.limit_temp_min > L.limit_temp_min ? "daytime" : "night"} limits (${r0(lim.limit_temp_min)}–${r0(lim.limit_temp_max)} °C) start around ${fTime.format(lim.t)}` : "") +
    (cold ? `; it may dip below the minimum around ${fTime.format(cold.t)} (${f1(cold.temp_c)} °C)` : "") +
    (warm ? `; it may go above the maximum around ${fTime.format(warm.t)} (${f1(warm.temp_c)} °C)` : "") + "." + made;
}

function liveAnswer(q) {
  const ql = q.toLowerCase() + " ", L = S.latest;
  if (!L) return "There are no readings yet. Start the sensor box (or the pretend sensor), then ask again.";
  if (isFuture(ql)) return futureAnswer(ql);
  const has = (...w) => w.some((x) => ql.includes(x));
  const nowAsked = has(" now", "right now", "current", "at the moment", "is it ");
  const { rows, dt, label } = windowFor(ql), s = rows.length ? stats(rows, dt) : null;
  const age = Date.now() - L.t, stale = age > 15 * 60e3 ? ` (note: the last reading is ${ago(age)})` : "";
  const nowT = `${f1(L.temp_c)} °C indoors (allowed ${r0(L.limit_temp_min)}–${r0(L.limit_temp_max)} °C) and ${f1(L.temp_out_c)} °C outdoors`;
  if (has("sensor", "working", "data", "reading", "broken", "stuck")) {
    return sensorChecks().map(([, t]) => t).join(" ") + stale;
  }
  if (has("vent", "window", "open", "cool")) {
    if (L.temp_c > L.limit_temp_max + 0.3) return `Yes — it is ${nowT}, above the limit; the vents are at ${f1(L.ventilation_mm_s)} mm/s to cool it${L.temp_out_c > L.limit_temp_max ? ", but it is warmer outdoors than the limit, so vents can only help a little" : ""}.${stale}`;
    return `No need now — it is ${nowT}; the vents are only at ${f1(L.ventilation_mm_s)} mm/s, mainly to keep humidity at the ${LIM.rh_max} % limit` +
      (s && s.warmH > 0 ? `. ${label[0].toUpperCase() + label.slice(1)} it was too warm for ${f1(s.warmH)} h and the vents were open ${f1(s.ventH)} h.` : ".") + stale;
  }
  if (has("heat")) {
    const why = L.temp_out_c < L.limit_temp_min ? ` because it is ${f1(L.temp_out_c)} °C outdoors and the greenhouse must stay above ${r0(L.limit_temp_min)} °C` : "";
    return `Heating is ${r0(L.heating_w_m2)} W/m² now${why}.` + (s ? ` ${label[0].toUpperCase() + label.slice(1)} it used ${s.kwh.toFixed(2)} kWh/m² (about ${(s.kwh * PRICES.heat_cad_per_kwh).toFixed(2)} CAD/m²), averaging ${r0(s.heatAvg)} W/m² while it was ${span(...s.To)} °C outdoors.` : "") + stale;
  }
  if (has("humid", "moist", "wet")) {
    return `Indoor humidity is ${r0(L.rh_pct)} % now (limit ${LIM.rh_max} %), outdoor ${r0(L.rh_out_pct)} %.` +
      (s ? ` ${label[0].toUpperCase() + label.slice(1)} indoor humidity was ${span(...s.RH)} % and sat at the limit for ${f1(s.humidH)} h; the controller holds it there with small vent openings.` : "") + stale;
  }
  if (has("co2", "co₂", "carbon")) {
    return `Indoor CO₂ is ${r0(L.co2_ppm)} ppm now (outdoor ${r0(L.co2_out_ppm)} ppm); dosing is ${L.co2_supply_mg_m2_s > 0.001 ? "on" : "off"}.` +
      (s ? ` ${label[0].toUpperCase() + label.slice(1)} ${r0(s.co2g)} g/m² of CO₂ was added, in daylight when it helps the lettuce grow.` : "") + stale;
  }
  if (has("grow", "harvest", "yield", "weight", "lettuce", "crop")) {
    return `The lettuce (crop ${L.crop_cycle}) weighs ${r0(L.dry_weight_g_m2)} g dry weight/m² now` + (s && s.growth !== null ? `; it grew ${f1(s.growth)} g/m² ${label}.` : ".") + stale;
  }
  if (has("cost", "money", "price", "energy", "gas")) {
    return s ? `${label[0].toUpperCase() + label.slice(1)} heating cost about ${(s.kwh * PRICES.heat_cad_per_kwh).toFixed(2)} CAD/m² (${s.kwh.toFixed(2)} kWh/m²) and CO₂ ${(s.co2g / 1000 * PRICES.co2_cad_per_kg).toFixed(3)} CAD/m² (assumed prices).` : "No readings in this period.";
  }
  // temperature / "is it okay" / anything else
  const inRange = L.temp_c >= L.limit_temp_min - 0.3 && L.temp_c <= L.limit_temp_max + 0.3;
  let txt = inRange ? `Yes — it is ${nowT}, within the allowed range.` :
    L.temp_c < L.limit_temp_min ? `Not right now — it is ${nowT}: below the minimum; heating is at ${r0(L.heating_w_m2)} W/m² to bring it up.` :
      `Not right now — it is ${nowT}: above the maximum; the vents are at ${f1(L.ventilation_mm_s)} mm/s.`;
  if (s && !nowAsked) txt += (s.coldH || s.warmH) ? ` ${label[0].toUpperCase() + label.slice(1)} it was too cold for ${f1(s.coldH)} h and too warm for ${f1(s.warmH)} h (indoor ${span(...s.T)} °C).` :
    ` ${label[0].toUpperCase() + label.slice(1)} it stayed within its limits (indoor ${span(...s.T)} °C, outdoor ${span(...s.To)} °C).`;
  return txt + stale;
}

function liveContext(q) {
  const ql = q.toLowerCase(), L = S.latest;
  const lines = [`Quick answer from the data: ${liveAnswer(q)}`];
  if (L) lines.push(`Now (${fFull.format(L.t)}): indoor ${f1(L.temp_c)} °C, ${r0(L.rh_pct)} % humidity, ${r0(L.co2_ppm)} ppm CO₂; outdoor ${f1(L.temp_out_c)} °C, ${r0(L.rh_out_pct)} %, ${r0(L.co2_out_ppm)} ppm; heating ${r0(L.heating_w_m2)} W/m², vents ${f1(L.ventilation_mm_s)} mm/s.`);
  const topics = { heating: ["heat"], temperature: ["temperat", "warm", "cold", "hot", "okay"], humidity: ["humid"], ventilation: ["vent", "window", "open"], co2: ["co2", "co₂"] };
  const named = Object.keys(topics).filter((k) => topics[k].some((w) => ql.includes(w)));
  if (isFuture(ql + " ") && S.fc.rows.length) {
    lines.push(`Controller forecast made at ${fTime.format(fcMade())} (it assumes the weather forecast is right):`);
    forecastEvents().forEach((e) => lines.push(`- ${e.x}`));
  } else {
    makeFacts().filter((f) => f.v !== "day" && f.v !== "forecast" && (!named.length || named.includes(f.v))).slice(0, 4).forEach((f) => lines.push(`- ${f.x}`));
  }
  return lines.join("\n");
}

function renderExamples() {
  if (!S.ai) return;
  S.ai.setExamples(["Is the temperature okay now?", "Will heating go up tonight?", "What will the temperature be in 2 hours?",
    "Will the vents open later?", "Why is heating high today?", `How was humidity in ${RANGE_LABEL[S.range]}?`]);
}

// ---------------- start
function boot() {
  if (!CFG.url || !CFG.key) { show("off"); return; }
  $("signin-form").addEventListener("submit", signIn);
  $("signout").addEventListener("click", () => signOut());
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("topic").addEventListener("change", renderFacts);
  if (window.GreenhouseAI) {
    S.ai = GreenhouseAI.setup({ answer: liveAnswer, context: liveContext,
      dataNote: CFG.source === "pretend" ? "The readings come from a pretend sensor that replays a greenhouse simulation." : "The readings come from live greenhouse sensors." });
    renderExamples();
  }
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    S.range = b.dataset.range;
    document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderExamples();
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
