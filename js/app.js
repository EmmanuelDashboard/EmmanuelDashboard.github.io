/*
 * Emmanuel Dashboard for Pecan Thinning
 * Opens the results ZIP of the pecan canopy notebook, shows crowns (EVEN / ODD),
 * the wavy thinning diagonals, the canopy total of every diagonal and the
 * thinning start, and lets the user correct crowns. Runs fully in the browser.
 */
(() => {
  "use strict";

  // ------------------------------------------------------------------ config
  const COLORS = { EVEN: "#1e9be9", ODD: "#ef4b3f" };
  const LINE_COLORS = { EVEN: "#0a4f9e", ODD: "#b01e16" };
  const EXCLUDED_COLOR = "#8a9099";
  const SELECT_COLOR = "#ffd400";
  const STORAGE_PREFIX = "pecan-thinning-dashboard:v1:";
  const CROWN_SIMPLIFY_M = 0.12;   // ~1 pixel of the orthomosaic
  const LINE_SIMPLIFY_M = 0.04;

  // ------------------------------------------------------------------ state
  const state = {
    loaded: false,
    name: "",
    key: "",
    summary: null,
    proj: null,
    origin: [0, 0],
    crowns: new Map(),
    diagonals: new Map(),
    axis: null,
    start: null,
    startOnly: false,
    selected: null,
    selectedDiagonal: null,
    mode: "select",
    mergeFrom: null,
    history: [],
    future: [],
    fillOpacity: 0.42,
    fitZoom: 0,
    addedCount: 0,
    totals: null,
  };

  // ------------------------------------------------------------------ helpers
  const $ = (id) => document.getElementById(id);
  const family = (d) => (Math.abs(d) % 2 === 1 ? "ODD" : "EVEN");
  const dname = (d) => "D" + String(d).padStart(2, "0");
  const fmtInt = (n) => Math.round(n).toLocaleString("en-US");
  const fmtArea = (a) => fmtInt(a) + " m²";
  const fmt1 = (n) => (Math.round(n * 10) / 10).toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

  function toast(message, kind = "", ms = 3200) {
    const el = $("toast");
    el.textContent = message;
    el.className = "toast" + (kind ? " " + kind : "");
    el.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => { el.hidden = true; }, ms);
  }

  function setLoading(text) {
    $("loading").hidden = !text;
    if (text) $("loading-text").textContent = text;
  }

  const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));

  // Save a file: through the host's save prompt where the page runs inside
  // claude.ai (window.claude), otherwise as a normal browser download.
  async function download(name, text, type = "text/csv") {
    const blob = text instanceof Blob ? text : new Blob([text], { type: type + ";charset=utf-8" });
    if (window.claude && typeof window.claude.use === "function") {
      let saver = null;
      try { saver = await window.claude.use("downloads"); } catch (e) { saver = null; }
      if (!saver) { toast("Saving files is not available in this view.", "error"); return; }
      try {
        await saver.save({ filename: name.replace(/\.geojson$/i, ".json"), data: blob });
        toast("Saved " + name.replace(/\.geojson$/i, ".json"));
      } catch (err) {
        const code = err && err.code;
        if (code === "declined") return;
        toast(code === "rate_limited" ? "A save prompt is already open." : "The file could not be saved here.", "error");
      }
      return;
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  function csvCell(v) {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }
  const csvRow = (cells) => cells.map(csvCell).join(",");

  function storageGet(key) {
    try { return JSON.parse(localStorage.getItem(key) || "null"); } catch (e) { return null; }
  }
  function storageSet(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch (e) { return false; }
  }
  function storageRemove(key) {
    try { localStorage.removeItem(key); } catch (e) { /* ignore */ }
  }

  // ------------------------------------------------------------------ geometry
  function segDist2(p, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    let t = 0;
    const len2 = dx * dx + dy * dy;
    if (len2 > 0) t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
    const x = a[0] + t * dx - p[0], y = a[1] + t * dy - p[1];
    return x * x + y * y;
  }

  // Douglas-Peucker on an open or closed list of points.
  function simplify(points, tol) {
    const n = points.length;
    if (n <= 4) return points;
    const keep = new Uint8Array(n);
    keep[0] = keep[n - 1] = 1;
    const stack = [[0, n - 1]];
    const tol2 = tol * tol;
    while (stack.length) {
      const [a, b] = stack.pop();
      let max = 0, idx = -1;
      for (let i = a + 1; i < b; i++) {
        const d = segDist2(points[i], points[a], points[b]);
        if (d > max) { max = d; idx = i; }
      }
      if (max > tol2 && idx > 0) {
        keep[idx] = 1;
        stack.push([a, idx], [idx, b]);
      }
    }
    const out = [];
    for (let i = 0; i < n; i++) if (keep[i]) out.push(points[i]);
    return out.length >= 4 || out.length === n ? out : points;
  }

  function ringArea(ring) {
    let s = 0;
    for (let i = 0, n = ring.length; i < n; i++) {
      const [x1, y1] = ring[i];
      const [x2, y2] = ring[(i + 1) % n];
      s += x1 * y2 - x2 * y1;
    }
    return Math.abs(s) / 2;
  }

  function polysArea(polys) {
    let a = 0;
    for (const poly of polys) {
      a += ringArea(poly[0]);
      for (let h = 1; h < poly.length; h++) a -= ringArea(poly[h]);
    }
    return a;
  }

  function polysCentroid(polys) {
    let sx = 0, sy = 0, sa = 0;
    for (const poly of polys) {
      const ring = poly[0];
      let a = 0, cx = 0, cy = 0;
      for (let i = 0, n = ring.length; i < n; i++) {
        const [x1, y1] = ring[i];
        const [x2, y2] = ring[(i + 1) % n];
        const f = x1 * y2 - x2 * y1;
        a += f; cx += (x1 + x2) * f; cy += (y1 + y2) * f;
      }
      if (a) { sx += cx / 3; sy += cy / 3; sa += a; }
    }
    if (!sa) { const r = polys[0][0]; return r[0]; }
    return [sx / sa, sy / sa];
  }

  function polysBBox(polys) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const poly of polys) for (const [x, y] of poly[0]) {
      if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y;
    }
    return [x0, y0, x1, y1];
  }

  const bboxHit = (a, b) => a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3];

  function pathLength(points) {
    let s = 0;
    for (let i = 1; i < points.length; i++) s += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
    return s;
  }

  const toLL = ([x, y]) => [y, x];
  const polysToLL = (polys) => polys.map((poly) => poly.map((ring) => ring.map(toLL)));

  // ------------------------------------------------------------------ projection
  function makeProjection(crsText, sampleLonLat) {
    let def = null;
    const m = /EPSG:(\d+)/i.exec(crsText || "");
    if (m) {
      const code = +m[1];
      if (code >= 32601 && code <= 32660) def = `+proj=utm +zone=${code - 32600} +datum=WGS84 +units=m +no_defs`;
      else if (code >= 32701 && code <= 32760) def = `+proj=utm +zone=${code - 32700} +south +datum=WGS84 +units=m +no_defs`;
      else if (code >= 26901 && code <= 26923) def = `+proj=utm +zone=${code - 26900} +datum=NAD83 +units=m +no_defs`;
      else if (code >= 6330 && code <= 6348) def = `+proj=utm +zone=${code - 6329} +ellps=GRS80 +units=m +no_defs`;
    }
    const exact = !!def;
    if (!def) {
      const [lon, lat] = sampleLonLat;
      const zone = Math.floor((lon + 180) / 6) + 1;
      def = `+proj=utm +zone=${zone}${lat < 0 ? " +south" : ""} +datum=WGS84 +units=m +no_defs`;
    }
    const p = proj4("EPSG:4326", def);
    return { forward: (ll) => p.forward(ll), inverse: (xy) => p.inverse(xy), def, exact };
  }

  // ------------------------------------------------------------------ map
  let map, renderers, layers, chart;

  function initMap() {
    map = L.map("map", {
      crs: L.CRS.Simple,
      minZoom: -6,
      maxZoom: 6,
      zoomSnap: 0.25,
      zoomDelta: 0.5,
      wheelPxPerZoomLevel: 90,
      attributionControl: false,
      preferCanvas: true,
      zoomControl: true,
    });
    map.setView([0, 0], 0);

    // Crowns and lines share ONE canvas, so a click reaches whatever is
    // under the cursor (a canvas on top would swallow clicks meant for the
    // one below). Start and selection overlays never take clicks.
    const panes = [["image", 300], ["main", 410], ["start", 430], ["select", 440]];
    for (const [name, z] of panes) map.createPane(name).style.zIndex = z;
    for (const name of ["image", "start", "select"]) map.getPane(name).style.pointerEvents = "none";

    const main = L.canvas({ pane: "main", padding: 0.4, tolerance: 4 });
    renderers = {
      crowns: main,
      lines: main,
      start: L.canvas({ pane: "start", padding: 0.4 }),
      select: L.canvas({ pane: "select", padding: 0.4 }),
    };

    layers = {
      image: L.layerGroup(),
      crowns: L.layerGroup(),
      lines: { EVEN: L.layerGroup(), ODD: L.layerGroup() },
      arrows: { EVEN: L.layerGroup(), ODD: L.layerGroup() },
      labels: L.layerGroup(),
      start: L.layerGroup(),
      select: L.layerGroup(),
      draw: L.layerGroup(),
    };
    for (const key of ["image", "crowns", "labels", "start", "select", "draw"]) layers[key].addTo(map);
    for (const f of ["EVEN", "ODD"]) { layers.lines[f].addTo(map); layers.arrows[f].addTo(map); }

    L.control.scale({ imperial: false, position: "bottomleft", maxWidth: 160 }).addTo(map);

    const North = L.Control.extend({
      options: { position: "topright" },
      onAdd() {
        const div = L.DomUtil.create("div", "north-arrow");
        div.innerHTML = '<svg viewBox="0 0 16 18" aria-hidden="true"><path d="M8 1 14 16 8 12 2 16Z" fill="#1d2420"/></svg>N';
        div.title = "North (grid north of the field's UTM zone)";
        return div;
      },
    });
    new North().addTo(map);

    const Fit = L.Control.extend({
      options: { position: "topleft" },
      onAdd() {
        const div = L.DomUtil.create("div", "leaflet-bar");
        const a = L.DomUtil.create("a", "", div);
        a.href = "#";
        a.title = "Show the whole field";
        a.setAttribute("role", "button");
        a.setAttribute("aria-label", "Show the whole field");
        a.innerHTML = "&#x2922;";
        a.style.fontSize = "16px";
        L.DomEvent.on(a, "click", (e) => { L.DomEvent.preventDefault(e); fitField(); });
        L.DomEvent.disableClickPropagation(div);
        return div;
      },
    });
    new Fit().addTo(map);

    const Coords = L.Control.extend({
      options: { position: "bottomright" },
      onAdd() {
        const div = L.DomUtil.create("div", "coords");
        div.id = "coords";
        div.textContent = "";
        return div;
      },
    });
    new Coords().addTo(map);

    map.on("mousemove", (e) => {
      if (!state.loaded) return;
      const ll = state.toLonLat ? state.toLonLat(e.latlng.lng, e.latlng.lat) : null;
      $("coords").textContent = ll
        ? `${ll[1].toFixed(6)}, ${ll[0].toFixed(6)}`
        : `${e.latlng.lng.toFixed(1)} m, ${e.latlng.lat.toFixed(1)} m`;
    });
    map.on("zoomend", updateLabelVisibility);
    map.on("click", onMapClick);
    map.on("dblclick", (e) => { if (state.mode === "draw") { L.DomEvent.stop(e); finishDraw(); } });
  }

  function fitField() {
    if (!state.loaded) return;
    map.fitBounds(state.bounds, { padding: [24, 24] });
  }

  // ------------------------------------------------------------------ loading
  function basename(path) {
    return path.split(/[\\/]/).pop().toLowerCase();
  }

  const IMAGE_FILE = /\.(tif|tiff|jpg|jpeg|png|webp)$/i;

  async function openFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    try {
      const zip = files.find((f) => /\.zip$/i.test(f.name));
      const notebookFiles = files.some((f) => basename(f.name) === "crowns_wgs84.geojson");
      const images = files.filter((f) => IMAGE_FILE.test(f.name));
      if (zip) {
        await loadZip(zip, zip.name);
      } else if (images.length && !notebookFiles) {
        await analyseImage(images);
      } else {
        const byName = new Map(files.map((f) => [basename(f.name), f]));
        await loadEntries({
          name: files[0].name,
          text: async (n) => (byName.has(n) ? byName.get(n).text() : null),
          blob: async (n) => (byName.has(n) ? byName.get(n) : null),
        });
      }
    } catch (err) {
      console.error(err);
      setLoading(null);
      toast(err.message || String(err), "error", 7000);
    }
  }

  // ------------------------------------------------------------------ analyse an image here
  const UTM_DEF = (code) => {
    if (code >= 32601 && code <= 32660) return `+proj=utm +zone=${code - 32600} +datum=WGS84 +units=m +no_defs`;
    if (code >= 32701 && code <= 32760) return `+proj=utm +zone=${code - 32700} +south +datum=WGS84 +units=m +no_defs`;
    if (code >= 26901 && code <= 26923) return `+proj=utm +zone=${code - 26900} +datum=NAD83 +units=m +no_defs`;
    return null;
  };

  async function inspectImage(file) {
    const info = { file, name: file.name.replace(/\.[^.]+$/, ""), geo: null };
    if (/\.tiff?$/i.test(file.name)) {
      const tiff = await GeoTIFF.fromBlob(file);
      const image = await tiff.getImage(0);
      const fd = image.fileDirectory;
      Object.assign(info, {
        kind: "tiff", tiff, image,
        width: image.getWidth(), height: image.getHeight(),
        spp: image.getSamplesPerPixel(),
        bits: Number((fd.BitsPerSample && fd.BitsPerSample[0]) || fd.BitsPerSample || 8),
        extra: fd.ExtraSamples ? Array.from(fd.ExtraSamples) : [],
        nodata: typeof image.getGDALNoData === "function" ? image.getGDALNoData() : null,
      });
      if (info.spp < 3) throw new Error("This TIFF has only " + info.spp + " band(s). An RGB drone image (orthomosaic) is needed, not an elevation file.");
      info.geo = georeference(image, info.width, info.height);
    } else {
      const bitmap = await createImageBitmap(file);
      Object.assign(info, { kind: "picture", bitmap, width: bitmap.width, height: bitmap.height });
    }
    return info;
  }

  // Ground size of the pixels (metres) and how to get longitude / latitude.
  function georeference(image, width, height) {
    let keys, res, origin;
    try { keys = image.getGeoKeys(); res = image.getResolution(); origin = image.getOrigin(); } catch (e) { return null; }
    if (!keys || !res || !origin || !res[0]) return null;
    const rx = Math.abs(res[0]), ry = Math.abs(res[1]);
    const minx = origin[0], maxy = origin[1], miny = maxy - ry * height;
    const pcs = keys.ProjectedCSTypeGeoKey, gcs = keys.GeographicTypeGeoKey;
    // Raw grid of the image (the shared analysis samples the DSM with it).
    const raw = { left: origin[0], top: origin[1], resX: res[0], resY: res[1] };
    if (keys.GTModelTypeGeoKey === 2 || (!pcs && gcs)) {
      const { px, py } = PecanAnalysis.groundPixelSize({ kind: "geographic", rx, ry, miny, maxy });
      const mx = px / rx, my = py / ry;
      return { crs: `EPSG:${gcs || 4326} (latitude / longitude)`, px, py, raw, origin: [0, 0], toLonLat: (x, y) => [minx + x / mx, miny + y / my] };
    }
    if ([3857, 3785, 900913, 102100, 102113].includes(pcs)) {
      const R = 6378137;
      const { px, py } = PecanAnalysis.groundPixelSize({ kind: "webmercator", rx, ry, miny, maxy });
      const k = px / rx;
      return {
        crs: "EPSG:3857 (Web Mercator)", px, py, raw, origin: [0, 0],
        toLonLat: (x, y) => { const X = minx + x / k, Y = miny + y / k; return [(X / R) * 180 / Math.PI, (2 * Math.atan(Math.exp(Y / R)) - Math.PI / 2) * 180 / Math.PI]; },
      };
    }
    if (pcs && pcs !== 32767) {
      const unit = keys.ProjLinearUnitsGeoKey;
      const f = unit === 9002 ? 0.3048 : unit === 9003 ? 1200 / 3937 : 1;
      const def = UTM_DEF(pcs);
      const p = def ? proj4("EPSG:4326", def) : null;
      const { px, py } = PecanAnalysis.groundPixelSize({ kind: "projected", rx, ry, miny, maxy, unitFactor: f });
      return {
        crs: `EPSG:${pcs}`, px, py, raw,
        origin: p ? [minx, miny] : [0, 0],
        toLonLat: p ? (x, y) => p.inverse([minx + x, miny + y]) : null,
      };
    }
    return null;
  }

  // 'elevation' for a one-band GeoTIFF (DSM / DEM), otherwise 'image'.
  async function fileKind(file) {
    if (!/\.tiff?$/i.test(file.name)) return "image";
    try {
      const tiff = await GeoTIFF.fromBlob(file);
      const image = await tiff.getImage(0);
      return image.getSamplesPerPixel() < 3 ? "elevation" : "image";
    } catch (e) {
      return "image";
    }
  }

  // EPSG code of a GeoTIFF's coordinates, and a proj4 definition for it.
  function epsgOf(keys) {
    if (!keys) return null;
    if (keys.ProjectedCSTypeGeoKey && keys.ProjectedCSTypeGeoKey !== 32767) return keys.ProjectedCSTypeGeoKey;
    return keys.GeographicTypeGeoKey || null;
  }
  function projDef(code) {
    if ([4326, 4269, 4258].includes(code)) return "EPSG:4326";
    if ([3857, 3785, 900913, 102100, 102113].includes(code)) return "EPSG:3857";
    return UTM_DEF(code);
  }

  // The elevation raster (DSM) on the analysis grid of the image: W x H
  // pixels over the image's extent, NaN where there is no elevation.
  async function readSurface(file, info, W, H) {
    const tiff = await GeoTIFF.fromBlob(file);
    const image = await tiff.getImage(0);
    if (image.getSamplesPerPixel() !== 1) throw new Error("it has more than one band (an elevation file has one)");
    const rKeys = info.image.getGeoKeys(), dKeys = image.getGeoKeys();
    const rRes = info.image.getResolution(), rOrig = info.image.getOrigin();
    const dRes = image.getResolution(), dOrig = image.getOrigin();
    const dW = image.getWidth(), dH = image.getHeight();
    const rCode = epsgOf(rKeys), dCode = epsgOf(dKeys);
    let toDsm = null;
    if (rCode && dCode && rCode !== dCode) {
      const a = projDef(rCode), b = projDef(dCode);
      if (!a || !b) throw new Error(`it uses another coordinate system (EPSG:${dCode}) than the image (EPSG:${rCode}); export both in the same one`);
      const p = proj4(a, b);
      toDsm = (x, y) => p.forward([x, y]);
    }
    if (!toDsm) {
      // Same coordinate system: full-resolution DSM sampled by the shared
      // code (identical to the notebook).
      const full = (await image.readRasters({ samples: [0] }))[0];
      const nodataFull = typeof image.getGDALNoData === "function" ? image.getGDALNoData() : null;
      const { surface, inside } = PecanAnalysis.sampleSurface(
        { data: full, width: dW, height: dH, left: dOrig[0], top: dOrig[1], resX: dRes[0], resY: dRes[1], nodata: nodataFull },
        { left: rOrig[0], top: rOrig[1], resX: rRes[0], resY: rRes[1] },
        info.width, info.height, W, H,
      );
      if (inside < 0.2 * W * H) throw new Error("it does not cover this image (another field?)");
      return surface;
    }
    // Image-grid pixel centre -> image coordinates.
    const sx = (info.width * rRes[0]) / W, sy = (info.height * rRes[1]) / H;
    const X = (x) => rOrig[0] + (x + 0.5) * sx, Y = (y) => rOrig[1] + (y + 0.5) * sy;

    // Read the DSM at about the analysis resolution.
    let scale = 1;
    if (!toDsm) scale = Math.min(1, Math.abs(dRes[0]) / Math.abs(sx));
    else scale = Math.min(1, (2 * W) / dW);
    let rw = Math.max(16, Math.round(dW * scale)), rh = Math.max(16, Math.round(dH * scale));
    if (rw * rh > 24e6) { const k = Math.sqrt(24e6 / (rw * rh)); rw = Math.round(rw * k); rh = Math.round(rh * k); }
    const data = (await image.readRasters({ samples: [0], width: rw, height: rh }))[0];
    const nodataRaw = typeof image.getGDALNoData === "function" ? image.getGDALNoData() : null;
    const nodata = nodataRaw === null || nodataRaw === undefined ? null : Number(nodataRaw);
    const cx = (dW * dRes[0]) / rw, cy = (dH * dRes[1]) / rh;
    const valueAt = (px, py) => {
      const c = Math.floor((px - dOrig[0]) / cx), r = Math.floor((py - dOrig[1]) / cy);
      if (c < 0 || r < 0 || c >= rw || r >= rh) return NaN;
      const v = data[r * rw + c];
      return !Number.isFinite(v) || (nodata !== null && v === nodata) || v < -1e5 || v > 1e5 ? NaN : v;
    };

    const surface = new Float32Array(W * H);
    let inside = 0;
    if (!toDsm) {
      for (let y = 0; y < H; y++) {
        const py = Y(y);
        for (let x = 0; x < W; x++) {
          const v = valueAt(X(x), py);
          surface[y * W + x] = v;
          if (Number.isFinite(v)) inside++;
        }
      }
    } else {
      // Another coordinate system: transform a coarse grid, interpolate.
      const k = 16, gw = Math.ceil(W / k) + 1, gh = Math.ceil(H / k) + 1;
      const gx = new Float64Array(gw * gh), gy = new Float64Array(gw * gh);
      for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
        const [u, v] = toDsm(X(Math.min(W - 1, i * k)), Y(Math.min(H - 1, j * k)));
        gx[j * gw + i] = u; gy[j * gw + i] = v;
      }
      for (let y = 0; y < H; y++) {
        const fj = Math.min(gh - 1.001, y / k), j0 = Math.floor(fj), tj = fj - j0;
        for (let x = 0; x < W; x++) {
          const fi = Math.min(gw - 1.001, x / k), i0 = Math.floor(fi), ti = fi - i0;
          const a00 = j0 * gw + i0, a01 = a00 + 1, a10 = a00 + gw, a11 = a10 + 1;
          const u = (1 - tj) * ((1 - ti) * gx[a00] + ti * gx[a01]) + tj * ((1 - ti) * gx[a10] + ti * gx[a11]);
          const v = (1 - tj) * ((1 - ti) * gy[a00] + ti * gy[a01]) + tj * ((1 - ti) * gy[a10] + ti * gy[a11]);
          const z = valueAt(u, v);
          surface[y * W + x] = z;
          if (Number.isFinite(z)) inside++;
        }
      }
    }
    if (inside < 0.2 * W * H) throw new Error("it does not cover this image (another field?)");
    return surface;
  }

  function askAnalysis(info, dsmGiven) {
    return new Promise((resolve) => {
      const modal = $("analyse-modal");
      let dsmFile = info.geo ? dsmGiven || null : null;
      const showDsm = () => {
        $("an-dsm-name").textContent = dsmFile ? dsmFile.name : "None: image only";
        $("an-dsm-clear").hidden = !dsmFile;
      };
      $("an-dsm-row").hidden = !info.geo;
      $("an-dsm-pick").onclick = () => $("an-dsm").click();
      $("an-dsm-clear").onclick = () => { dsmFile = null; showDsm(); };
      $("an-dsm").onchange = () => {
        const f = $("an-dsm").files[0];
        if (f) dsmFile = f;
        $("an-dsm").value = "";
        showDsm();
      };
      showDsm();
      $("an-file").textContent = info.file.name;
      const facts = [["Image", `${fmtInt(info.width)} × ${fmtInt(info.height)} pixels`]];
      if (info.geo) {
        facts.push(["Ground pixel size", `${fmt1(Math.sqrt(info.geo.px * info.geo.py) * 100)} cm`]);
        facts.push(["Coordinates", info.geo.crs]);
      } else {
        facts.push(["Coordinates", "none (not a GeoTIFF)"]);
      }
      if (info.bits > 8) facts.push(["Colour depth", info.bits + "-bit (converted)"]);
      $("an-facts").innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("");
      $("an-pixel-row").hidden = !!info.geo;
      modal.returnValue = "";
      modal.showModal();
      modal.addEventListener("close", function done() {
        modal.removeEventListener("close", done);
        if (modal.returnValue !== "go") { resolve(null); return; }
        const design = $("an-design").value;
        resolve({
          designFt: design === "auto" ? null : design.split(",").map(Number),
          budget: Number($("an-detail").value) * 1e6,
          pixel: info.geo ? null : Number($("an-pixel").value) / 100,
          dsm: dsmFile,
        });
      });
    });
  }

  function alphaIndex(info) {
    return info.kind === "tiff" && info.spp >= 4
      && (info.extra.includes(1) || info.extra.includes(2) || (info.extra.length === 0 && info.spp === 4)) ? 3 : -1;
  }

  // Share of the image that is field (alpha band), so the detail budget is
  // spent on the field, not on the empty corners of the image.
  async function fieldFraction(info) {
    const alpha = alphaIndex(info);
    if (alpha < 0) return 1;
    try {
      const w = Math.min(600, info.width), h = Math.max(1, Math.round((info.height * w) / info.width));
      let src = info.image;
      const count = await info.tiff.getImageCount();
      for (let i = 1; i < count; i++) {
        const im = await info.tiff.getImage(i);
        if (im.getWidth() >= w && im.getWidth() < src.getWidth() && im.getSamplesPerPixel() > alpha) src = im;
      }
      const data = await src.readRasters({ samples: [alpha], width: w, height: h });
      let inside = 0;
      for (const v of data[0]) if (v > 127) inside++;
      return Math.max(0.05, inside / data[0].length);
    } catch (e) {
      return 1;
    }
  }

  // Field share of a GeoTIFF from its full-resolution alpha band (the same
  // count as in the notebook), 1 without an alpha band.
  async function fieldFractionFull(info) {
    const alpha = alphaIndex(info);
    if (info.kind !== "tiff" || alpha < 0) return 1;
    const W = info.width, H = info.height;
    const rows = Math.max(1, Math.floor(60e6 / W));
    let inside = 0;
    for (let r0 = 0; r0 < H; r0 += rows) {
      const r1 = Math.min(H, r0 + rows);
      const data = await info.image.readRasters({ samples: [alpha], window: [0, r0, W, r1] });
      inside += PecanAnalysis.countField(data[0]);
      setLoading(`Measuring the field … ${Math.round((100 * r1) / H)}%`);
      await nextFrame();
    }
    return inside / (W * H);
  }

  // GeoTIFF: full-resolution rows averaged onto the W x H analysis grid by
  // the shared code (identical to the notebook).
  async function readGrid(info, W, H) {
    const alpha = alphaIndex(info);
    const samples = alpha >= 0 ? [0, 1, 2, alpha] : [0, 1, 2];
    const grid = PecanAnalysis.createGrid(info.width, info.height, W, H, {
      hasAlpha: alpha >= 0, nodata: info.nodata, bits: info.bits,
    });
    const rows = Math.max(1, Math.floor(30e6 / (info.width * samples.length)));
    for (let r0 = 0; r0 < info.height; r0 += rows) {
      const r1 = Math.min(info.height, r0 + rows);
      const data = await info.image.readRasters({ samples, window: [0, r0, info.width, r1] });
      grid.addRows(r0, r1 - r0, data);
      setLoading(`Reading the image … ${Math.round((100 * r1) / info.height)}%`);
      await nextFrame();
    }
    return grid.finish();
  }

  async function readPixels(info, W, H) {
    if (info.kind === "tiff") return readGrid(info, W, H);
    const rgba = new Uint8ClampedArray(W * H * 4);
    if (info.kind === "picture") {
      const canvas = document.createElement("canvas");
      canvas.width = W; canvas.height = H;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(info.bitmap, 0, 0, W, H);
      rgba.set(ctx.getImageData(0, 0, W, H).data);
      return rgba;
    }
    // GeoTIFF: use an overview if there is one; read in bands of rows.
    let src = info.image, sw = info.width, sh = info.height;
    const count = await info.tiff.getImageCount();
    for (let i = 1; i < count; i++) {
      const im = await info.tiff.getImage(i);
      if (im.getWidth() >= W && im.getWidth() < sw && im.getSamplesPerPixel() >= 3) { src = im; sw = im.getWidth(); sh = im.getHeight(); }
    }
    const alpha = alphaIndex(info);
    const samples = alpha >= 0 ? [0, 1, 2, 3] : [0, 1, 2];
    const wide = info.bits > 8;
    const raw = wide ? [new Float32Array(W * H), new Float32Array(W * H), new Float32Array(W * H)] : null;
    const rows = Math.max(1, Math.floor(30e6 / (sw * samples.length)));
    for (let r0 = 0; r0 < sh; r0 += rows) {
      const r1 = Math.min(sh, r0 + rows);
      const t0 = Math.round((r0 * H) / sh), t1 = Math.round((r1 * H) / sh);
      if (t1 <= t0) continue;
      const data = await src.readRasters({ samples, window: [0, r0, sw, r1], width: W, height: t1 - t0, resampleMethod: "bilinear" });
      for (let y = t0; y < t1; y++) {
        for (let x = 0; x < W; x++) {
          const k = (y - t0) * W + x, i = y * W + x, o = 4 * i;
          if (wide) { raw[0][i] = data[0][k]; raw[1][i] = data[1][k]; raw[2][i] = data[2][k]; }
          else { rgba[o] = data[0][k]; rgba[o + 1] = data[1][k]; rgba[o + 2] = data[2][k]; }
          let ok = alpha >= 0 ? data[3][k] > 127 : true;
          if (ok && alpha < 0 && info.nodata !== null && info.nodata !== undefined) {
            const nd = Number(info.nodata);
            if (data[0][k] === nd && data[1][k] === nd && data[2][k] === nd) ok = false;
          }
          rgba[o + 3] = ok ? 255 : 0;
        }
      }
      setLoading(`Reading the image … ${Math.round((100 * r1) / sh)}%`);
      await nextFrame();
    }
    if (wide) {  // 16-bit: stretch the 1st-99th percentiles to 0-255
      for (let c = 0; c < 3; c++) {
        const vals = [];
        for (let i = 0; i < W * H; i += 7) if (rgba[4 * i + 3]) vals.push(raw[c][i]);
        vals.sort((p, q) => p - q);
        const lo = vals[Math.floor(0.01 * (vals.length - 1))] || 0, hi = vals[Math.floor(0.99 * (vals.length - 1))] || 1;
        for (let i = 0; i < W * H; i++) rgba[4 * i + c] = ((raw[c][i] - lo) / Math.max(hi - lo, 1e-9)) * 255;
      }
    }
    return rgba;
  }

  async function previewUrl(rgba, W, H) {
    const a = document.createElement("canvas");
    a.width = W; a.height = H;
    a.getContext("2d").putImageData(new ImageData(rgba, W, H), 0, 0);
    const b = document.createElement("canvas");
    b.width = W; b.height = H;
    const ctx = b.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);
    ctx.drawImage(a, 0, 0);
    const blob = await new Promise((resolve) => b.toBlob(resolve, "image/jpeg", 0.88));
    return URL.createObjectURL(blob);
  }

  async function analyseImage(files) {
    files = Array.isArray(files) ? files : [files];
    setLoading("Opening " + files.map((f) => f.name).join(" + ") + " …");
    await nextFrame();
    let info = null, dsm = null;
    try {
      for (const f of files) {
        if ((await fileKind(f)) === "elevation") { if (!dsm) dsm = f; }
        else if (!info) info = await inspectImage(f);
      }
    } finally { setLoading(null); }
    if (!info) throw new Error("This is an elevation (DSM) file. Drop it together with the drone image (orthomosaic) of the same field, or add it in the window that opens for the image.");
    const file = info.file;
    const choice = await askAnalysis(info, dsm);
    if (!choice) return;
    const px = info.geo ? info.geo.px : choice.pixel, py = info.geo ? info.geo.py : choice.pixel;
    if (!(px > 0 && py > 0)) throw new Error("Enter the ground size of one pixel.");
    setLoading("Measuring the field …");
    await nextFrame();
    const fraction = info.kind === "tiff" ? await fieldFractionFull(info) : 1;
    const { W, H, pixel } = PecanAnalysis.gridSize(info.width, info.height, px, py, fraction, choice.budget);
    const t0 = performance.now();
    setLoading("Reading the image …");
    await nextFrame();
    const rgba = await readPixels(info, W, H);
    const url = await previewUrl(rgba, W, H);
    let surface = null, dsmProblem = "";
    if (choice.dsm) {
      setLoading("Reading the elevation (DSM) …");
      await nextFrame();
      try {
        surface = await readSurface(choice.dsm, info, W, H);
      } catch (err) {
        console.warn(err);
        dsmProblem = `${choice.dsm.name} was not used: ${err.message || err}.`;
      }
    }
    let result;
    try {
      result = await PecanAnalysis.analyze({ rgba, width: W, height: H, pixel, surface }, { designFt: choice.designFt, directionHintDeg: 114 }, (text) => setLoading(text));
    } catch (err) {
      URL.revokeObjectURL(url);
      setLoading(null);
      throw err;
    }
    surface = null;
    const elevation = result.segmentation.elevation;
    if (choice.dsm && !dsmProblem && !elevation.used) dsmProblem = `${choice.dsm.name} was not used: ${elevation.reason}.`;
    if (choice.dsm) elevation.file = choice.dsm.name;
    setLoading("Drawing the field …");
    await nextFrame();
    buildFromAnalysis(result, {
      name: info.name, pixel, crs: info.geo ? info.geo.crs : null,
      origin: info.geo ? info.geo.origin : [0, 0], toLonLat: info.geo ? info.geo.toLonLat : null,
      previewUrl: url, widthM: W * pixel, heightM: H * pixel,
    });
    setLoading(null);
    const took = Math.round((performance.now() - t0) / 1000);
    toast(dsmProblem
      ? `Analysed ${file.name} from the image only. ${dsmProblem}`
      : `Analysed ${file.name}${elevation.used ? " with its elevation (DSM)" : ""} in ${took} s: ${fmtInt(state.crowns.size)} crowns, ${state.diagonals.size} diagonals.`,
      dsmProblem ? "error" : "", dsmProblem ? 10000 : 6000);
  }

  async function loadZip(blob, name) {
    setLoading("Opening " + name + " …");
    await nextFrame();
    const zip = await JSZip.loadAsync(blob);
    const byName = new Map();
    zip.forEach((path, entry) => {
      if (!entry.dir) {
        const base = basename(path);
        if (!byName.has(base)) byName.set(base, entry);
      }
    });
    await loadEntries({
      name: name.replace(/\.zip$/i, ""),
      text: async (n) => (byName.has(n) ? byName.get(n).async("string") : null),
      blob: async (n) => (byName.has(n) ? byName.get(n).async("blob") : null),
    });
  }

  async function loadEntries(src) {
    setLoading("Reading crowns …");
    await nextFrame();
    const crownsText = await src.text("crowns_wgs84.geojson");
    const guidesText = await src.text("guides_wgs84.geojson");
    if (!crownsText || !guidesText) {
      throw new Error("This ZIP has no pecan results in it. Drop the drone image itself (GeoTIFF, JPG or PNG) to analyse it here, or a results ZIP from the notebook.");
    }
    const summaryText = await src.text("summary.json");
    const previewMetaText = await src.text("web_preview.json");
    const previewBlob = previewMetaText ? await src.blob("web_preview.jpg") : null;

    const data = {
      name: src.name,
      crowns: JSON.parse(crownsText),
      guides: JSON.parse(guidesText),
      summary: summaryText ? JSON.parse(summaryText) : null,
      previewMeta: previewMetaText ? JSON.parse(previewMetaText) : null,
      previewBlob,
    };
    setLoading("Drawing the field …");
    await nextFrame();
    buildDataset(data);
    setLoading(null);
  }

  // ------------------------------------------------------------------ dataset
  function firstCoord(geom) {
    let c = geom.coordinates;
    while (Array.isArray(c[0])) c = c[0];
    return c;
  }

  function geomToPolys(geom, project) {
    if (!geom) return [];
    const polys = geom.type === "Polygon" ? [geom.coordinates] : geom.type === "MultiPolygon" ? geom.coordinates : [];
    return polys.map((poly) => poly.map((ring) => simplify(ring.map(project), CROWN_SIMPLIFY_M)));
  }

  function geomToLines(geom, project) {
    if (!geom) return [];
    const lines = geom.type === "LineString" ? [geom.coordinates]
      : geom.type === "MultiLineString" ? geom.coordinates
      : geom.type === "GeometryCollection" ? geom.geometries.flatMap((g) => (g.type === "LineString" ? [g.coordinates] : g.type === "MultiLineString" ? g.coordinates : []))
      : [];
    return lines.filter((l) => l.length > 1).map((l) => simplify(l.map(project), LINE_SIMPLIFY_M));
  }

  function resetState() {
    for (const group of [layers.image, layers.crowns, layers.labels, layers.start, layers.select, layers.draw]) group.clearLayers();
    for (const f of ["EVEN", "ODD"]) { layers.lines[f].clearLayers(); layers.arrows[f].clearLayers(); }
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    Object.assign(state, {
      loaded: false, crowns: new Map(), diagonals: new Map(), selected: null, selectedDiagonal: null,
      mode: "select", mergeFrom: null, history: [], future: [], addedCount: 0, startOnly: false, previewUrl: null,
    });
    $("toggle-start-only").checked = false;
  }

  function buildDataset(data) {
    resetState();
    const features = (data.crowns.features || []).filter((f) => f.geometry);
    if (!features.length) throw new Error("No crowns in crowns_wgs84.geojson.");

    state.name = data.name;
    state.summary = data.summary;
    const crsText = (data.summary && data.summary.crs) || (data.previewMeta && data.previewMeta.crs) || "";
    state.proj = makeProjection(crsText, firstCoord(features[0].geometry));

    // Local metres: origin at the lower-left of the drone image (or of the crowns).
    let origin;
    const meta = data.previewMeta;
    const previewOk = meta && data.previewBlob && state.proj.exact && meta.crs === crsText;
    if (previewOk) {
      origin = [meta.bounds[0], meta.bounds[1]];
    } else {
      origin = [Infinity, Infinity];
      for (const f of features.slice(0, 400)) {
        const [x, y] = state.proj.forward(firstCoord(f.geometry));
        origin = [Math.min(origin[0], x), Math.min(origin[1], y)];
      }
      origin = [Math.floor(origin[0] - 50), Math.floor(origin[1] - 50)];
    }
    state.origin = origin;
    state.toLonLat = (x, y) => state.proj.inverse([x + origin[0], y + origin[1]]);
    const project = (ll) => {
      const [x, y] = state.proj.forward(ll);
      return [x - origin[0], y - origin[1]];
    };

    // Diagonals.
    const settings = (data.summary && data.summary.diagonal_settings) || {};
    if (Number.isFinite(settings.diagonal_bearing_deg)) {
      const b = (settings.diagonal_bearing_deg * Math.PI) / 180;
      state.axis = [Math.sin(b), Math.cos(b)];
    } else {
      state.axis = null;
    }
    for (const f of data.guides.features || []) {
      const number = Number(f.properties && f.properties.diagonal);
      if (!Number.isFinite(number)) continue;
      const pieces = geomToLines(f.geometry, project);
      state.diagonals.set(number, {
        number,
        family: family(number),
        pieces,
        length: pieces.reduce((s, p) => s + pathLength(p), 0),
        props: f.properties || {},
      });
    }

    // Crowns.
    let bounds = [Infinity, Infinity, -Infinity, -Infinity];
    for (const f of features) {
      const p = f.properties || {};
      const id = String(p.crown_id || p.label_id || "C" + state.crowns.size);
      const polys = geomToPolys(f.geometry, project);
      if (!polys.length) continue;
      const diagonal = Number(p.diagonal);
      const bbox = polysBBox(polys);
      bounds = [Math.min(bounds[0], bbox[0]), Math.min(bounds[1], bbox[1]), Math.max(bounds[2], bbox[2]), Math.max(bounds[3], bbox[3])];
      state.crowns.set(id, {
        id,
        props: p,
        geomWgs: f.geometry,
        polys,
        bbox,
        area: Number(p.area_m2) || polysArea(polys),
        diagonal: Number.isFinite(diagonal) ? diagonal : 0,
        origDiagonal: Number.isFinite(diagonal) ? diagonal : null,
        excluded: false,
        removed: false,
        source: "model",
        members: [],
        layer: null,
      });
      if (!state.diagonals.has(diagonal) && Number.isFinite(diagonal)) {
        state.diagonals.set(diagonal, { number: diagonal, family: family(diagonal), pieces: [], length: 0, props: {} });
      }
    }
    state.bounds = L.latLngBounds([bounds[1], bounds[0]], [bounds[3], bounds[2]]);

    let preview = null;
    if (previewOk) {
      const [l, b, r, t] = meta.bounds;
      preview = { url: URL.createObjectURL(data.previewBlob), bounds: [[b - origin[1], l - origin[0]], [t - origin[1], r - origin[0]]] };
    }
    finishDataset(preview, "This results ZIP has no drone image (older notebook version).");
  }

  // Results of the analysis done here in the browser (local metres).
  function buildFromAnalysis(result, meta) {
    resetState();
    state.name = meta.name;
    state.summary = {
      crs: meta.crs || null,
      source: "browser",
      pixel_m: meta.pixel,
      diagonal_settings: result.settings,
      field_area_m2: result.field_area_m2,
      canopy_area_m2: result.canopy_area_m2,
      segmentation: result.segmentation || { elevation: { used: false } },
      start: { diagonal: 2 },
    };
    state.proj = null;
    state.origin = meta.origin || [0, 0];
    state.toLonLat = meta.toLonLat || null;
    const b = (result.settings.diagonal_bearing_deg * Math.PI) / 180;
    state.axis = [Math.sin(b), Math.cos(b)];

    for (const d of result.diagonals) {
      const pieces = d.pieces.map((p) => simplify(p, LINE_SIMPLIFY_M)).filter((p) => p.length > 1);
      state.diagonals.set(d.number, {
        number: d.number, family: family(d.number), pieces,
        length: pieces.reduce((s, p) => s + pathLength(p), 0), props: {},
      });
    }
    let bounds = [Infinity, Infinity, -Infinity, -Infinity];
    for (const c of result.crowns) {
      const bbox = polysBBox(c.polys);
      bounds = [Math.min(bounds[0], bbox[0]), Math.min(bounds[1], bbox[1]), Math.max(bounds[2], bbox[2]), Math.max(bounds[3], bbox[3])];
      state.crowns.set(c.id, {
        id: c.id,
        props: {
          crown_id: c.id, area_m2: c.area, diagonal: c.diagonal, grid_row: c.grid_row, grid_tree: c.grid_tree,
          top_x: c.top[0] + state.origin[0], top_y: c.top[1] + state.origin[1], review_flags: c.flags,
          height_median_m: c.height_median_m, height_p90_m: c.height_p90_m,
          height_top_m: c.height_top_m, height_mean_m: c.height_mean_m,
        },
        geomWgs: null,
        polys: c.polys,
        bbox,
        area: c.area,
        diagonal: c.diagonal,
        origDiagonal: c.diagonal,
        excluded: false,
        removed: false,
        source: "model",
        members: [],
        layer: null,
      });
      if (!state.diagonals.has(c.diagonal)) {
        state.diagonals.set(c.diagonal, { number: c.diagonal, family: family(c.diagonal), pieces: [], length: 0, props: {} });
      }
    }
    if (!state.crowns.size) throw new Error("No pecan crowns were found in this image.");
    state.bounds = L.latLngBounds([bounds[1], bounds[0]], [bounds[3], bounds[2]]);
    finishDataset({ url: meta.previewUrl, bounds: [[0, 0], [meta.heightM, meta.widthM]] }, "");
  }

  function finishDataset(preview, noPreviewNote) {
    const settings = (state.summary && state.summary.diagonal_settings) || {};
    state.lattice = buildLattice(settings);

    // Drone image.
    if (preview) {
      state.previewUrl = preview.url;
      L.imageOverlay(preview.url, preview.bounds, { pane: "image", interactive: false }).addTo(layers.image);
    }
    $("layer-image").disabled = !preview;
    $("layer-image").parentElement.title = preview ? "" : noPreviewNote;

    // Start.
    const evens = [...state.diagonals.values()].filter((d) => d.family === "EVEN" && d.pieces.length).map((d) => d.number).sort((a, b) => a - b);
    const suggested = state.summary && state.summary.start && Number(state.summary.start.diagonal);
    state.start = evens.includes(suggested) ? suggested : evens.includes(2) ? 2 : evens[0] ?? null;

    // Key for saved corrections.
    let total = 0;
    for (const c of state.crowns.values()) total += c.area;
    state.key = STORAGE_PREFIX + [state.name, state.crowns.size, Math.round(total)].join("|");

    drawCrowns();
    drawDiagonals();
    state.loaded = true;

    $("dropzone").hidden = true;
    for (const id of ["card-field", "card-totals", "card-plan", "card-checks", "card-start", "card-layers", "card-edit", "card-diagonals", "chart-panel"]) $(id).hidden = false;
    state.conflictAt = -1;
    $("btn-export").disabled = false;

    map.invalidateSize();
    fitField();
    state.fitZoom = map.getZoom();

    const saved = storageGet(state.key);
    if (saved && Number.isFinite(saved.start) && evens.includes(saved.start)) state.start = saved.start;
    let restored = 0;
    if (saved && Array.isArray(saved.edits) && saved.edits.length) {
      for (const edit of saved.edits) {
        try { applyEdit(edit); state.history.push(edit); restored++; } catch (e) { console.warn("Skipped a saved correction", e); }
      }
    }

    fillFieldCard();
    fillStartSelect(evens);
    drawStart();
    refreshAll();
    initChart();
    updateLabelVisibility();

    toast(restored
      ? `Opened ${state.name}. Restored ${restored} correction${restored === 1 ? "" : "s"} saved in this browser.`
      : `Opened ${state.name}: ${fmtInt(state.crowns.size)} crowns, ${state.diagonals.size} diagonals.`, "", 4500);
  }

  // ------------------------------------------------------------------ drawing
  function crownStyle(c) {
    if (c.excluded) {
      return { color: "#ffffff", weight: 0.8, opacity: 0.9, fillColor: EXCLUDED_COLOR, fillOpacity: Math.max(0.3, state.fillOpacity), dashArray: "3 3" };
    }
    const f = family(c.diagonal);
    let fo = state.fillOpacity;
    if (state.startOnly && c.diagonal !== state.start) fo *= 0.25;
    if (state.selectedDiagonal !== null && c.diagonal === state.selectedDiagonal) fo = Math.min(0.85, fo + 0.3);
    const edited = isEdited(c);
    return {
      color: edited ? "#111111" : "#ffffff",
      weight: edited ? 1.6 : 0.7,
      opacity: 0.95,
      fillColor: COLORS[f],
      fillOpacity: fo,
      dashArray: null,
    };
  }

  function isEdited(c) {
    return c.source === "added" || c.members.length > 0 || (c.origDiagonal !== null && c.diagonal !== c.origDiagonal);
  }

  function crownTip(c) {
    const f = c.excluded ? "not pecan" : family(c.diagonal);
    return `<strong>${c.id}</strong> · ${dname(c.diagonal)} ${f}<br>${fmt1(c.area)} m²`;
  }

  function makeCrownLayer(c) {
    const layer = L.polygon(polysToLL(c.polys), { renderer: renderers.crowns, pane: "main", ...crownStyle(c) });
    layer.bindTooltip(() => crownTip(c), { sticky: true, className: "crown-tip", direction: "top", offset: [0, -6] });
    layer.on("click", (e) => onCrownClick(c, e));
    c.layer = layer;
    return layer;
  }

  function drawCrowns() {
    layers.crowns.clearLayers();
    for (const c of state.crowns.values()) if (!c.removed) layers.crowns.addLayer(makeCrownLayer(c));
  }

  // A crown added later goes under the lines (they share one canvas).
  function addCrownLayer(c) {
    const layer = makeCrownLayer(c);
    layers.crowns.addLayer(layer);
    layer.bringToBack();
    return layer;
  }

  function restyleCrowns(ids) {
    const list = ids ? ids.map((id) => state.crowns.get(id)).filter(Boolean) : [...state.crowns.values()];
    for (const c of list) if (c.layer && !c.removed) c.layer.setStyle(crownStyle(c));
  }

  // End of a diagonal: lowest (southern-most) end, as in the notebook maps.
  function diagonalEnd(d) {
    let best = null;
    for (const piece of d.pieces) {
      for (const [i, j] of [[0, 1], [piece.length - 1, piece.length - 2]]) {
        const tip = piece[i];
        if (!best || tip[1] < best.tip[1]) {
          let k = j, back = piece[j];
          const step = i === 0 ? 1 : -1;
          while (k > 0 && k < piece.length - 1 && Math.hypot(back[0] - tip[0], back[1] - tip[1]) < 3) { k += step; back = piece[k]; }
          const len = Math.hypot(tip[0] - back[0], tip[1] - back[1]) || 1;
          best = { tip, dir: [(tip[0] - back[0]) / len, (tip[1] - back[1]) / len], piece, atStart: i === 0 };
        }
      }
    }
    return best;
  }

  function diagonalTop(d) {
    let best = null;
    for (const piece of d.pieces) for (const tip of [piece[0], piece[piece.length - 1]]) if (!best || tip[1] > best[1]) best = tip;
    return best;
  }

  function arrowHead(end, length, width) {
    const [x, y] = end.tip, [dx, dy] = end.dir;
    const bx = x - dx * length, by = y - dy * length;
    return [[x, y], [bx - dy * width, by + dx * width], [bx + dy * width, by - dx * width]].map(toLL);
  }

  function drawDiagonals() {
    for (const f of ["EVEN", "ODD"]) { layers.lines[f].clearLayers(); layers.arrows[f].clearLayers(); }
    layers.labels.clearLayers();
    for (const d of state.diagonals.values()) {
      if (!d.pieces.length) continue;
      const colour = LINE_COLORS[d.family];
      d.lineLayers = [];
      d.haloLayers = [];
      for (const piece of d.pieces) {
        const ll = piece.map(toLL);
        const halo = L.polyline(ll, { renderer: renderers.lines, pane: "main", color: "#ffffff", weight: 4.2, opacity: 0.9, interactive: false });
        halo.addTo(layers.lines[d.family]);
        d.haloLayers.push(halo);
        const line = L.polyline(ll, { renderer: renderers.lines, pane: "main", color: colour, weight: 2, opacity: 1 });
        line.bindTooltip(() => diagonalTip(d), { sticky: true, className: "crown-tip", direction: "top", offset: [0, -6] });
        line.on("click", (e) => onDiagonalClick(d, e));
        line.addTo(layers.lines[d.family]);
        d.lineLayers.push(line);
      }
      const end = diagonalEnd(d);
      d.end = end;
      if (end) {
        d.arrowLayer = L.polygon(arrowHead(end, 4, 1.9), { renderer: renderers.lines, pane: "main", color: "#ffffff", weight: 1, fillColor: colour, fillOpacity: 1, interactive: false });
        d.arrowLayer.addTo(layers.arrows[d.family]);
        d.label = L.marker(labelPoint(end), { icon: labelIcon(d, 0), interactive: false, keyboard: false });
        layers.labels.addLayer(d.label);
      }
    }
  }

  function labelAxis(end) {
    let [ax, ay] = state.axis || end.dir;
    if (ax * end.dir[0] + ay * end.dir[1] < 0) { ax = -ax; ay = -ay; }
    return [ax, ay];
  }

  function labelPoint(end) {
    const [ax, ay] = labelAxis(end);
    return toLL([end.tip[0] + ax * 1.6, end.tip[1] + ay * 1.6]);
  }

  function labelIcon(d, area) {
    const [ax, ay] = labelAxis(d.end);
    let angle = (-Math.atan2(ay, ax) * 180) / Math.PI;
    let style;
    if (Math.abs(angle) > 90) {
      angle -= Math.sign(angle) * 180;
      style = `transform-origin:100% 50%;transform:translate(-100%,-50%) rotate(${angle.toFixed(1)}deg)`;
    } else {
      style = `transform-origin:0 50%;transform:translate(0,-50%) rotate(${angle.toFixed(1)}deg)`;
    }
    return L.divIcon({
      className: "diag-label " + d.family.toLowerCase(),
      html: `<span style="${style}">${fmtArea(area)}</span>`,
      iconSize: [0, 0],
      iconAnchor: [0, 0],
    });
  }

  function diagonalTip(d) {
    const t = state.totals && state.totals.per.get(d.number);
    return `<strong>${dname(d.number)}</strong> · ${d.family}<br>${t ? fmtInt(t.count) + " crowns · " + fmtArea(t.area) : ""}`;
  }

  // Line width follows the zoom: thin when the whole field is shown (lines
  // are ~5 m apart), full width when zoomed in.
  function styleLinesForZoom() {
    const spacing = ((state.summary && state.summary.diagonal_settings && state.summary.diagonal_settings.diagonal_spacing_m) || 5.4) * Math.pow(2, map.getZoom());
    const weight = Math.max(0.7, Math.min(2.2, spacing * 0.16));
    const halo = weight + Math.max(0.8, Math.min(2.2, spacing * 0.12));
    // Arrow heads keep about the same size on screen at every zoom.
    const pxPerM = Math.pow(2, map.getZoom());
    const arrowPx = Math.max(7, Math.min(15, spacing * 1.1));
    for (const d of state.diagonals.values()) {
      for (const l of d.lineLayers || []) l.setStyle({ weight });
      for (const l of d.haloLayers || []) l.setStyle({ weight: halo });
      if (d.arrowLayer && d.end) d.arrowLayer.setLatLngs(arrowHead(d.end, arrowPx / pxPerM, (0.45 * arrowPx) / pxPerM));
    }
    if (state.startArrow) {
      const d = state.diagonals.get(state.start);
      if (d && d.end) state.startArrow.setLatLngs(arrowHead(d.end, (1.9 * arrowPx) / pxPerM, (0.95 * arrowPx) / pxPerM));
    }
  }

  function updateLabelVisibility() {
    if (!state.loaded) return;
    styleLinesForZoom();
    const want = $("layer-labels").checked;
    const zoomOk = map.getZoom() >= state.fitZoom + 0.75;
    const show = want && zoomOk && !state.startOnly;
    if (show && !map.hasLayer(layers.labels)) layers.labels.addTo(map);
    if (!show && map.hasLayer(layers.labels)) map.removeLayer(layers.labels);
    const note = $("map-note");
    if (want && !zoomOk && !state.startOnly) {
      note.textContent = "Zoom in to read the canopy total at the end of each line";
      note.hidden = false;
    } else {
      note.hidden = true;
    }
  }

  function drawStart() {
    layers.start.clearLayers();
    const d = state.diagonals.get(state.start);
    if (!d || !d.pieces.length) return;
    for (const piece of d.pieces) {
      const ll = piece.map(toLL);
      L.polyline(ll, { renderer: renderers.start, pane: "start", color: "#ffffff", weight: 9, opacity: 0.95, interactive: false }).addTo(layers.start);
      L.polyline(ll, { renderer: renderers.start, pane: "start", color: LINE_COLORS.EVEN, weight: 5, opacity: 1, interactive: false }).addTo(layers.start);
    }
    state.startArrow = null;
    if (d.end) {
      state.startArrow = L.polygon(arrowHead(d.end, 9, 4.5), { renderer: renderers.start, pane: "start", color: "#ffffff", weight: 2, fillColor: LINE_COLORS.EVEN, fillOpacity: 1, interactive: false });
      state.startArrow.addTo(layers.start);
      if (state.loaded) styleLinesForZoom();
    }
    const top = diagonalTop(d);
    if (top) {
      L.marker(toLL(top), {
        icon: L.divIcon({ className: "start-label", html: `<span>START<br>${dname(d.number)} (EVEN)</span>`, iconSize: [0, 0] }),
        interactive: false,
        keyboard: false,
        zIndexOffset: 1000,
      }).addTo(layers.start);
    }
  }

  function drawSelection() {
    layers.select.clearLayers();
    const c = state.selected && state.crowns.get(state.selected);
    if (c && !c.removed) {
      L.polygon(polysToLL(c.polys), { renderer: renderers.select, pane: "select", color: SELECT_COLOR, weight: 3.2, fill: false, interactive: false }).addTo(layers.select);
    }
    const d = state.selectedDiagonal !== null && state.diagonals.get(state.selectedDiagonal);
    if (d) {
      for (const piece of d.pieces) {
        L.polyline(piece.map(toLL), { renderer: renderers.select, pane: "select", color: SELECT_COLOR, weight: 7, opacity: 0.55, interactive: false }).addTo(layers.select);
        L.polyline(piece.map(toLL), { renderer: renderers.select, pane: "select", color: LINE_COLORS[d.family], weight: 3, opacity: 1, interactive: false }).addTo(layers.select);
      }
    }
  }

  function applyLineVisibility() {
    for (const f of ["EVEN", "ODD"]) {
      const on = $("layer-" + f.toLowerCase()).checked && !state.startOnly;
      for (const g of [layers.lines[f], layers.arrows[f]]) {
        if (on && !map.hasLayer(g)) g.addTo(map);
        if (!on && map.hasLayer(g)) map.removeLayer(g);
      }
    }
    const crownsOn = $("layer-crowns").checked;
    if (crownsOn && !map.hasLayer(layers.crowns)) {
      layers.crowns.addTo(map);
      layers.crowns.eachLayer((layer) => layer.bringToBack());
    }
    if (!crownsOn && map.hasLayer(layers.crowns)) map.removeLayer(layers.crowns);
    const imageOn = $("layer-image").checked;
    if (imageOn && !map.hasLayer(layers.image)) layers.image.addTo(map);
    if (!imageOn && map.hasLayer(layers.image)) map.removeLayer(layers.image);
    updateLabelVisibility();
  }

  // ------------------------------------------------------------------ totals + panels
  function computeTotals() {
    const per = new Map();
    for (const d of state.diagonals.keys()) per.set(d, { count: 0, area: 0 });
    const fam = { EVEN: { count: 0, area: 0, diagonals: 0 }, ODD: { count: 0, area: 0, diagonals: 0 } };
    const excluded = { count: 0, area: 0 };
    for (const c of state.crowns.values()) {
      if (c.removed) continue;
      if (c.excluded) { excluded.count++; excluded.area += c.area; continue; }
      if (!per.has(c.diagonal)) per.set(c.diagonal, { count: 0, area: 0 });
      const p = per.get(c.diagonal);
      p.count++;
      p.area += c.area;
    }
    for (const [d, p] of per) {
      const f = fam[family(d)];
      f.count += p.count;
      f.area += p.area;
      if (p.count) f.diagonals++;
    }
    return { per, fam, excluded, all: { count: fam.EVEN.count + fam.ODD.count, area: fam.EVEN.area + fam.ODD.area, diagonals: fam.EVEN.diagonals + fam.ODD.diagonals } };
  }

  function refreshAll() {
    state.totals = computeTotals();
    const t = state.totals;
    $("t-even-area").textContent = fmtArea(t.fam.EVEN.area);
    $("t-even-sub").textContent = `${fmtInt(t.fam.EVEN.count)} crowns · ${t.fam.EVEN.diagonals} diagonals`;
    $("t-odd-area").textContent = fmtArea(t.fam.ODD.area);
    $("t-odd-sub").textContent = `${fmtInt(t.fam.ODD.count)} crowns · ${t.fam.ODD.diagonals} diagonals`;
    $("t-all-area").textContent = fmtArea(t.all.area);
    $("t-all-sub").textContent = `${fmtInt(t.all.count)} crowns · ${t.all.diagonals} diagonals`;
    const ex = $("t-excluded");
    ex.hidden = !t.excluded.count;
    ex.textContent = `Not counted (marked “not a pecan”): ${fmtInt(t.excluded.count)} crowns, ${fmtArea(t.excluded.area)}.`;

    const n = state.history.length;
    $("edit-badge").hidden = !n;
    $("edit-badge").textContent = `${n} correction${n === 1 ? "" : "s"}`;
    $("edit-count").textContent = n ? `${n} correction${n === 1 ? "" : "s"} · saved in this browser` : "No corrections yet.";
    $("btn-undo").disabled = !n;
    $("btn-redo").disabled = !state.future.length;
    $("btn-reset").disabled = !n;

    for (const d of state.diagonals.values()) {
      if (d.label) d.label.setIcon(labelIcon(d, t.per.get(d.number) ? t.per.get(d.number).area : 0));
    }
    fillTable();
    fillStartFacts();
    fillSelected();
    fillPlan();
    fillChecks();
    updateChart();
  }

  // ------------------------------------------------------------------ decision support
  const pct = (part, whole) => (whole > 0 ? `${fmt1((100 * part) / whole)} %` : "–");

  function meanHeight(fam) {
    const hs = [];
    for (const c of state.crowns.values()) {
      if (c.removed || c.excluded || family(c.diagonal) !== fam) continue;
      const v = c.props && c.props.height_top_m;
      if (v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v))) hs.push(Number(v));
    }
    return hs.length ? hs.reduce((s, x) => s + x, 0) / hs.length : null;
  }

  // What thinning one family of diagonals does to the orchard.
  function fillPlan() {
    const t = state.totals;
    if (!t) return;
    const fam = $("plan-family").value || "EVEN";
    const other = fam === "EVEN" ? "ODD" : "EVEN";
    const rem = t.fam[fam], keep = t.fam[other];
    const field = state.summary && Number(state.summary.field_area_m2);
    const ds = (state.summary && state.summary.diagonal_settings) || {};
    const facts = [
      ["Trees to remove", `${fmtInt(rem.count)} · ${pct(rem.count, t.all.count)}`],
      ["Canopy removed", `${fmtArea(rem.area)} · ${pct(rem.area, t.all.area)}`],
      ["Trees left", fmtInt(keep.count)],
      ["Canopy left", fmtArea(keep.area)],
    ];
    if (field > 0) facts.push(["Canopy cover", `${pct(t.all.area, field)} → ${pct(keep.area, field)}`]);
    const hr = meanHeight(fam), hk = meanHeight(other);
    if (hr !== null && hk !== null) facts.push(["Height removed / left", `${fmt1(hr)} / ${fmt1(hk)} m`]);
    if (ds.in_row_spacing_m) {
      const after = 2 * ds.in_row_spacing_m;
      facts.push(["Spacing in row after", `${fmt1(after)} m · ${Math.round(after / 0.3048)} ft`]);
    }
    facts.push(["Diagonals to remove", String(rem.diagonals)]);
    $("plan-facts").innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("");
    const firstOdd = [...state.diagonals.values()].filter((d) => d.family === "ODD" && d.pieces.length).map((d) => d.number).sort((a, b) => a - b)[0];
    $("plan-note").textContent = fam === "EVEN"
      ? `Every blue (EVEN) tree is removed, every red (ODD) tree stays. Start on ${state.start !== null ? dname(state.start) : "the START line"} and follow the arrows. Compare: removing ODD instead takes ${fmtArea(keep.area)} (${fmtInt(keep.count)} trees).`
      : `Every red (ODD) tree is removed, every blue (EVEN) tree stays. Start on ${firstOdd !== undefined ? dname(firstOdd) : "the first ODD line"} and follow the arrows. Compare: removing EVEN instead takes ${fmtArea(keep.area)} (${fmtInt(keep.count)} trees).`;
  }

  // Neighbouring crowns in a row with the same colour (after corrections).
  function rowConflicts() {
    const rows = new Map();
    for (const c of state.crowns.values()) {
      if (c.removed || c.excluded) continue;
      const p = c.props || {};
      const r = Number(p.grid_row), k = Number(p.grid_tree), b = Number(p.planting_block || 1);
      if (!Number.isFinite(r) || !Number.isFinite(k) || p.grid_row === null || p.grid_row === "") continue;
      const key = b + ":" + r;
      if (!rows.has(key)) rows.set(key, []);
      rows.get(key).push([k, c]);
    }
    let pairs = 0;
    const bad = [];
    for (const list of rows.values()) {
      list.sort((x, y) => x[0] - y[0]);
      for (let i = 1; i < list.length; i++) {
        pairs++;
        if (family(list[i][1].diagonal) === family(list[i - 1][1].diagonal)) bad.push([list[i - 1][1], list[i][1]]);
      }
    }
    return { pairs, bad };
  }

  function fillChecks() {
    const s = state.summary || {};
    const ds = s.diagonal_settings || {};
    const el = (s.segmentation && s.segmentation.elevation) || {};
    const rc = rowConflicts();
    state.conflicts = rc.bad;
    const offGrid = [...state.crowns.values()].filter((c) => !c.removed && !c.excluded && /OFF_GRID/.test((c.props && c.props.review_flags) || "")).length;
    const items = [];
    items.push(rc.pairs
      ? [rc.bad.length === 0, rc.bad.length === 0
        ? `Blue / red alternate along every row: 0 of ${fmtInt(rc.pairs)} neighbours share a colour.`
        : `${rc.bad.length} of ${fmtInt(rc.pairs)} row neighbours share a colour.`, rc.bad.length ? "Show" : ""]
      : [null, "No row positions in these results (older notebook version).", ""]);
    items.push([!!el.used, el.used
      ? `Elevation (DSM) used${el.file ? ` (${el.file})` : el.path ? ` (${String(el.path).split(/[\\/]/).pop()})` : ""}: grass and weeds lower than 1 m are not counted; treetops from tree height.`
      : `No elevation (DSM)${el.reason ? `: ${el.reason}` : ""}. Add the DSM of the same flight for the most accurate canopy.`, ""]);
    if (Number.isFinite(ds.crowns_crossed_by_their_diagonal)) items.push([ds.crowns_crossed_by_their_diagonal >= 0.97, `${fmt1(100 * ds.crowns_crossed_by_their_diagonal)} % of crowns are crossed by their own diagonal line.`, ""]);
    items.push([offGrid === 0 ? true : null, offGrid ? `${fmtInt(offGrid)} crown${offGrid === 1 ? "" : "s"} off the planting grid (flag OFF_GRID): worth a look.` : "Every crown sits on the planting grid.", ""]);
    const shadow = s.segmentation && s.segmentation.cast_shadow;
    if (shadow && Number.isFinite(shadow.shadow_removed_m2)) items.push([true, `Cast shadow trimmed from crown edges: ${fmtArea(shadow.shadow_removed_m2)}.`, ""]);
    $("checks-list").innerHTML = items.map(([ok, text, action]) =>
      `<li class="${ok === true ? "ok" : ok === false ? "warn" : "info"}"><span>${escapeHtml(text)}</span>${action ? ` <button type="button" class="link-btn" data-check="${action}">${action}</button>` : ""}</li>`).join("");
  }

  function showNextConflict() {
    const list = state.conflicts || [];
    if (!list.length) return;
    state.conflictAt = ((state.conflictAt ?? -1) + 1) % list.length;
    const [a] = list[state.conflictAt];
    selectCrown(a.id, true);
    toast(`Row neighbours with the same colour ${state.conflictAt + 1} of ${list.length}: ${list[state.conflictAt][0].id} and ${list[state.conflictAt][1].id}. Move one to the next diagonal with ‹ ›.`, "", 6000);
  }

  // The trees to remove, for the field crew (GPS position of each crown).
  function exportPlan() {
    const fam = $("plan-family").value || "EVEN";
    const chosen = activeCrowns().filter((c) => !c.excluded && family(c.diagonal) === fam);
    const order = new Map();
    const byDiag = new Map();
    for (const c of chosen) {
      if (!byDiag.has(c.diagonal)) byDiag.set(c.diagonal, []);
      byDiag.get(c.diagonal).push(c);
    }
    for (const list of byDiag.values()) {
      list.sort((a, b) => polysCentroid(b.polys)[1] - polysCentroid(a.polys)[1]);
      list.forEach((c, k) => order.set(c.id, k + 1));
    }
    const header = ["crown_id", "diagonal", "family", "tree_on_diagonal", "grid_row", "grid_tree", "canopy_m2", "height_top_m"];
    header.push(...(state.toLonLat ? ["longitude", "latitude"] : ["x_m", "y_m"]));
    const lines = [csvRow(header)];
    chosen.sort((a, b) => a.diagonal - b.diagonal || order.get(a.id) - order.get(b.id));
    for (const c of chosen) {
      const p = c.props || {};
      const [x, y] = polysCentroid(c.polys);
      const where = state.toLonLat ? state.toLonLat(x, y).map((v) => v.toFixed(8)) : [x.toFixed(2), y.toFixed(2)];
      const height = p.height_top_m !== null && p.height_top_m !== undefined && p.height_top_m !== "" && Number.isFinite(Number(p.height_top_m)) ? Number(p.height_top_m).toFixed(2) : "";
      lines.push(csvRow([c.id, dname(c.diagonal), fam, order.get(c.id), p.grid_row ?? "", p.grid_tree ?? "", c.area.toFixed(2), height, ...where]));
    }
    download(`${fileBase()}_trees_to_remove_${fam}.csv`, lines.join("\n") + "\n");
  }

  function fillFieldCard() {
    const s = state.summary || {};
    const ds = s.diagonal_settings || {};
    $("field-name").textContent = "";
    const facts = [];
    facts.push([s.source === "browser" ? "Image" : "Results", state.name]);
    const elevationUsed = !!(s.segmentation && s.segmentation.elevation && s.segmentation.elevation.used);
    if (s.source === "browser") {
      facts.push(["Analysis", elevationUsed ? "in this browser (image + DSM)" : "in this browser (image only)"]);
      facts.push(["Pixel size", `${fmt1(s.pixel_m * 100)} cm`]);
    }
    if (ds.in_row_spacing_m) facts.push(["Tree spacing", `${fmt1(ds.in_row_spacing_m)} m × ${fmt1(ds.row_spacing_m)} m`]);
    if (Number.isFinite(ds.row_bearing_deg)) facts.push(["Row bearing", `${Math.round(ds.row_bearing_deg)}°`]);
    if (ds.diagonal_spacing_m) facts.push(["Diagonals", `${state.diagonals.size}, ${fmt1(ds.diagonal_spacing_m)} m apart`]);
    else facts.push(["Diagonals", String(state.diagonals.size)]);
    if (s.field_area_m2) facts.push(["Field area", `${fmt1(s.field_area_m2 / 10000)} ha`]);
    facts.push([s.source === "browser" ? "Crowns found" : "Crowns (notebook)", fmtInt(state.crowns.size)]);
    if (s.segmentation && s.segmentation.elevation) facts.push(["Elevation (DSM)", elevationUsed ? "used: tree heights" : "not used"]);
    $("field-facts").innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd title="${String(v).replace(/"/g, "&quot;")}">${escapeHtml(String(v))}</dd>`).join("");
    $("field-facts").querySelector("dd").style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:190px;justify-self:end";
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  }

  function fillStartSelect(evens) {
    const sel = $("start-select");
    sel.innerHTML = evens.map((d) => `<option value="${d}">${dname(d)}</option>`).join("");
    if (state.start !== null) sel.value = String(state.start);
  }

  function fillStartFacts() {
    const d = state.diagonals.get(state.start);
    const t = state.totals && state.totals.per.get(state.start);
    $("start-facts").innerHTML = d
      ? `<strong>${dname(d.number)}</strong> (EVEN) · ${t ? fmtInt(t.count) : 0} crowns · <strong>${fmtArea(t ? t.area : 0)}</strong> canopy · line ${fmtInt(d.length)} m. The arrow shows the direction of travel.`
      : "No EVEN diagonal found.";
  }

  function fillTable() {
    const t = state.totals;
    const rows = [...t.per.entries()].filter(([d]) => state.diagonals.has(d)).sort((a, b) => a[0] - b[0]);
    $("diag-table").tBodies[0].innerHTML = rows.map(([d, p]) => {
      const f = family(d).toLowerCase();
      const active = d === state.selectedDiagonal ? " class=\"active\"" : "";
      return `<tr data-d="${d}"${active}><td><span class="dot ${f}"></span>${dname(d)}${d === state.start ? " ▶" : ""}</td><td>${fmtInt(p.count)}</td><td>${fmtInt(p.area)}</td></tr>`;
    }).join("");
    const foot = [
      ["EVEN total", t.fam.EVEN.count, t.fam.EVEN.area],
      ["ODD total", t.fam.ODD.count, t.fam.ODD.area],
      ["All", t.all.count, t.all.area],
    ];
    if (t.excluded.count) foot.push(["Not a pecan", t.excluded.count, t.excluded.area]);
    $("diag-table").tFoot.innerHTML = foot.map(([k, n, a]) => `<tr><td>${k}</td><td>${fmtInt(n)}</td><td>${fmtInt(a)}</td></tr>`).join("");
  }

  function fillSelected() {
    const c = state.selected && state.crowns.get(state.selected);
    const box = $("selected-crown");
    box.hidden = !c || c.removed;
    if (c && !c.removed) {
      $("sel-id").textContent = c.id + (c.source === "added" ? " (added)" : "");
      const chip = $("sel-family");
      chip.textContent = c.excluded ? "NOT PECAN" : family(c.diagonal);
      chip.className = "chip " + (c.excluded ? "excluded" : family(c.diagonal).toLowerCase());
      const p = c.props || {};
      const facts = [["Canopy", fmt1(c.area) + " m²"]];
      if (Number.isFinite(Number(p.height_top_m)) && p.height_top_m !== null) facts.push(["Tree height", fmt1(Number(p.height_top_m)) + " m"]);
      if (p.grid_row !== undefined && p.grid_row !== null) facts.push(["Row / tree", `${p.grid_row} / ${p.grid_tree}`]);
      if (c.origDiagonal !== null && c.origDiagonal !== c.diagonal) facts.push(["Notebook diagonal", dname(c.origDiagonal)]);
      if (c.members.length) facts.push(["Merged pieces", c.members.join(", ")]);
      if (p.review_flags) facts.push(["Flags", String(p.review_flags).replace(/;/g, ", ")]);
      $("sel-facts").innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("");
      const sel = $("sel-diagonal");
      const numbers = [...state.diagonals.keys()].sort((a, b) => a - b);
      if (sel.dataset.count !== String(numbers.length)) {
        sel.innerHTML = numbers.map((d) => `<option value="${d}">${dname(d)} · ${family(d)}</option>`).join("");
        sel.dataset.count = String(numbers.length);
      }
      sel.value = String(c.diagonal);
      $("sel-prev").disabled = !numbers.includes(c.diagonal - 1);
      $("sel-next").disabled = !numbers.includes(c.diagonal + 1);
      $("sel-exclude").textContent = c.excluded ? "Count as pecan again" : "Not a pecan";
      $("sel-delete").hidden = c.source !== "added";
      $("sel-merge").disabled = c.excluded;
    }

    const d = state.selectedDiagonal !== null && state.diagonals.get(state.selectedDiagonal);
    $("selected-diagonal").hidden = !d;
    if (d) {
      const t = state.totals.per.get(d.number) || { count: 0, area: 0 };
      $("seld-id").textContent = dname(d.number);
      $("seld-family").textContent = d.family;
      $("seld-family").className = "chip " + d.family.toLowerCase();
      $("seld-facts").innerHTML = [
        ["Crowns", fmtInt(t.count)],
        ["Canopy", fmtArea(t.area)],
        ["Line length", fmtInt(d.length) + " m"],
      ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
      $("seld-start").hidden = d.family !== "EVEN";
      $("seld-start").disabled = d.number === state.start;
      $("seld-start").textContent = d.number === state.start ? "This is the start" : "Start here";
    }
  }

  // ------------------------------------------------------------------ chart
  function initChart() {
    const ctx = $("chart");
    const numbers = [...state.diagonals.keys()].sort((a, b) => a - b);
    const data = {
      labels: numbers.map(dname),
      datasets: [{
        data: numbers.map(() => 0),
        backgroundColor: numbers.map((d) => COLORS[family(d)]),
        borderWidth: 0,
        barPercentage: 0.9,
        categoryPercentage: 0.95,
      }],
    };
    if (chart) chart.destroy();
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const tick = dark ? "#a3ada5" : "#5e6a63";
    const grid = dark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.06)";
    chart = new Chart(ctx, {
      type: "bar",
      data,
      options: {
        animation: false,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => `${items[0].label} · ${family(numbers[items[0].dataIndex])}`,
              label: (item) => {
                const p = state.totals.per.get(numbers[item.dataIndex]) || { count: 0, area: 0 };
                return `${fmtArea(p.area)} · ${fmtInt(p.count)} crowns`;
              },
            },
          },
        },
        scales: {
          x: { ticks: { color: tick, font: { size: 9 }, maxRotation: 90, minRotation: 90, autoSkip: true, autoSkipPadding: 2 }, grid: { display: false } },
          y: { ticks: { color: tick, callback: (v) => fmtInt(v) }, grid: { color: grid }, title: { display: true, text: "Canopy (m²)", color: tick } },
        },
        onClick: (_event, elements) => {
          if (elements.length) selectDiagonal(numbers[elements[0].index], true);
        },
      },
    });
    chart.$numbers = numbers;
    updateChart();
  }

  function updateChart() {
    if (!chart) return;
    const numbers = chart.$numbers;
    chart.data.datasets[0].data = numbers.map((d) => (state.totals.per.get(d) || { area: 0 }).area);
    chart.data.datasets[0].backgroundColor = numbers.map((d) => (d === state.selectedDiagonal ? "#f2b600" : COLORS[family(d)]));
    chart.update("none");
    const t = state.totals;
    $("chart-totals").innerHTML = `EVEN <b>${fmtArea(t.fam.EVEN.area)}</b> · ODD <b>${fmtArea(t.fam.ODD.area)}</b> · All <b>${fmtArea(t.all.area)}</b>`;
  }

  // ------------------------------------------------------------------ selection
  function selectCrown(id, zoom = false) {
    state.selected = id;
    state.selectedDiagonal = null;
    drawSelection();
    restyleCrowns();
    refreshAll();
    const c = state.crowns.get(id);
    if (zoom && c) map.fitBounds(L.latLngBounds([c.bbox[1], c.bbox[0]], [c.bbox[3], c.bbox[2]]).pad(2.5), { maxZoom: map.getZoom() > state.fitZoom + 3 ? map.getZoom() : state.fitZoom + 4 });
  }

  function selectDiagonal(number, zoom = false) {
    state.selectedDiagonal = number;
    state.selected = null;
    drawSelection();
    restyleCrowns();
    refreshAll();
    const d = state.diagonals.get(number);
    if (zoom && d && d.pieces.length) map.fitBounds(L.latLngBounds(d.pieces.flat().map(toLL)).pad(0.15));
    const row = $("diag-table").querySelector(`tr[data-d="${number}"]`);
    if (row) row.scrollIntoView({ block: "nearest" });
  }

  function clearSelection() {
    if (state.selected === null && state.selectedDiagonal === null) return;
    state.selected = null;
    state.selectedDiagonal = null;
    drawSelection();
    restyleCrowns();
    refreshAll();
  }

  function onCrownClick(c, e) {
    if (state.mode === "draw") return;
    L.DomEvent.stopPropagation(e);
    if (state.mode === "merge") {
      const keep = state.crowns.get(state.mergeFrom);
      if (!keep || c.id === keep.id) { setMode("select"); return; }
      if (c.excluded) { toast("That crown is marked “not a pecan”. Count it again first.", "error"); return; }
      mergeCrowns(keep, c);
      setMode("select");
      return;
    }
    selectCrown(c.id);
  }

  function onDiagonalClick(d, e) {
    if (state.mode !== "select") return;
    L.DomEvent.stopPropagation(e);
    selectDiagonal(d.number);
  }

  function onMapClick(e) {
    if (state.mode === "draw") { addDrawPoint(e.latlng); return; }
    if (state.mode === "merge") { setMode("select"); return; }
    clearSelection();
  }

  // ------------------------------------------------------------------ modes
  function setMode(mode) {
    if (state.mode === "draw" && mode !== "draw") cancelDraw();
    state.mode = mode;
    document.body.classList.toggle("drawing", mode === "draw" || mode === "merge");
    for (const b of document.querySelectorAll(".mode")) b.classList.toggle("active", b.dataset.mode === (mode === "merge" ? "select" : mode));
    const hint = $("mode-hint");
    hint.classList.toggle("busy", mode !== "select");
    if (mode === "draw") {
      map.doubleClickZoom.disable();
      hint.textContent = "Click around the missed crown. Double-click (or Enter) to finish, Esc to cancel.";
    } else {
      map.doubleClickZoom.enable();
      hint.textContent = mode === "merge"
        ? `Click the crown to merge into ${state.mergeFrom}. Esc cancels.`
        : "Click a crown or a line on the map.";
    }
  }

  // ------------------------------------------------------------------ draw a missed crown
  const draw = { points: [], line: null };

  function addDrawPoint(latlng) {
    const p = [latlng.lng, latlng.lat];
    const last = draw.points[draw.points.length - 1];
    if (last && Math.hypot(p[0] - last[0], p[1] - last[1]) < 0.15) return;
    draw.points.push(p);
    layers.draw.clearLayers();
    const ll = draw.points.map(toLL);
    L.polyline(ll.concat(ll.length > 2 ? [ll[0]] : []), { renderer: renderers.select, pane: "select", color: SELECT_COLOR, weight: 2.5, dashArray: "6 4", interactive: false }).addTo(layers.draw);
    for (const q of ll) L.circleMarker(q, { renderer: renderers.select, pane: "select", radius: 3.5, color: "#111", weight: 1, fillColor: SELECT_COLOR, fillOpacity: 1, interactive: false }).addTo(layers.draw);
  }

  function cancelDraw() {
    draw.points = [];
    layers.draw.clearLayers();
  }

  function finishDraw() {
    const pts = draw.points.slice();
    cancelDraw();
    if (pts.length < 3) { toast("Click at least three points around the crown."); return; }
    const ring = pts.concat([pts[0]]);
    let polys = [[ring]];
    const box = polysBBox(polys);
    const clips = [];
    for (const c of state.crowns.values()) if (!c.removed && bboxHit(c.bbox, box)) clips.push(c.polys);
    try {
      if (clips.length) polys = polygonClipping.difference(polys, ...clips);
    } catch (err) {
      console.warn(err);
    }
    const area = polysArea(polys);
    if (!polys.length || area < 0.5) { toast("That area is already covered by other crowns."); return; }
    const centre = polysCentroid(polys);
    const diagonal = diagonalAt(centre) ?? nearestDiagonal(centre);
    state.addedCount = Math.max(state.addedCount, ...[...state.crowns.keys()].filter((k) => /^A\d+$/.test(k)).map((k) => +k.slice(1)), 0) + 1;
    const id = "A" + String(state.addedCount).padStart(4, "0");
    commit({ t: "add", id, polys, area, diagonal });
    setMode("select");
    selectCrown(id);
    toast(`Added ${id}: ${fmt1(area)} m² on ${dname(diagonal)} (${family(diagonal)}). Use ‹ › to change its diagonal.`, "", 5000);
  }

  // Diagonal of a planting position, from the planting grid: the diagonal
  // number grows by one every diagonal spacing across the diagonals; the 8
  // nearest trees give the local offset (rows may bend).
  function diagonalAt(p) {
    const lat = state.lattice;
    if (!lat || !lat.trees.length) return null;
    const o = p[0] * lat.n[0] + p[1] * lat.n[1];
    const near = lat.trees
      .map((t) => [Math.hypot(t.x - p[0], t.y - p[1]), t])
      .sort((a, b) => a[0] - b[0])
      .slice(0, 8)
      .map(([, t]) => t.d + (lat.sign * (o - t.o)) / lat.pitch)
      .sort((a, b) => a - b);
    const mid = near.length >> 1;
    const estimate = near.length % 2 ? near[mid] : (near[mid - 1] + near[mid]) / 2;
    const d = Math.round(estimate);
    return state.diagonals.has(d) ? d : null;
  }

  function buildLattice(settings) {
    if (!Number.isFinite(settings.diagonal_bearing_deg) || !(settings.diagonal_spacing_m > 0)) return null;
    const b = (settings.diagonal_bearing_deg * Math.PI) / 180;
    const n = [-Math.cos(b), Math.sin(b)];
    const trees = [];
    for (const c of state.crowns.values()) {
      const p = c.props;
      if (c.origDiagonal === null || p.grid_row === null || p.grid_row === undefined) continue;
      let x, y;
      if (Number.isFinite(Number(p.top_x)) && p.top_x !== null) { x = p.top_x - state.origin[0]; y = p.top_y - state.origin[1]; }
      else [x, y] = polysCentroid(c.polys);
      trees.push({ x, y, d: c.origDiagonal, o: x * n[0] + y * n[1] });
    }
    if (trees.length < 3) return null;
    const md = trees.reduce((s, t) => s + t.d, 0) / trees.length;
    const mo = trees.reduce((s, t) => s + t.o, 0) / trees.length;
    const cov = trees.reduce((s, t) => s + (t.d - md) * (t.o - mo), 0);
    return { n, pitch: settings.diagonal_spacing_m, sign: cov >= 0 ? 1 : -1, trees };
  }

  function nearestDiagonal(p) {
    let best = null, bestD = Infinity;
    for (const d of state.diagonals.values()) {
      for (const piece of d.pieces) {
        for (let i = 1; i < piece.length; i++) {
          const dist = segDist2(p, piece[i - 1], piece[i]);
          if (dist < bestD) { bestD = dist; best = d.number; }
        }
      }
    }
    return best ?? 0;
  }

  // ------------------------------------------------------------------ edits (undo / redo / save)
  function applyEdit(e) {
    switch (e.t) {
      case "reassign": {
        const c = state.crowns.get(e.id);
        c.diagonal = e.to;
        if (c.layer) c.layer.setStyle(crownStyle(c));
        break;
      }
      case "exclude": {
        const c = state.crowns.get(e.id);
        c.excluded = e.value;
        if (c.layer) c.layer.setStyle(crownStyle(c));
        break;
      }
      case "merge": {
        const keep = state.crowns.get(e.keep), drop = state.crowns.get(e.drop);
        keep.polys = e.polys;
        keep.bbox = polysBBox(e.polys);
        keep.area = e.areaAfter;
        keep.members = keep.members.concat([e.drop]);
        drop.removed = true;
        if (drop.layer) { layers.crowns.removeLayer(drop.layer); drop.layer = null; }
        if (keep.layer) { keep.layer.setLatLngs(polysToLL(keep.polys)); keep.layer.setStyle(crownStyle(keep)); }
        break;
      }
      case "add": {
        const c = {
          id: e.id, props: {}, geomWgs: null, polys: e.polys, bbox: polysBBox(e.polys), area: e.area,
          diagonal: e.diagonal, origDiagonal: null, excluded: false, removed: false, source: "added", members: [], layer: null,
        };
        state.crowns.set(e.id, c);
        addCrownLayer(c);
        break;
      }
      case "delete": {
        const c = state.crowns.get(e.id);
        c.removed = true;
        if (c.layer) { layers.crowns.removeLayer(c.layer); c.layer = null; }
        if (state.selected === e.id) state.selected = null;
        break;
      }
      default:
        throw new Error("Unknown correction " + e.t);
    }
  }

  function revertEdit(e) {
    switch (e.t) {
      case "reassign": {
        const c = state.crowns.get(e.id);
        c.diagonal = e.from;
        if (c.layer) c.layer.setStyle(crownStyle(c));
        break;
      }
      case "exclude": {
        const c = state.crowns.get(e.id);
        c.excluded = !e.value;
        if (c.layer) c.layer.setStyle(crownStyle(c));
        break;
      }
      case "merge": {
        const keep = state.crowns.get(e.keep), drop = state.crowns.get(e.drop);
        keep.polys = e.prevPolys;
        keep.bbox = polysBBox(e.prevPolys);
        keep.area = e.prevArea;
        keep.members = keep.members.filter((m) => m !== e.drop);
        drop.removed = false;
        addCrownLayer(drop);
        if (keep.layer) { keep.layer.setLatLngs(polysToLL(keep.polys)); keep.layer.setStyle(crownStyle(keep)); }
        break;
      }
      case "add": {
        const c = state.crowns.get(e.id);
        if (c && c.layer) layers.crowns.removeLayer(c.layer);
        state.crowns.delete(e.id);
        if (state.selected === e.id) state.selected = null;
        break;
      }
      case "delete": {
        const c = state.crowns.get(e.id);
        c.removed = false;
        addCrownLayer(c);
        break;
      }
      default:
        break;
    }
  }

  function persist() {
    if (!state.key) return;
    if (!state.history.length && state.start === defaultStart()) { storageRemove(state.key); return; }
    if (!storageSet(state.key, { edits: state.history, start: state.start, saved: new Date().toISOString() })) {
      toast("Could not save corrections in this browser (storage full or blocked). Export them to keep them.", "error", 6000);
    }
  }

  function defaultStart() {
    const s = state.summary && state.summary.start && Number(state.summary.start.diagonal);
    return Number.isFinite(s) ? s : 2;
  }

  function commit(edit) {
    applyEdit(edit);
    state.history.push(edit);
    state.future = [];
    persist();
    drawSelection();
    refreshAll();
  }

  function undo() {
    const e = state.history.pop();
    if (!e) return;
    revertEdit(e);
    state.future.push(e);
    persist();
    drawSelection();
    refreshAll();
    toast("Undone: " + describe(e));
  }

  function redo() {
    const e = state.future.pop();
    if (!e) return;
    applyEdit(e);
    state.history.push(e);
    persist();
    drawSelection();
    refreshAll();
    toast("Redone: " + describe(e));
  }

  // Two clicks: the first arms the button, the second resets.
  function resetAll() {
    if (!state.history.length) return;
    const button = $("btn-reset");
    if (!button.dataset.armed) {
      button.dataset.armed = "1";
      button.textContent = `Click again to remove ${state.history.length}`;
      clearTimeout(resetAll.timer);
      resetAll.timer = setTimeout(() => { delete button.dataset.armed; button.textContent = "Reset all"; }, 4000);
      return;
    }
    clearTimeout(resetAll.timer);
    delete button.dataset.armed;
    button.textContent = "Reset all";
    while (state.history.length) revertEdit(state.history.pop());
    state.future = [];
    persist();
    drawSelection();
    refreshAll();
    toast("All corrections removed.");
  }

  function describe(e) {
    switch (e.t) {
      case "reassign": return `${e.id} moved from ${dname(e.from)} to ${dname(e.to)}`;
      case "exclude": return `${e.id} ${e.value ? "marked not a pecan" : "counted as pecan again"}`;
      case "merge": return `${e.drop} merged into ${e.keep}`;
      case "add": return `${e.id} added on ${dname(e.diagonal)}`;
      case "delete": return `${e.id} deleted`;
      default: return e.t;
    }
  }

  function reassign(c, to) {
    if (!state.diagonals.has(to) || to === c.diagonal) return;
    commit({ t: "reassign", id: c.id, from: c.diagonal, to });
  }

  function mergeCrowns(keep, drop) {
    let polys;
    try {
      polys = polygonClipping.union(keep.polys, drop.polys);
    } catch (err) {
      console.error(err);
      toast("These two crowns could not be merged.", "error");
      return;
    }
    commit({ t: "merge", keep: keep.id, drop: drop.id, polys, prevPolys: keep.polys, prevArea: keep.area, areaAfter: keep.area + drop.area });
    selectCrown(keep.id);
    toast(`${drop.id} merged into ${keep.id}: ${fmt1(keep.area)} m² on ${dname(keep.diagonal)}.`);
  }

  // ------------------------------------------------------------------ export
  function activeCrowns() {
    return [...state.crowns.values()].filter((c) => !c.removed);
  }

  function exportDiagonals() {
    const t = state.totals;
    const lines = [csvRow(["diagonal", "family", "crowns", "canopy_m2"])];
    for (const [d, p] of [...t.per.entries()].filter(([d]) => state.diagonals.has(d)).sort((a, b) => a[0] - b[0])) {
      lines.push(csvRow([dname(d), family(d), p.count, p.area.toFixed(2)]));
    }
    lines.push("");
    lines.push(csvRow(["EVEN total", "EVEN", t.fam.EVEN.count, t.fam.EVEN.area.toFixed(2)]));
    lines.push(csvRow(["ODD total", "ODD", t.fam.ODD.count, t.fam.ODD.area.toFixed(2)]));
    lines.push(csvRow(["ALL total", "EVEN + ODD", t.all.count, t.all.area.toFixed(2)]));
    if (t.excluded.count) lines.push(csvRow(["Not a pecan (not counted)", "", t.excluded.count, t.excluded.area.toFixed(2)]));
    download(fileBase() + "_canopy_per_diagonal.csv", lines.join("\n") + "\n");
  }

  function exportCrownsCsv() {
    const lines = [csvRow(["crown_id", "diagonal", "family", "canopy_m2", "status", "source", "notebook_diagonal", "merged_pieces", "height_top_m", "top_x", "top_y"])];
    for (const c of activeCrowns()) {
      const p = c.props || {};
      lines.push(csvRow([
        c.id, dname(c.diagonal), family(c.diagonal), c.area.toFixed(2), c.excluded ? "not_pecan" : "pecan", c.source,
        c.origDiagonal === null ? "" : dname(c.origDiagonal), c.members.join(" "),
        p.height_top_m ?? "", p.top_x ?? "", p.top_y ?? "",
      ]));
    }
    download(fileBase() + "_crowns_corrected.csv", lines.join("\n") + "\n");
  }

  // WGS84 when the image is georeferenced; otherwise local metres.
  function polysToWgs(polys) {
    const coords = polys.map((poly) => poly.map((ring) => ring.map(([x, y]) => {
      if (!state.toLonLat) return [+x.toFixed(3), +y.toFixed(3)];
      const [lon, lat] = state.toLonLat(x, y);
      return [+lon.toFixed(8), +lat.toFixed(8)];
    })));
    return coords.length === 1 ? { type: "Polygon", coordinates: coords[0] } : { type: "MultiPolygon", coordinates: coords };
  }

  function exportGeoJson() {
    const features = activeCrowns().map((c) => ({
      type: "Feature",
      geometry: c.source === "model" && !c.members.length && c.geomWgs ? c.geomWgs : polysToWgs(c.polys),
      properties: {
        ...c.props,
        crown_id: c.id,
        diagonal: c.diagonal,
        family: family(c.diagonal),
        area_m2: +c.area.toFixed(3),
        status: c.excluded ? "not_pecan" : "pecan",
        source: c.source,
        notebook_diagonal: c.origDiagonal,
        merged_pieces: c.members.join(" ") || null,
        corrected: isEdited(c) || c.excluded,
      },
    }));
    download(fileBase() + "_crowns_corrected_wgs84.geojson", JSON.stringify({ type: "FeatureCollection", features }), "application/geo+json");
  }

  function exportLog() {
    const lines = [csvRow(["n", "correction", "crown", "details"])];
    state.history.forEach((e, i) => lines.push(csvRow([i + 1, e.t, e.id || e.keep || "", describe(e)])));
    lines.push(csvRow(["", "start", "", `Thinning starts on ${dname(state.start)} (EVEN)`]));
    download(fileBase() + "_corrections_log.csv", lines.join("\n") + "\n");
  }

  function fileBase() {
    return (state.name || "pecan").replace(/[^\w.-]+/g, "_");
  }

  // ------------------------------------------------------------------ wiring
  function wire() {
    const input = $("file-input");
    $("btn-open").addEventListener("click", () => input.click());
    $("btn-open-2").addEventListener("click", () => input.click());
    input.addEventListener("change", () => { openFiles(input.files); input.value = ""; });

    let depth = 0;
    window.addEventListener("dragenter", (e) => { if (hasFiles(e)) { e.preventDefault(); depth++; $("drag-overlay").hidden = false; } });
    window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
    window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) $("drag-overlay").hidden = true; });
    window.addEventListener("drop", (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      $("drag-overlay").hidden = true;
      openFiles(e.dataTransfer.files);
    });

    const exportBtn = $("btn-export"), menu = $("export-menu");
    exportBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      menu.hidden = !menu.hidden;
      exportBtn.setAttribute("aria-expanded", String(!menu.hidden));
    });
    document.addEventListener("click", () => { menu.hidden = true; exportBtn.setAttribute("aria-expanded", "false"); });
    menu.addEventListener("click", (e) => {
      const kind = e.target.dataset.export;
      if (!kind) return;
      if (kind === "diagonals") exportDiagonals();
      if (kind === "crowns-csv") exportCrownsCsv();
      if (kind === "plan") exportPlan();
      if (kind === "crowns-geojson") exportGeoJson();
      if (kind === "log") exportLog();
      if (kind === "print") window.print();
    });

    $("btn-help").addEventListener("click", () => $("help-modal").showModal());
    $("link-about").addEventListener("click", (e) => { e.preventDefault(); $("about-modal").showModal(); });

    for (const id of ["layer-image", "layer-crowns", "layer-even", "layer-odd", "layer-labels"]) $(id).addEventListener("change", applyLineVisibility);
    $("fill-opacity").addEventListener("input", (e) => { state.fillOpacity = +e.target.value; restyleCrowns(); });

    $("plan-family").addEventListener("change", fillPlan);
    $("btn-plan-export").addEventListener("click", exportPlan);
    $("checks-list").addEventListener("click", (e) => {
      if (e.target.dataset && e.target.dataset.check === "Show") showNextConflict();
    });

    $("start-select").addEventListener("change", (e) => setStart(+e.target.value));
    $("btn-zoom-start").addEventListener("click", () => {
      const d = state.diagonals.get(state.start);
      if (d && d.pieces.length) map.fitBounds(L.latLngBounds(d.pieces.flat().map(toLL)).pad(0.25));
    });
    $("toggle-start-only").addEventListener("change", (e) => {
      state.startOnly = e.target.checked;
      restyleCrowns();
      applyLineVisibility();
    });

    for (const b of document.querySelectorAll(".mode")) b.addEventListener("click", () => setMode(b.dataset.mode));

    $("sel-prev").addEventListener("click", () => { const c = state.crowns.get(state.selected); if (c) reassign(c, c.diagonal - 1); });
    $("sel-next").addEventListener("click", () => { const c = state.crowns.get(state.selected); if (c) reassign(c, c.diagonal + 1); });
    $("sel-diagonal").addEventListener("change", (e) => { const c = state.crowns.get(state.selected); if (c) reassign(c, +e.target.value); });
    $("sel-exclude").addEventListener("click", () => {
      const c = state.crowns.get(state.selected);
      if (c) commit({ t: "exclude", id: c.id, value: !c.excluded });
    });
    $("sel-merge").addEventListener("click", () => {
      if (!state.selected) return;
      state.mergeFrom = state.selected;
      setMode("merge");
    });
    $("sel-delete").addEventListener("click", () => {
      const c = state.crowns.get(state.selected);
      if (c && c.source === "added") commit({ t: "delete", id: c.id });
    });
    $("sel-zoom").addEventListener("click", () => { if (state.selected) selectCrown(state.selected, true); });
    $("seld-zoom").addEventListener("click", () => { if (state.selectedDiagonal !== null) selectDiagonal(state.selectedDiagonal, true); });
    $("seld-start").addEventListener("click", () => { if (state.selectedDiagonal !== null) setStart(state.selectedDiagonal); });

    $("btn-undo").addEventListener("click", undo);
    $("btn-redo").addEventListener("click", redo);
    $("btn-reset").addEventListener("click", resetAll);

    $("diag-table").tBodies[0].addEventListener("click", (e) => {
      const tr = e.target.closest("tr[data-d]");
      if (tr) selectDiagonal(+tr.dataset.d, true);
    });

    $("search").addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      const q = e.target.value.trim().toUpperCase();
      if (!q || !state.loaded) return;
      const dm = /^D?0*(\d+)$/.exec(q);
      if (dm && state.diagonals.has(+dm[1])) { selectDiagonal(+dm[1], true); return; }
      const hit = state.crowns.get(q) || [...state.crowns.values()].find((c) => c.id.toUpperCase() === q);
      if (hit && !hit.removed) { selectCrown(hit.id, true); return; }
      toast(`Nothing called “${e.target.value.trim()}”. Try D35 or a crown ID like C01234.`);
    });

    document.addEventListener("keydown", (e) => {
      if (!state.loaded) return;
      const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement && document.activeElement.tagName);
      if (e.key === "Escape") {
        if (state.mode !== "select") setMode("select");
        else clearSelection();
      } else if (e.key === "Enter" && state.mode === "draw" && !typing) {
        finishDraw();
      } else if (e.key === "Backspace" && state.mode === "draw" && !typing && draw.points.length) {
        e.preventDefault();
        draw.points.pop();
        const pts = draw.points.slice();
        cancelDraw();
        for (const p of pts) addDrawPoint(L.latLng(p[1], p[0]));
      } else if ((e.ctrlKey || e.metaKey) && !typing && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) redo(); else undo();
      } else if ((e.ctrlKey || e.metaKey) && !typing && e.key.toLowerCase() === "y") {
        e.preventDefault();
        redo();
      }
    });
  }

  function hasFiles(e) {
    return e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
  }

  function setStart(number) {
    if (!state.diagonals.has(number) || family(number) !== "EVEN") return;
    state.start = number;
    $("start-select").value = String(number);
    drawStart();
    restyleCrowns();
    persist();
    refreshAll();
    toast(`Thinning starts on ${dname(number)} (EVEN).`);
  }

  // ------------------------------------------------------------------ start
  async function boot() {
    initMap();
    wire();
    const params = new URLSearchParams(location.search);
    const url = params.get("data");
    if (url) {
      try {
        setLoading("Downloading results …");
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Could not download ${url} (${res.status}).`);
        await loadZip(await res.blob(), decodeURIComponent(url.split("/").pop() || "results.zip"));
      } catch (err) {
        console.error(err);
        setLoading(null);
        toast(err.message || String(err), "error", 7000);
      }
    }
  }

  window.PecanDashboard = {
    state, openFiles, selectCrown, selectDiagonal, undo, redo, setStart, exportDiagonals, exportPlan, rowConflicts,
    reassign: (id, to) => reassign(state.crowns.get(id), to),
    diagonalAt: (x, y) => diagonalAt([x - state.origin[0], y - state.origin[1]]),
  };
  boot();
})();
