/*
 * Emmanuel Dashboard for Pecan Thinning: image analysis in the browser.
 *
 * Same method as the pecan canopy notebook:
 *   1. colour + multi-scale texture features;
 *   2. training pixels harvested from the image itself (canopy, ground
 *      vegetation, soil / road, shadow) and a classifier for P(canopy);
 *   3. with an elevation file (DSM): canopy height = DSM - terrain; only
 *      canopy at least 1 m tall is a tree (grass and weeds go), crown edges
 *      from the image;
 *   4. clear cast shadow trimmed from the crown edges (gentle);
 *   5. crowns split by a marker-controlled watershed (treetops from the DSM,
 *      or crown centres with the image only);
 *   6. planting grid measured from the treetops; grid position of every
 *      crown; one crown per planting position (a crown is never cut in two);
 *   7. young (replanted) trees: a small, round, sunlit crown as green as the
 *      pecan leaves, standing on an EMPTY planting position on bare ground,
 *      becomes a crown (flag YOUNG_TREE) although it is under 2 m² / 1 m;
 *      then every empty position of the 40 x 20 ft grid is searched for a
 *      sapling (an island of leaves on the tree strip), counted only when
 *      the measured grid matches the design and the same search finds far
 *      fewer lone spots halfway between positions (planted saplings, not
 *      weeds); a run of young trees halfway between positions is a row
 *      planted half a spacing off and is counted too; then, where the
 *      original image is given (opts.chip), the positions still empty are
 *      looked at in full resolution: a young crown of at least 0.3 m² with
 *      its own dark tree shadow beside it (weeds and grass lie flat);
 *   8. EVEN / ODD diagonals as wavy lines through the crowns, one unbroken
 *      line per diagonal (also across white areas), ending in the direction
 *      of the diagonal, never crossing each other, keeping off canopy of the
 *      other colour wherever there is a free path; along every row
 *      neighbouring crowns always alternate (1 blue, 2 red ...), also across
 *      a missing tree.
 * Input: RGBA pixels on a square grid of `pixel` metres (north up), and
 * optionally `surface`: DSM elevation on the same grid (NaN = no data).
 */
(() => {
  "use strict";

  // ------------------------------------------------------------------ rasters
  function boxesForGauss(sigma, n = 3) {
    const ideal = Math.sqrt((12 * sigma * sigma) / n + 1);
    let wl = Math.floor(ideal);
    if (wl % 2 === 0) wl--;
    const wu = wl + 2;
    const m = Math.round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4));
    const sizes = [];
    for (let i = 0; i < n; i++) sizes.push(i < m ? wl : wu);
    return sizes;
  }

  function boxPass(src, dst, w, h, r, horizontal) {
    if (r <= 0) { dst.set(src); return; }
    const norm = 1 / (2 * r + 1);
    if (horizontal) {
      for (let y = 0; y < h; y++) {
        const o = y * w;
        let acc = 0;
        for (let k = -r; k <= r; k++) acc += src[o + Math.min(w - 1, Math.max(0, k))];
        for (let x = 0; x < w; x++) {
          dst[o + x] = acc * norm;
          acc += src[o + Math.min(w - 1, x + r + 1)] - src[o + Math.max(0, x - r)];
        }
      }
    } else {
      const acc = new Float64Array(w);
      for (let k = -r; k <= r; k++) {
        const o = Math.min(h - 1, Math.max(0, k)) * w;
        for (let x = 0; x < w; x++) acc[x] += src[o + x];
      }
      for (let y = 0; y < h; y++) {
        const o = y * w;
        const add = Math.min(h - 1, y + r + 1) * w, sub = Math.max(0, y - r) * w;
        for (let x = 0; x < w; x++) {
          dst[o + x] = acc[x] * norm;
          acc[x] += src[add + x] - src[sub + x];
        }
      }
    }
  }

  function gauss(src, w, h, sigma) {
    const out = Float32Array.from(src);
    if (!(sigma >= 0.35)) return out;
    const tmp = new Float32Array(src.length);
    for (const size of boxesForGauss(sigma)) {
      const r = (size - 1) >> 1;
      boxPass(out, tmp, w, h, r, true);
      boxPass(tmp, out, w, h, r, false);
    }
    return out;
  }

  function maskedGauss(values, weight, w, h, sigma, weightBlur) {
    const n = values.length;
    const prod = new Float32Array(n);
    for (let i = 0; i < n; i++) prod[i] = values[i] * weight[i];
    const num = gauss(prod, w, h, sigma);
    const den = weightBlur || gauss(weight, w, h, sigma);
    for (let i = 0; i < n; i++) num[i] = den[i] > 1e-6 ? num[i] / den[i] : 0;
    return num;
  }

  // Exact Euclidean distance (pixels) from each pixel with mask=1 to the
  // nearest pixel with mask=0 (Felzenszwalb & Huttenlocher).
  function edt(mask, w, h) {
    const INF = 1e20;
    const g = new Float64Array(w * h);
    for (let i = 0; i < w * h; i++) g[i] = mask[i] ? INF : 0;
    const n = Math.max(w, h);
    const f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
    const pass = (len) => {
      let k = 0;
      v[0] = 0; z[0] = -INF; z[1] = INF;
      for (let q = 1; q < len; q++) {
        let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
        while (s <= z[k]) { k--; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
        k++; v[k] = q; z[k] = s; z[k + 1] = INF;
      }
      k = 0;
      for (let q = 0; q < len; q++) {
        while (z[k + 1] < q) k++;
        d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
      }
    };
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) f[y] = g[y * w + x];
      pass(h);
      for (let y = 0; y < h; y++) g[y * w + x] = d[y];
    }
    const out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const o = y * w;
      for (let x = 0; x < w; x++) f[x] = g[o + x];
      pass(w);
      for (let x = 0; x < w; x++) out[o + x] = Math.sqrt(d[x]);
    }
    return out;
  }

  const not = (m) => { const o = new Uint8Array(m.length); for (let i = 0; i < m.length; i++) o[i] = m[i] ? 0 : 1; return o; };
  function erode(mask, w, h, r) {
    if (r < 0.5) return Uint8Array.from(mask);
    const d = edt(mask, w, h), o = new Uint8Array(mask.length);
    for (let i = 0; i < o.length; i++) o[i] = mask[i] && d[i] > r ? 1 : 0;
    return o;
  }
  function dilate(mask, w, h, r) {
    if (r < 0.5) return Uint8Array.from(mask);
    const d = edt(not(mask), w, h), o = new Uint8Array(mask.length);
    for (let i = 0; i < o.length; i++) o[i] = mask[i] || d[i] <= r ? 1 : 0;
    return o;
  }
  const opening = (m, w, h, r) => dilate(erode(m, w, h, r), w, h, r);
  const closing = (m, w, h, r) => erode(dilate(m, w, h, r), w, h, r);

  // 4-connected components. Returns { labels: Int32Array, count }.
  function label(mask, w, h) {
    const labels = new Int32Array(w * h);
    const parent = [0];
    const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!mask[i]) continue;
        const up = y ? labels[i - w] : 0, left = x ? labels[i - 1] : 0;
        if (!up && !left) { parent.push(parent.length); labels[i] = parent.length - 1; }
        else if (up && left) {
          const a = find(up), b = find(left);
          labels[i] = Math.min(a, b);
          if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
        } else labels[i] = up || left;
      }
    }
    const remap = new Int32Array(parent.length);
    let count = 0;
    for (let k = 1; k < parent.length; k++) { const r = find(k); if (!remap[r]) remap[r] = ++count; remap[k] = remap[r]; }
    for (let i = 0; i < labels.length; i++) if (labels[i]) labels[i] = remap[labels[i]];
    return { labels, count };
  }

  function dropSmall(mask, w, h, minPixels) {
    const { labels, count } = label(mask, w, h);
    const size = new Int32Array(count + 1);
    for (let i = 0; i < labels.length; i++) size[labels[i]]++;
    const o = new Uint8Array(mask.length);
    for (let i = 0; i < o.length; i++) o[i] = labels[i] && size[labels[i]] >= minPixels ? 1 : 0;
    return o;
  }

  // Holes of the mask smaller than maxPixels are filled.
  function fillSmallHoles(mask, w, h, maxPixels) {
    const { labels, count } = label(not(mask), w, h);
    const size = new Int32Array(count + 1), edge = new Uint8Array(count + 1);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const l = labels[y * w + x];
      if (!l) continue;
      size[l]++;
      if (!x || !y || x === w - 1 || y === h - 1) edge[l] = 1;
    }
    const o = Uint8Array.from(mask);
    for (let i = 0; i < o.length; i++) { const l = labels[i]; if (l && !edge[l] && size[l] < maxPixels) o[i] = 1; }
    return o;
  }

  function sample(values, every, test) {
    const out = [];
    for (let i = 0; i < values.length; i += every) if (!test || test(i)) out.push(values[i]);
    return out;
  }
  function percentile(list, p) {
    if (!list.length) return 0;
    const s = Float64Array.from(list).sort();
    const k = Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))));
    return s[k];
  }
  function otsu(list) {
    let lo = Infinity, hi = -Infinity;
    for (const v of list) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!(hi > lo)) return lo;
    const bins = 256, hist = new Float64Array(bins);
    for (const v of list) hist[Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo)) * bins))]++;
    let total = list.length, sum = 0;
    for (let i = 0; i < bins; i++) sum += i * hist[i];
    let wB = 0, sumB = 0, best = 0, at = 0;
    for (let i = 0; i < bins; i++) {
      wB += hist[i];
      if (!wB) continue;
      const wF = total - wB;
      if (!wF) break;
      sumB += i * hist[i];
      const mB = sumB / wB, mF = (sum - sumB) / wF;
      const between = wB * wF * (mB - mF) * (mB - mF);
      if (between > best) { best = between; at = i; }
    }
    return lo + ((at + 0.5) / bins) * (hi - lo);
  }
  // Lowest point of the smoothed histogram between its two main modes.
  function histogramValley(list, low, high, bins = 60) {
    const counts = new Float64Array(bins);
    for (const v of list) if (v >= low && v <= high) counts[Math.min(bins - 1, Math.floor(((v - low) / (high - low)) * bins))]++;
    const smooth = new Float64Array(bins);
    for (let i = 0; i < bins; i++) { let s = 0; for (let k = -2; k <= 2; k++) { const j = i + k; if (j >= 0 && j < bins) s += counts[j]; } smooth[i] = s / 5; }
    const peaks = [];
    for (let i = 1; i < bins - 1; i++) if (smooth[i] >= smooth[i - 1] && smooth[i] >= smooth[i + 1]) peaks.push(i);
    if (peaks.length < 2) return null;
    const top = peaks.sort((a, b) => smooth[a] - smooth[b]).slice(-2).sort((a, b) => a - b);
    if (top[1] - top[0] < 3) return null;
    let at = top[0];
    for (let i = top[0]; i <= top[1]; i++) if (smooth[i] < smooth[at]) at = i;
    return low + ((at + 0.5) / bins) * (high - low);
  }

  // ------------------------------------------------------------------ canopy
  const FEATURES = ["Y", "exg", "astar", "bstar", "t025", "t050", "t100", "ll_m1", "ll_m3", "reldark", "exg_s05", "exg_s15", "bstar_s05", "astar_s05"];

  function features(rgba, valid, w, h, pixel) {
    const n = w * h;
    const lin = new Float32Array(256);
    for (let i = 0; i < 256; i++) { const c = i / 255; lin[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
    const F = {};
    for (const k of ["Y", "exg", "astar", "bstar"]) F[k] = new Float32Array(n);
    const logY = new Float32Array(n), weight = new Float32Array(n);
    const fl = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
    for (let i = 0; i < n; i++) {
      const R = rgba[4 * i], G = rgba[4 * i + 1], B = rgba[4 * i + 2];
      const tot = R + G + B + 1;
      F.exg[i] = (2 * G - R - B) / tot;
      const Y = 0.299 * R + 0.587 * G + 0.114 * B;
      F.Y[i] = Y;
      logY[i] = Math.log1p(Y);
      weight[i] = valid[i];
      const r = lin[R], g = lin[G], b = lin[B];
      const X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
      const YY = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
      const fx = fl(X), fy = fl(YY), fz = fl(Z);
      F.astar[i] = 500 * (fx - fy);
      F.bstar[i] = 200 * (fy - fz);
    }
    // Texture: local spread of log-brightness at three scales (at least ~1.5 px).
    const base = Math.max(0.25, 1.5 * pixel);
    const sq = new Float32Array(n);
    for (let i = 0; i < n; i++) sq[i] = logY[i] * logY[i];
    for (const [name, sigmaM] of [["t025", base], ["t050", 2 * base], ["t100", 4 * base]]) {
      const s = sigmaM / pixel;
      const wb = gauss(weight, w, h, s);
      const mean = maskedGauss(logY, weight, w, h, s, wb);
      const meanSq = maskedGauss(sq, weight, w, h, s, wb);
      const t = new Float32Array(n);
      for (let i = 0; i < n; i++) t[i] = Math.sqrt(Math.max(meanSq[i] - mean[i] * mean[i], 0));
      F[name] = t;
      if (name === "t100") F.ll_m1 = mean;
    }
    F.ll_m3 = maskedGauss(logY, weight, w, h, 3.0 / pixel);
    F.reldark = new Float32Array(n);
    for (let i = 0; i < n; i++) F.reldark[i] = F.ll_m3[i] - logY[i];
    F.exg_s05 = maskedGauss(F.exg, weight, w, h, 0.5 / pixel);
    F.exg_s15 = maskedGauss(F.exg, weight, w, h, 1.5 / pixel);
    F.bstar_s05 = maskedGauss(F.bstar, weight, w, h, 0.5 / pixel);
    F.astar_s05 = maskedGauss(F.astar, weight, w, h, 0.5 / pixel);
    return F;
  }

  function harvest(F, valid, w, h, pixel) {
    const n = w * h, area = pixel * pixel;
    const texture = F.t050, wide = F.t100, Y = F.Y, exgLocal = F.exg_s05;
    const ownExg = gauss(F.exg, w, h, 1.0), ownA = gauss(F.astar, w, h, 1.0);
    const greenLoose = new Uint8Array(n);
    for (let i = 0; i < n; i++) greenLoose[i] = valid[i] && exgLocal[i] > 0.05 ? 1 : 0;
    const tSample = sample(texture, 7, (i) => greenLoose[i]);
    if (tSample.length < 300) throw new Error("Too few green pixels to learn the pecan canopy. Is this an RGB image of a pecan orchard?");
    const tOtsu = otsu(tSample);
    let tValley = histogramValley(tSample, 0, 0.8);
    if (tValley === null || !(tValley > 0.05 && tValley < tOtsu)) tValley = 0.5 * tOtsu;
    const fieldY = sample(Y, 3, (i) => valid[i]);
    const darkY = percentile(fieldY, 30), brightY = percentile(fieldY, 50);
    let darkValley = histogramValley(sample(ownExg, 5, (i) => valid[i] && Y[i] < darkY), -0.1, 0.4);
    if (darkValley === null || !(darkValley > 0.03 && darkValley < 0.25)) darkValley = 0.12;

    let tg = new Uint8Array(n);
    for (let i = 0; i < n; i++) tg[i] = greenLoose[i] && texture[i] > tValley ? 1 : 0;
    tg = opening(tg, w, h, 0.3 / pixel);
    tg = dropSmall(tg, w, h, Math.round(4.0 / area));
    const core = erode(tg, w, h, 0.6 / pixel);
    const toTexture = edt(not(tg), w, h);

    const seeds = new Int8Array(n).fill(-1);
    for (let i = 0; i < n; i++) {
      if (!valid[i]) continue;
      const ownGreen = ownExg[i] > darkValley + 0.01 && ownA[i] < -5;
      const ownNeutral = ownExg[i] < darkValley - 0.04 && ownA[i] > -4.5;
      if (ownExg[i] > 0.08 && exgLocal[i] > 0.06 && texture[i] < 0.85 * tValley && wide[i] < tOtsu && Y[i] > darkY && toTexture[i] * pixel > 1.0) seeds[i] = 1;
      if (ownExg[i] < 0.03 && Y[i] > brightY && !core[i]) seeds[i] = 2;
      if (Y[i] < darkY && ownNeutral && !core[i]) seeds[i] = 3;
      if (core[i] && ownGreen && texture[i] > tValley) seeds[i] = 0;
    }
    return { seeds, ownExg, thresholds: { tOtsu, tValley, darkY, brightY, darkValley } };
  }

  // Quadratic discriminant (one Gaussian per class, shrunk covariance).
  function trainQda(F, seeds, perClass = 25000) {
    const k = FEATURES.length;
    const cols = FEATURES.map((name) => F[name]);
    const classes = [];
    let seed = 12345;
    const rand = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    const picks = [[], [], [], []];
    for (let i = 0; i < seeds.length; i++) if (seeds[i] >= 0) picks[seeds[i]].push(i);
    if (picks[0].length < 50) throw new Error("No clear pecan crowns were found to learn from. Check that this is an RGB orchard image and the pixel size is right.");
    const all = [];
    for (let c = 0; c < 4; c++) {
      let list = picks[c];
      if (list.length < 30) continue;
      if (list.length > perClass) {
        for (let j = list.length - 1; j > 0; j--) { const r = Math.floor(rand() * (j + 1)); [list[j], list[r]] = [list[r], list[j]]; }
        list = list.slice(0, perClass);
      }
      classes.push({ c, idx: list });
      all.push(...list);
    }
    if (classes.length < 2) throw new Error("Could not separate canopy from the ground in this image.");
    const mean = new Float64Array(k), std = new Float64Array(k);
    for (const i of all) for (let f = 0; f < k; f++) mean[f] += cols[f][i];
    for (let f = 0; f < k; f++) mean[f] /= all.length;
    for (const i of all) for (let f = 0; f < k; f++) std[f] += (cols[f][i] - mean[f]) ** 2;
    for (let f = 0; f < k; f++) std[f] = Math.sqrt(std[f] / all.length) || 1;
    const total = all.length;
    const models = classes.map(({ c, idx }) => {
      const mu = new Float64Array(k), cov = new Float64Array(k * k), x = new Float64Array(k);
      for (const i of idx) for (let f = 0; f < k; f++) mu[f] += (cols[f][i] - mean[f]) / std[f];
      for (let f = 0; f < k; f++) mu[f] /= idx.length;
      for (const i of idx) {
        for (let f = 0; f < k; f++) x[f] = (cols[f][i] - mean[f]) / std[f] - mu[f];
        for (let a = 0; a < k; a++) for (let b = 0; b <= a; b++) cov[a * k + b] += x[a] * x[b];
      }
      for (let a = 0; a < k; a++) for (let b = 0; b <= a; b++) { cov[a * k + b] /= idx.length; cov[b * k + a] = cov[a * k + b]; }
      const lam = 0.1;
      for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) cov[a * k + b] = (a === b ? cov[a * k + a] + 1e-3 : (1 - lam) * cov[a * k + b]);
      const L = new Float64Array(k * k);  // Cholesky
      for (let a = 0; a < k; a++) {
        for (let b = 0; b <= a; b++) {
          let s = cov[a * k + b];
          for (let j = 0; j < b; j++) s -= L[a * k + j] * L[b * k + j];
          L[a * k + b] = a === b ? Math.sqrt(Math.max(s, 1e-9)) : s / L[b * k + b];
        }
      }
      let logdet = 0;
      for (let a = 0; a < k; a++) logdet += 2 * Math.log(L[a * k + a]);
      return { c, mu, L, bias: -0.5 * logdet + Math.log(idx.length / total) };
    });
    return { models, mean, std, cols, k };
  }

  function predictCanopy(model, valid, skip, n) {
    const { models, mean, std, cols, k } = model;
    const P = new Float32Array(n);
    const z = new Float64Array(k), y = new Float64Array(k), score = new Float64Array(models.length);
    const canopy = models.findIndex((m) => m.c === 0);
    for (let i = 0; i < n; i++) {
      if (!valid[i] || skip[i]) continue;
      for (let f = 0; f < k; f++) z[f] = (cols[f][i] - mean[f]) / std[f];
      let best = -Infinity;
      for (let m = 0; m < models.length; m++) {
        const { mu, L, bias } = models[m];
        let q = 0;
        for (let a = 0; a < k; a++) {
          let s = z[a] - mu[a];
          const row = a * k;
          for (let j = 0; j < a; j++) s -= L[row + j] * y[j];
          y[a] = s / L[row + a];
          q += y[a] * y[a];
        }
        score[m] = bias - 0.5 * q;
        if (score[m] > best) best = score[m];
      }
      let sum = 0;
      for (let m = 0; m < models.length; m++) sum += Math.exp(score[m] - best);
      P[i] = Math.exp(score[canopy] - best) / sum;
    }
    return P;
  }

  // ------------------------------------------------------------------ crowns
  function maxFilter(src, w, h, r) {
    const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
    const run = (get, set, len) => {
      // simple deque sliding max
      const q = new Int32Array(len);
      let head = 0, tail = 0;
      for (let i = 0; i < len + r; i++) {
        if (i < len) {
          while (tail > head && get(q[tail - 1]) <= get(i)) tail--;
          q[tail++] = i;
        }
        const c = i - r;
        if (c >= 0) {
          while (q[head] < c - r) head++;
          set(c, get(q[head]));
        }
      }
    };
    for (let y = 0; y < h; y++) { const o = y * w; run((i) => src[o + i], (i, v) => { tmp[o + i] = v; }, w); }
    for (let x = 0; x < w; x++) run((i) => tmp[i * w + x], (i, v) => { out[i * w + x] = v; }, h);
    return out;
  }

  function watershed(score, markers, mask, w, h) {
    const n = w * h;
    const labels = Int32Array.from(markers);
    let max = 0;
    for (let i = 0; i < n; i++) if (mask[i] && score[i] > max) max = score[i];
    const levels = 1024;
    const level = (i) => Math.min(levels - 1, Math.max(0, Math.floor(((max - score[i]) / (max || 1)) * (levels - 1))));
    const buckets = Array.from({ length: levels }, () => []);
    for (let i = 0; i < n; i++) if (labels[i]) buckets[level(i)].push(i);
    for (let b = 0; b < levels; b++) {
      const bucket = buckets[b];
      for (let t = 0; t < bucket.length; t++) {
        const i = bucket[t], x = i % w, l = labels[i];
        const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, i - w, i + w];
        for (const j of nb) {
          if (j < 0 || j >= n || !mask[j] || labels[j]) continue;
          labels[j] = l;
          buckets[Math.max(b, level(j))].push(j);
        }
      }
      buckets[b] = null;
    }
    return labels;
  }

  // Outer boundary (and holes) of one label inside its box, as polygons in
  // pixel-corner coordinates (x right, y down).
  function tracePolygons(labels, w, lab, box) {
    const [x0, y0, x1, y1] = box;
    const W = x1 - x0 + 1;
    const edges = new Map();
    const starts = [];
    const add = (ax, ay, bx, by) => {
      const key = ay * (W + 2) + ax;
      const e = { ax, ay, bx, by, used: false };
      if (!edges.has(key)) edges.set(key, []);
      edges.get(key).push(e);
      starts.push(e);
    };
    const at = (x, y) => x >= x0 && x <= x1 && y >= y0 && y <= y1 && labels[y * w + x] === lab;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        if (labels[y * w + x] !== lab) continue;
        const cx = x - x0, cy = y - y0;
        if (!at(x, y - 1)) add(cx, cy, cx + 1, cy);
        if (!at(x + 1, y)) add(cx + 1, cy, cx + 1, cy + 1);
        if (!at(x, y + 1)) add(cx + 1, cy + 1, cx, cy + 1);
        if (!at(x - 1, y)) add(cx, cy + 1, cx, cy);
      }
    }
    const loops = [];
    for (const first of starts) {
      if (first.used) continue;
      const pts = [];
      let e = first;
      while (e && !e.used) {
        e.used = true;
        pts.push([e.ax, e.ay]);
        const dx = e.bx - e.ax, dy = e.by - e.ay;
        const options = (edges.get(e.by * (W + 2) + e.bx) || []).filter((o) => !o.used);
        if (!options.length) break;
        if (options.length === 1) { e = options[0]; continue; }
        const pref = [[-dy, dx], [dx, dy], [dy, -dx]];  // right, straight, left (y down)
        e = null;
        for (const [px, py] of pref) {
          const hit = options.find((o) => o.bx - o.ax === px && o.by - o.ay === py);
          if (hit) { e = hit; break; }
        }
        if (!e) e = options[0];
      }
      if (pts.length < 4) continue;
      // drop collinear points
      const clean = [];
      for (let i = 0; i < pts.length; i++) {
        const a = pts[(i - 1 + pts.length) % pts.length], b = pts[i], c = pts[(i + 1) % pts.length];
        if ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) !== 0) clean.push([b[0] + x0, b[1] + y0]);
      }
      if (clean.length < 3) continue;
      let area = 0;
      for (let i = 0; i < clean.length; i++) { const [ax, ay] = clean[i], [bx, by] = clean[(i + 1) % clean.length]; area += ax * by - bx * ay; }
      loops.push({ pts: clean, area: area / 2 });
    }
    const outers = loops.filter((l) => l.area > 0);
    const holes = loops.filter((l) => l.area < 0);
    const polys = outers.map((o) => [o.pts]);
    for (const hole of holes) {
      const [px, py] = hole.pts[0];
      const owner = outers.findIndex((o) => inside(o.pts, px + 0.5, py + 0.5)) ;
      if (owner >= 0) polys[owner].push(hole.pts);
    }
    return polys;
  }

  function inside(ring, x, y) {
    let c = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
    }
    return c;
  }

  function simplifyRing(points, tol) {
    if (points.length <= 4) return points;
    const n = points.length;
    const keep = new Uint8Array(n);
    keep[0] = keep[n - 1] = 1;
    const stack = [[0, n - 1]];
    const t2 = tol * tol;
    const d2 = (p, a, b) => {
      const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy;
      let t = l2 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2 : 0;
      t = Math.max(0, Math.min(1, t));
      return (a[0] + t * dx - p[0]) ** 2 + (a[1] + t * dy - p[1]) ** 2;
    };
    while (stack.length) {
      const [a, b] = stack.pop();
      let best = 0, at = -1;
      for (let i = a + 1; i < b; i++) { const d = d2(points[i], points[a], points[b]); if (d > best) { best = d; at = i; } }
      if (best > t2) { keep[at] = 1; stack.push([a, at], [at, b]); }
    }
    const out = points.filter((_, i) => keep[i]);
    return out.length >= 3 ? out : points;
  }

  // ------------------------------------------------------------------ grid
  function pairsWithin(points, radius) {
    const cell = radius, grid = new Map();
    const key = (cx, cy) => cx * 73856093 ^ cy * 19349663;
    points.forEach(([x, y], i) => {
      const k = key(Math.floor(x / cell), Math.floor(y / cell));
      if (!grid.has(k)) grid.set(k, []);
      grid.get(k).push(i);
    });
    const pairs = [], r2 = radius * radius;
    points.forEach(([x, y], i) => {
      const cx = Math.floor(x / cell), cy = Math.floor(y / cell);
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
        const list = grid.get(key(cx + dx, cy + dy));
        if (!list) continue;
        for (const j of list) {
          if (j <= i) continue;
          const ex = points[j][0] - x, ey = points[j][1] - y;
          if (ex * ex + ey * ey <= r2) pairs.push([i, j]);
        }
      }
    });
    return pairs;
  }

  function nearestDistances(points) {
    let r = 4;
    for (let attempt = 0; attempt < 6; attempt++, r *= 2) {
      const best = new Float64Array(points.length).fill(Infinity);
      for (const [i, j] of pairsWithin(points, r)) {
        const d = Math.hypot(points[j][0] - points[i][0], points[j][1] - points[i][1]);
        if (d < best[i]) best[i] = d;
        if (d < best[j]) best[j] = d;
      }
      const finite = Array.from(best).filter(Number.isFinite);
      if (finite.length > 0.8 * points.length) return percentile(finite, 50);
    }
    return 6;
  }

  const len = (v) => Math.hypot(v[0], v[1]);
  const cross = (u, v) => Math.abs(u[0] * v[1] - u[1] * v[0]) / (len(u) * len(v));
  const inv2 = (a, b) => { const det = a[0] * b[1] - b[0] * a[1]; return [[b[1] / det, -b[0] / det], [-a[1] / det, a[0] / det]]; };

  // Distance between rows: treetops projected across the rows; the profile
  // repeats every row. Returns { P, at(m) }: P = the smallest distance that
  // repeats clearly (also at twice that distance), at(m) = how strongly a
  // distance m repeats (0..1). null: too few rows to tell.
  function rowPeriod(points, across, inRow) {
    const proj = points.map(([x, y]) => x * across[0] + y * across[1]);
    const lo = Math.min(...proj), hi = Math.max(...proj), bin = 0.25;
    const nb = Math.floor((hi - lo) / bin) + 1;
    if (nb < 40) return null;
    let hist = new Float32Array(nb);
    for (const v of proj) hist[Math.floor((v - lo) / bin)]++;
    hist = gauss(hist, nb, 1, 0.6 / bin);
    let mean = 0;
    for (const v of hist) mean += v;
    mean /= nb;
    const acRaw = (lag) => { let s = 0; for (let i = 0; i + lag < nb; i++) s += (hist[i] - mean) * (hist[i + lag] - mean); return s / (nb - lag); };
    const ac0 = acRaw(0);
    if (!(ac0 > 0)) return null;
    const maxLag = Math.min(nb - 1, Math.floor(80 / bin));
    const vals = new Float32Array(maxLag + 1);
    for (let lag = 0; lag <= maxLag; lag++) vals[lag] = acRaw(lag) / ac0;
    const at = (m, tol = 0.04) => {
      let best = -1;
      for (let l = Math.round((m * (1 - tol)) / bin); l <= Math.round((m * (1 + tol)) / bin); l++) if (l >= 0 && l <= maxLag) best = Math.max(best, vals[l]);
      return best;
    };
    let P = null;
    for (let l = Math.ceil((1.2 * inRow) / bin); l < Math.min(maxLag, Math.floor(40 / bin)); l++) {
      if (!(vals[l] >= 0.35 && vals[l] >= vals[l - 1] && vals[l] >= vals[l + 1])) continue;
      const twice = 2 * l <= maxLag ? at(2 * l * bin, 0.08) : vals[l];
      if (twice >= 0.35) { P = l * bin; break; }
    }
    return { P, at };
  }

  // How clearly the treetops form rows running at angle ang (radians):
  // projected across the rows they pile up every row spacing (lag within
  // 0.7 to 1.4 times the design distance), and not at half of it (that
  // would be the trees within a row, i.e. columns).
  function rowScore(points, ang, between) {
    const across = [-Math.sin(ang), Math.cos(ang)], bin = 0.25;
    const proj = points.map(([x, y]) => x * across[0] + y * across[1]);
    let lo = Infinity, hi = -Infinity;
    for (const v of proj) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const nb = Math.floor((hi - lo) / bin) + 1;
    if (nb < 40) return { score: 0, lag: 0 };
    let hist = new Float32Array(nb);
    for (const v of proj) hist[Math.floor((v - lo) / bin)]++;
    hist = gauss(hist, nb, 1, 0.6 / bin);
    let mean = 0;
    for (const v of hist) mean += v;
    mean /= nb;
    const ac = (lag) => { let s = 0; for (let i = 0; i + lag < nb; i++) s += (hist[i] - mean) * (hist[i + lag] - mean); return s / (nb - lag); };
    const ac0 = ac(0);
    if (!(ac0 > 0)) return { score: 0, lag: 0 };
    let best = { score: -Infinity, lag: 0 };
    for (let l = Math.round((0.7 * between) / bin); l <= Math.min(nb - 1, Math.round((1.4 * between) / bin)); l++) {
      const v = ac(l) / ac0, half = ac(Math.round(l / 2)) / ac0;
      const sc = v - Math.max(0, half);
      if (sc > best.score) best = { score: sc, lag: l * bin };
    }
    return best;
  }
  // The direction of the rows: the angle at which the treetops form rows
  // most clearly (0.5° steps, then refined).
  function bestRowDirection(points, between) {
    let best = { score: -Infinity, ang: 0 };
    for (let d = 0; d < 180; d += 1) {
      const r = rowScore(points, (d * Math.PI) / 180, between);
      if (r.score > best.score) best = { score: r.score, ang: (d * Math.PI) / 180, lag: r.lag };
    }
    for (let d = -1; d <= 1; d += 0.25) {
      const ang = best.ang + (d * Math.PI) / 180, r = rowScore(points, ang, between);
      if (r.score > best.score) best = { score: r.score, ang, lag: r.lag };
    }
    return best.score > 0 ? best : null;
  }

  function measureGrid(points, designFt, opts0 = {}) {
    const nearest = nearestDistances(points);
    const radius = 3.2 * nearest;
    const pairs = pairsWithin(points, radius);
    if (points.length < 20 || pairs.length < 50) throw new Error("Too few crowns to measure the planting grid.");
    const width = nearest / 24;
    const nb = Math.ceil((2 * radius) / width) + 1;
    const hist = new Float32Array(nb * nb);
    for (const [i, j] of pairs) {
      for (const s of [1, -1]) {
        const dx = s * (points[j][0] - points[i][0]), dy = s * (points[j][1] - points[i][1]);
        const bx = Math.floor((dx + radius) / width), by = Math.floor((dy + radius) / width);
        if (bx >= 0 && by >= 0 && bx < nb && by < nb) hist[bx * nb + by]++;
      }
    }
    const smooth = gauss(hist, nb, nb, 1.2);
    const win = Math.max(3, Math.round(nearest / 3 / width) | 1);
    const mx = maxFilter(smooth, nb, nb, win >> 1);
    let top = 0;
    for (let i = 0; i < smooth.length; i++) if (smooth[i] > top) top = smooth[i];
    const found = [];
    for (let bx = 0; bx < nb; bx++) for (let by = 0; by < nb; by++) {
      const i = bx * nb + by;
      if (smooth[i] === mx[i] && smooth[i] > 0.15 * top) {
        const v = [-radius + (bx + 0.5) * width, -radius + (by + 0.5) * width];
        if (len(v) > 0.4 * nearest) found.push({ v, s: smooth[i] });
      }
    }
    if (!found.length) throw new Error("No regular planting grid found in the crowns.");
    const strongest = Math.max(...found.map((f) => f.s));
    const strong = found.filter((f) => f.s >= 0.35 * strongest);
    let a = null, b = null;
    if (designFt) {
      const [inRow, between] = designFt.map((f) => 0.3048 * f);
      const near = strong.filter((f) => Math.abs(len(f.v) - inRow) < 0.1 * inRow);
      if (near.length) {
        a = near.reduce((p, q) => (q.s > p.s ? q : p)).v;
        const nearB = strong.filter((f) => Math.abs(len(f.v) - between) < 0.1 * between && cross(a, f.v) > Math.sin((75 * Math.PI) / 180));
        b = nearB.length ? nearB.reduce((p, q) => (q.s > p.s ? q : p)).v : null;
      }
    }
    if (!a || !b) {
      const ordered = strong.map((f) => f.v).sort((p, q) => len(p) - len(q));
      a = ordered[0];
      b = ordered.slice(1).find((v) => cross(a, v) > Math.sin((20 * Math.PI) / 180)) || null;
      if (!b) throw new Error("Only one planting direction found in the crowns.");
    }
    // The rows must run where the treetops really form rows. Old orchards
    // with touching crowns have no clear tree-to-tree distance, and the
    // step picked above can then be a chance direction across the rows;
    // the rows themselves (parallel lines one row spacing apart) are then
    // the better guide, and the grid is turned to them.
    let turned = false;
    if (designFt && opts0.checkRows !== false) {
      const between = designFt[1] * 0.3048;
      const best = bestRowDirection(points, between);
      if (best) {
        const angA = Math.atan2(a[1], a[0]);
        let diff = Math.abs(angA - best.ang) % Math.PI;
        diff = Math.min(diff, Math.PI - diff);
        const own = rowScore(points, angA, between);
        if (opts0.debug) opts0.debug({ bestDeg: (best.ang * 180) / Math.PI, bestScore: best.score, bestLag: best.lag, ownDeg: (angA * 180) / Math.PI, ownScore: own.score, ownLag: own.lag });   // testing only
        // (only to rows one design spacing apart, clearly better than the
        // direction picked above)
        if (diff > (4 * Math.PI) / 180 && own.score < 0.6 * best.score && Math.abs(best.lag / between - 1) <= 0.15 && best.score >= 0.25) {
          const L = len(a);
          a = [L * Math.cos(best.ang), L * Math.sin(best.ang)];
          turned = true;
        }
      }
    }
    // The real distance between rows, measured on the treetops (the trees
    // of a row pile up at one place across the rows). Where the design
    // spacing is not what was planted, the measured one is used.
    {
      const ah = [a[0] / len(a), a[1] / len(a)], across = [-ah[1], ah[0]];
      const rp = rowPeriod(points, across, len(a));
      const P = rp && rp.P;
      const perp = (v) => Math.abs(v[0] * across[0] + v[1] * across[1]);
      if (P && (turned || (Math.abs(perp(b) / P - 1) > 0.12 && rp.at(perp(b)) < 0.7 * rp.at(P)))) {
        const fits = found.filter((f) => Math.abs(perp(f.v) / P - 1) <= 0.12 && cross(a, f.v) > Math.sin((20 * Math.PI) / 180));
        if (fits.length) b = fits.reduce((p, q) => (q.s > p.s ? q : p)).v;
        else {
          // step across one row; how far the next row is shifted along the
          // row comes from the tree pairs one row apart (circular mean)
          const sgn = b[0] * across[0] + b[1] * across[1] < 0 ? -1 : 1;
          const ia = len(a);
          let cs = 0, sn = 0;
          for (const [i, j] of pairsWithin(points, 1.3 * P)) {
            let dx = points[j][0] - points[i][0], dy = points[j][1] - points[i][1];
            let pp = dx * across[0] + dy * across[1];
            if (pp < 0) { dx = -dx; dy = -dy; pp = -pp; }
            if (Math.abs(pp / P - 1) > 0.15) continue;
            const along = dx * ah[0] + dy * ah[1], ang = (2 * Math.PI * along) / ia;
            cs += Math.cos(ang); sn += Math.sin(ang);
          }
          const shift = Math.hypot(cs, sn) > 0 ? (Math.atan2(sn, cs) / (2 * Math.PI)) * ia : 0;
          b = [sgn * (P * across[0] + shift * ah[0]), sgn * (P * across[1] + shift * ah[1])];
        }
      } else if (turned && !P) {
        const between = designFt[1] * 0.3048, sgn = b[0] * across[0] + b[1] * across[1] < 0 ? -1 : 1;
        b = [sgn * between * across[0], sgn * between * across[1]];
      }
    }
    for (let k = 0; k < 20; k++) {
      if (len(b) < len(a)) [a, b] = [b, a];
      const m = Math.round((a[0] * b[0] + a[1] * b[1]) / (a[0] * a[0] + a[1] * a[1]));
      if (m === 0) break;
      b = [b[0] - m * a[0], b[1] - m * a[1]];
    }
    const near = pairsWithin(points, 1.3 * Math.max(len(a), len(b)));
    for (let it = 0; it < 4; it++) {
      const I = inv2(a, b);
      let s00 = 0, s01 = 0, s11 = 0, d00 = 0, d01 = 0, d10 = 0, d11 = 0;
      const step = len(a);
      for (const [i, j] of near) {
        const dx = points[j][0] - points[i][0], dy = points[j][1] - points[i][1];
        const p = Math.round(I[0][0] * dx + I[0][1] * dy), q = Math.round(I[1][0] * dx + I[1][1] * dy);
        const ex = dx - (p * a[0] + q * b[0]), ey = dy - (p * a[1] + q * b[1]);
        if (Math.hypot(ex, ey) >= 0.25 * step || Math.max(Math.abs(p), Math.abs(q)) > 2 || !(Math.abs(p) + Math.abs(q))) continue;
        s00 += p * p; s01 += p * q; s11 += q * q;
        d00 += p * dx; d01 += p * dy; d10 += q * dx; d11 += q * dy;
      }
      const det = s00 * s11 - s01 * s01;
      if (Math.abs(det) < 1e-9) break;
      a = [(s11 * d00 - s01 * d10) / det, (s11 * d01 - s01 * d11) / det];
      b = [(-s01 * d00 + s00 * d10) / det, (-s01 * d01 + s00 * d11) / det];
    }
    // In-row step points down the image (south-ish), row step points right.
    if (a[1] > 0) a = [-a[0], -a[1]];
    if (b[0] < 0) b = [-b[0], -b[1]];
    return { a, b, nearest };
  }

  function gridPositions(points, a, b, tolerance = 0.3) {
    const n = points.length;
    const I = inv2(a, b);
    const inRow = len(a);
    const link = 1.25 * Math.max(len(a), len(b));
    const fit = (dx, dy) => {
      const u = I[0][0] * dx + I[0][1] * dy, v = I[1][0] * dx + I[1][1] * dy;
      const p = Math.round(u), q = Math.round(v);
      return { p, q, err: Math.max(Math.abs(u - p), Math.abs(v - q)) };
    };
    const edges = [];
    for (const [i, j] of pairsWithin(points, link)) {
      const f = fit(points[j][0] - points[i][0], points[j][1] - points[i][1]);
      if (f.err < tolerance && Math.max(Math.abs(f.p), Math.abs(f.q)) <= 2 && Math.abs(f.p) + Math.abs(f.q) > 0) {
        edges.push({ i, j, w: f.err + 0.01 * (Math.abs(f.p) + Math.abs(f.q)), p: f.p, q: f.q });
      }
    }
    edges.sort((x, y) => x.w - y.w);
    const parent = Int32Array.from({ length: n }, (_, i) => i);
    const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    const adj = Array.from({ length: n }, () => []);
    for (const e of edges) {
      const ra = find(e.i), rb = find(e.j);
      if (ra === rb) continue;
      parent[ra] = rb;
      adj[e.i].push([e.j, e.p, e.q]);
      adj[e.j].push([e.i, -e.p, -e.q]);
    }
    const local = Array.from({ length: n }, () => [0, 0]);
    const group = new Int32Array(n).fill(-1);
    let groups = 0;
    const sizes = [];
    for (let s = 0; s < n; s++) {
      if (group[s] >= 0) continue;
      const queue = [s];
      group[s] = groups;
      let size = 0;
      while (queue.length) {
        const u = queue.pop();
        size++;
        for (const [v, p, q] of adj[u]) {
          if (group[v] >= 0) continue;
          group[v] = groups;
          local[v] = [local[u][0] + p, local[u][1] + q];
          queue.push(v);
        }
      }
      sizes.push(size);
      groups++;
    }
    // The largest group is the planting block; other groups join it when
    // nearby placed trees agree on their position.
    const order = sizes.map((s, g) => g).sort((x, y) => sizes[y] - sizes[x]);
    const pos = Array.from({ length: n }, () => null);
    const placed = new Uint8Array(n);
    const members = Array.from({ length: groups }, () => []);
    for (let i = 0; i < n; i++) members[group[i]].push(i);
    for (const i of members[order[0]]) { pos[i] = local[i].slice(); placed[i] = 1; }
    let pending = order.slice(1);
    let changed = true;
    while (changed && pending.length) {
      changed = false;
      const placedPts = [], placedIdx = [];
      for (let i = 0; i < n; i++) if (placed[i]) { placedPts.push(points[i]); placedIdx.push(i); }
      for (const g of pending.slice()) {
        const votes = new Map();
        for (const u of members[g]) {
          for (let k = 0; k < placedIdx.length; k++) {
            const v = placedIdx[k];
            const dx = points[u][0] - points[v][0], dy = points[u][1] - points[v][1];
            const d = Math.hypot(dx, dy);
            if (d > 2.5 * link) continue;
            const f = fit(dx, dy);
            if (f.err >= tolerance) continue;
            const key = (pos[v][0] + f.p - local[u][0]) + "," + (pos[v][1] + f.q - local[u][1]);
            votes.set(key, (votes.get(key) || 0) + 1 / (1 + d / inRow));
          }
        }
        if (!votes.size) continue;
        let bestKey = null, best = 0, total = 0;
        for (const [k, w] of votes) { total += w; if (w > best) { best = w; bestKey = k; } }
        if (best >= 0.6 * total) {
          const [oi, oj] = bestKey.split(",").map(Number);
          for (const u of members[g]) { pos[u] = [local[u][0] + oi, local[u][1] + oj]; placed[u] = 1; }
          pending = pending.filter((x) => x !== g);
          changed = true;
        }
      }
    }
    // Off grid: far from where its placed neighbours put it.
    const fitM = new Float64Array(n).fill(NaN);
    const near = pairsWithin(points, link);
    const nbr = Array.from({ length: n }, () => []);
    for (const [i, j] of near) { nbr[i].push(j); nbr[j].push(i); }
    for (let t = 0; t < n; t++) {
      if (!placed[t]) continue;
      const px = [], py = [];
      for (const v of nbr[t]) {
        if (!placed[v]) continue;
        const di = pos[t][0] - pos[v][0], dj = pos[t][1] - pos[v][1];
        px.push(points[v][0] + di * a[0] + dj * b[0]);
        py.push(points[v][1] + di * a[1] + dj * b[1]);
      }
      if (!px.length) continue;
      const ex = points[t][0] - percentile(px, 50), ey = points[t][1] - percentile(py, 50);
      const u = I[0][0] * ex + I[0][1] * ey, v = I[1][0] * ex + I[1][1] * ey;
      fitM[t] = Math.hypot(ex, ey);
      if (Math.max(Math.abs(u), Math.abs(v)) > 0.4) { placed[t] = 0; pos[t] = null; }
    }
    return { pos, placed, fitM };
  }

  // ------------------------------------------------------------------ elevation (DSM)
  // Ground elevation under the trees: the DSM median (1 m cells) on pixels
  // the RGB classifier calls ground, filled in under the crowns by
  // multi-scale normalized convolution (same as the notebook).
  function terrainModel(surface, ground, w, h, pixel) {
    const step = Math.max(1, Math.round(1.0 / pixel));
    const W = Math.ceil(w / step), H = Math.ceil(h / step);
    const level = new Float32Array(W * H), usable = new Float32Array(W * H);
    const buf = [];
    const levels = [];
    for (let cy = 0; cy < H; cy++) for (let cx = 0; cx < W; cx++) {
      buf.length = 0;
      for (let y = cy * step; y < Math.min(h, (cy + 1) * step); y++) {
        for (let x = cx * step; x < Math.min(w, (cx + 1) * step); x++) {
          const i = y * w + x;
          if (ground[i]) buf.push(surface[i]);
        }
      }
      if (buf.length >= 0.25 * step * step) {
        buf.sort((p, q) => p - q);
        level[cy * W + cx] = buf[buf.length >> 1];
        usable[cy * W + cx] = 1;
        levels.push(level[cy * W + cx]);
      }
    }
    if (levels.length < 10) return null;
    const fallback = percentile(levels, 50);
    const terrain = new Float32Array(W * H).fill(NaN);
    const weighted = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) weighted[i] = level[i] * usable[i];
    for (const sigmaM of [1.5, 3, 6, 12, 24, 48]) {
      const s = sigmaM / (step * pixel);
      const support = gauss(usable, W, H, s), estimate = gauss(weighted, W, H, s);
      for (let i = 0; i < W * H; i++) if (Number.isNaN(terrain[i]) && support[i] > 0.15) terrain[i] = estimate[i] / Math.max(support[i], 1e-6);
    }
    for (let i = 0; i < W * H; i++) if (Number.isNaN(terrain[i])) terrain[i] = fallback;
    // Where hardly any bare ground shows (lush weeds or grass everywhere),
    // the terrain filled in from far away can lie metres above the real
    // ground (on a slope, in a hollow), and every tree there would look
    // too low. The DSM itself shows the ground: its low surface with the
    // trees taken off (morphological opening over 8 m). The terrain never
    // lies above that (plus 0.3 m).
    {
      const lowCell = new Float32Array(W * H).fill(Infinity);
      for (let cy = 0; cy < H; cy++) for (let cx = 0; cx < W; cx++) {
        buf.length = 0;
        for (let y = cy * step; y < Math.min(h, (cy + 1) * step); y++) {
          for (let x = cx * step; x < Math.min(w, (cx + 1) * step); x++) {
            const s = surface[y * w + x];
            if (Number.isFinite(s)) buf.push(s);
          }
        }
        if (buf.length >= 0.25 * step * step) { buf.sort((p, q) => p - q); lowCell[cy * W + cx] = buf[buf.length >> 2]; }
      }
      const R = Math.max(1, Math.round(8.0 / (step * pixel)));
      const pass = (src, horizontal, take, none) => {
        const out = new Float32Array(W * H);
        const L = horizontal ? W : H, M = horizontal ? H : W;
        for (let m = 0; m < M; m++) for (let l = 0; l < L; l++) {
          let best = none;
          for (let k = Math.max(0, l - R); k <= Math.min(L - 1, l + R); k++) {
            const v = src[horizontal ? m * W + k : k * W + m];
            if (Number.isFinite(v)) best = take(best, v);
          }
          out[horizontal ? m * W + l : l * W + m] = best;
        }
        return out;
      };
      const eroded = pass(pass(lowCell, true, Math.min, Infinity), false, Math.min, Infinity);
      const opened = pass(pass(eroded, true, Math.max, -Infinity), false, Math.max, -Infinity);
      for (let i = 0; i < W * H; i++) if (Number.isFinite(opened[i]) && Number.isFinite(lowCell[i]) && opened[i] + 0.3 < terrain[i]) terrain[i] = opened[i] + 0.3;
    }
    const smooth = gauss(terrain, W, H, 1.0 / (step * pixel));
    const out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const fy = Math.min(H - 1, Math.max(0, (y + 0.5) / step - 0.5)), y0 = Math.floor(fy), y1 = Math.min(H - 1, y0 + 1), ty = fy - y0;
      for (let x = 0; x < w; x++) {
        const fx = Math.min(W - 1, Math.max(0, (x + 0.5) / step - 0.5)), x0 = Math.floor(fx), x1 = Math.min(W - 1, x0 + 1), tx = fx - x0;
        out[y * w + x] = (1 - ty) * ((1 - tx) * smooth[y0 * W + x0] + tx * smooth[y0 * W + x1]) + ty * ((1 - tx) * smooth[y1 * W + x0] + tx * smooth[y1 * W + x1]);
      }
    }
    return out;
  }

  // The DSM decides WHAT is a tree (canopy at least 1 m tall), the RGB image
  // decides WHERE the crown edge is (a DSM smears crown edges).
  function elevationRefine(surface, mask, P, valid, w, h, pixel) {
    const n = w * h, area = pixel * pixel;
    const known = new Uint8Array(n);
    let knownCount = 0, validCount = 0;
    for (let i = 0; i < n; i++) {
      if (valid[i]) validCount++;
      if (valid[i] && Number.isFinite(surface[i])) { known[i] = 1; knownCount++; }
    }
    const coverage = knownCount / Math.max(validCount, 1);
    if (coverage < 0.5) return { mask, chm: null, info: { used: false, reason: `the elevation file covers only ${Math.round(100 * coverage)}% of the field` } };

    const low = percentile(sample(surface, 7, (i) => known[i]), 5);
    const isHeight = low >= -1 && low <= 1;
    let chm;
    if (isHeight) {
      chm = Float32Array.from(surface);
    } else {
      const clearance = edt(not(mask), w, h);
      const ground = new Uint8Array(n);
      for (let i = 0; i < n; i++) ground[i] = known[i] && P[i] < 0.2 && clearance[i] * pixel >= 0.6 ? 1 : 0;
      const terrain = terrainModel(surface, ground, w, h, pixel);
      if (!terrain) return { mask, chm: null, info: { used: false, reason: "too few ground pixels to model the terrain" } };
      chm = new Float32Array(n);
      for (let i = 0; i < n; i++) chm[i] = surface[i] - terrain[i];
    }
    for (let i = 0; i < n; i++) if (!known[i]) chm[i] = NaN;

    const under = sample(chm, 3, (i) => mask[i] && Number.isFinite(chm[i]));
    if (under.length && percentile(under, 90) < 1.0) {
      return { mask, chm: null, info: { used: false, reason: "no tree heights under the canopy (terrain-only file, or not the same field)" } };
    }

    const before = mask.reduce((s, v) => s + v, 0);
    const confirmed = new Uint8Array(n);
    for (let i = 0; i < n; i++) confirmed[i] = mask[i] && chm[i] >= 1.0 ? 1 : 0;
    const near = dilate(confirmed, w, h, 0.5 / pixel);
    let out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = mask[i] && (near[i] || !known[i]) ? 1 : 0;
    out = fillSmallHoles(out, w, h, Math.round(4.0 / area));
    out = opening(out, w, h, 0.2 / pixel);

    // Every canopy patch must be a tree: its 90th-percentile height >= 1 m.
    const { labels, count } = label(out, w, h);
    const values = Array.from({ length: count + 1 }, () => []);
    for (let i = 0; i < n; i++) if (labels[i] && Number.isFinite(chm[i])) values[labels[i]].push(chm[i]);
    const keep = new Uint8Array(count + 1);
    let lowPatches = 0;
    for (let l = 1; l <= count; l++) {
      keep[l] = !values[l].length || percentile(values[l], 90) >= 1.0 ? 1 : 0;
      if (!keep[l]) lowPatches++;
    }
    for (let i = 0; i < n; i++) out[i] = labels[i] && keep[labels[i]] ? 1 : 0;
    out = dropSmall(out, w, h, Math.round(2.0 / area));
    for (let i = 0; i < n; i++) if (!valid[i]) out[i] = 0;
    const after = out.reduce((s, v) => s + v, 0);
    return {
      mask: out, chm,
      info: {
        used: true, field_coverage: coverage, is_height_raster: isHeight,
        removed_by_height_m2: (before - after) * area, low_patches_removed: lowPatches,
      },
    };
  }

  // Clear cast shadow trimmed from the crown edges (gentle, as in the
  // notebook): dark AND less green than leaves AND at ground level in the
  // DSM; without a DSM, darker than the field's typical ground shadow.
  function trimShadow(rgba, mask, valid, chm, w, h, pixel, darkY) {
    const n = w * h, area = pixel * pixel;
    const Y = new Float32Array(n), E = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const R = rgba[4 * i], G = rgba[4 * i + 1], B = rgba[4 * i + 2];
      Y[i] = 0.299 * R + 0.587 * G + 0.114 * B;
      E[i] = (2 * G - R - B) / (R + G + B + 1);
    }
    const ys = gauss(Y, w, h, 0.25 / pixel), es = gauss(E, w, h, 0.25 / pixel);
    const groundExg = sample(es, 3, (i) => valid[i] && !mask[i] && ys[i] < darkY);
    const canopyExg = sample(es, 3, (i) => mask[i] && ys[i] < darkY);
    const enough = groundExg.length > 300 && canopyExg.length > 300;
    const cut = enough ? 0.5 * (percentile(groundExg, 50) + percentile(canopyExg, 50)) : 0.12;
    const veryDark = chm ? null : (enough ? percentile(sample(ys, 3, (i) => valid[i] && !mask[i] && ys[i] < darkY), 50) : 0.6 * darkY);
    const shadow = new Uint8Array(n);
    let candidates = 0;
    for (let i = 0; i < n; i++) {
      let s = mask[i] && ys[i] < darkY && es[i] < cut;
      if (s) s = chm ? chm[i] < 0.5 : ys[i] < veryDark;  // NaN height: kept
      if (s) { shadow[i] = 1; candidates++; }
    }
    if (!chm && candidates) {
      // Image only: without height a dark leaf can look like shadow, so only
      // the crown edge (1 m) is trimmed, and at most the darkest 3 % of the
      // canopy.
      const inside = edt(mask, w, h);
      let canopyPx = 0;
      const darkness = [];
      for (let i = 0; i < n; i++) {
        if (mask[i]) canopyPx++;
        if (shadow[i] && inside[i] * pixel > 1.0) shadow[i] = 0;
        if (shadow[i]) darkness.push(ys[i]);
      }
      const cap = Math.floor(0.03 * canopyPx);
      if (darkness.length > cap) {
        const limit = cap > 0 ? percentile(darkness, (100 * cap) / darkness.length) : -Infinity;
        for (let i = 0; i < n; i++) if (shadow[i] && !(ys[i] <= limit)) shadow[i] = 0;
      }
    }
    const keepPx = new Uint8Array(n);
    for (let i = 0; i < n; i++) keepPx[i] = mask[i] && !shadow[i] ? 1 : 0;
    let out = opening(keepPx, w, h, 0.2 / pixel);
    const filled = fillSmallHoles(out, w, h, Math.round(1.0 / area));
    for (let i = 0; i < n; i++) if (filled[i] && mask[i]) out[i] = 1;
    out = dropSmall(out, w, h, Math.round(2.0 / area));
    let removed = 0;
    for (let i = 0; i < n; i++) { if (!valid[i]) out[i] = 0; if (mask[i] && !out[i]) removed++; }
    return { mask: out, removed_m2: removed * area };
  }

  // ------------------------------------------------------------------ row alternation
  // Crowns next to each other in a row always get opposite colours, also
  // across a missing tree: the crowns on one side of the gap move one
  // position toward it (next diagonal, other colour). Per row there are two
  // ways to do it; the choice for all rows together with the fewest extra
  // steps in the diagonals (then the fewest moved crowns) is found exactly
  // by dynamic programming over the rows (as in the notebook).
  function alternateAlongRows(pos, alive, stepDir, coordinate = true) {
    const shifted = pos.map((p) => (p ? p.slice() : null));
    const moved = new Uint8Array(pos.length);
    const rows = new Map();
    pos.forEach((p, t) => {
      if (!p || !alive[t]) return;
      if (!rows.has(p[1])) rows.set(p[1], []);
      rows.get(p[1]).push(t);
    });
    const rowIds = [...rows.keys()].sort((p, q) => p - q);
    const optionsFor = (members, row) => {
      members = members.slice().sort((p, q) => pos[p][0] - pos[q][0]);
      const runs = [[members[0]]];
      for (let k = 1; k < members.length; k++) {
        if (pos[members[k]][0] - pos[members[k - 1]][0] > 1) runs.push([]);
        runs[runs.length - 1].push(members[k]);
      }
      const gaps = runs.slice(0, -1).map((run, r) => pos[runs[r + 1][0]][0] - pos[run[run.length - 1]][0]);
      const flip = [0];
      for (const g of gaps) flip.push(flip[flip.length - 1] ^ (g % 2 === 0 ? 1 : 0));
      const patterns = runs.length < 2 ? [flip] : [flip, flip.map((f) => 1 - f)];
      return patterns.map((pattern) => {
        const shift = new Map();
        runs.forEach((run, r) => {
          let s = 0;
          if (pattern[r]) {
            const intoLeft = r > 0 && !pattern[r - 1] && gaps[r - 1] % 2 === 0;
            const intoRight = r < runs.length - 1 && !pattern[r + 1] && gaps[r] % 2 === 0;
            s = intoLeft ? -1 : intoRight ? 1 : r > 0 ? -1 : 1;
          }
          for (const t of run) shift.set(t, s);
        });
        const diag = new Map();
        for (const [t, s] of shift) diag.set((pos[t][0] + s) * stepDir[1] - row * stepDir[0], s);
        let movedCount = 0;
        for (const s of shift.values()) if (s) movedCount++;
        return { shift, diag, moved: movedCount };
      });
    };
    const between = (earlier, later) => {
      let c = 0;
      for (const [k, s] of later.diag) if (earlier.diag.has(k) && earlier.diag.get(k) !== s) c++;
      return c;
    };
    const choices = rowIds.map((row) => optionsFor(rows.get(row), row));
    let extraSteps = 0;
    if (rowIds.length) {
      const cost = [choices[0].map((o) => 0.01 * o.moved)], back = [choices[0].map(() => 0)];
      for (let k = 1; k < rowIds.length; k++) {
        const adjacent = rowIds[k] - rowIds[k - 1] === 1;
        const here = [], prev = [];
        for (const option of choices[k]) {
          let best = 0, bestCost = Infinity;
          choices[k - 1].forEach((p, pi) => {
            const c = cost[k - 1][pi] + (adjacent ? between(p, option) : 0);
            if (c < bestCost) { bestCost = c; best = pi; }
          });
          here.push(bestCost + 0.01 * option.moved);
          prev.push(best);
        }
        cost.push(here); back.push(prev);
      }
      const last = cost[cost.length - 1];
      let pick = last.indexOf(Math.min(...last));
      const picked = new Array(rowIds.length);
      for (let k = rowIds.length - 1; k >= 0; k--) { picked[k] = pick; pick = back[k][pick]; }
      // A moved run may move either way (one position left or right: both
      // give it the other colour, so the colours stay exactly the same). It
      // moves the way the crowns of the rows beside it moved, so the lines
      // do not jog from row to row (iterated, each row against both of its
      // neighbours, until nothing changes).
      const current = rowIds.map((row, k) => choices[k][picked[k]]);
      if (coordinate) {
        const keyOf = (t, s, row) => (pos[t][0] + s) * stepDir[1] - row * stepDir[0];
        const runsOf = rowIds.map((row) => {
          const members = rows.get(row).slice().sort((p, q) => pos[p][0] - pos[q][0]);
          const runs = [[members[0]]];
          for (let k = 1; k < members.length; k++) {
            if (pos[members[k]][0] - pos[members[k - 1]][0] > 1) runs.push([]);
            runs[runs.length - 1].push(members[k]);
          }
          return runs;
        });
        const jogCost = (k, run, s) => {
          let c = 0;
          for (const kk of [k - 1, k + 1]) {
            if (kk < 0 || kk >= rowIds.length || Math.abs(rowIds[kk] - rowIds[k]) !== 1) continue;
            const nd = current[kk].diag;
            for (const t of run) { const key = keyOf(t, s, rowIds[k]); if (nd.has(key)) c += Math.abs(nd.get(key) - s); }
          }
          return c;
        };
        for (let sweep = 0; sweep < 12; sweep++) {
          let changed = 0;
          for (let k = 0; k < rowIds.length; k++) {
            const row = rowIds[k];
            let bestOpt = null, bestCost = 0.01 * current[k].moved;
            for (const run of runsOf[k]) bestCost += jogCost(k, run, current[k].shift.get(run[0]));
            bestCost -= 1e-6;                                   // keep the current choice on a tie
            for (const base of choices[k]) {
              const shift = new Map();
              let c = 0;
              for (const run of runsOf[k]) {
                const s0 = base.shift.get(run[0]);
                if (!s0) { for (const t of run) shift.set(t, 0); c += jogCost(k, run, 0); continue; }
                const cA = jogCost(k, run, s0), cB = jogCost(k, run, -s0);
                const s = cB < cA ? -s0 : s0;
                for (const t of run) shift.set(t, s);
                c += Math.min(cA, cB);
              }
              c += 0.01 * base.moved;
              if (c < bestCost - 1e-9) {
                const diag = new Map();
                for (const [t, s] of shift) diag.set(keyOf(t, s, row), s);
                bestCost = c; bestOpt = { shift, diag, moved: base.moved };
              }
            }
            if (bestOpt) { current[k] = bestOpt; changed++; }
          }
          if (!changed) break;
        }
      }
      rowIds.forEach((row, k) => {
        for (const [t, s] of current[k].shift) if (s) { shifted[t][0] += s; moved[t] = 1; }
        if (k && rowIds[k] - rowIds[k - 1] === 1) extraSteps += between(current[k - 1], current[k]);
      });
    }
    return { shifted, moved, extraSteps };
  }

  // ------------------------------------------------------------------ input
  // Shared by the dashboard and the Kaggle notebook, so both analyse exactly
  // the same pixels: the full-resolution image is averaged onto the analysis
  // grid (every source pixel goes to exactly one grid cell; integer sums, so
  // the order of reading does not matter), and the elevation (DSM) is sampled
  // at the centre of every grid cell.
  const DETAIL_PIXELS = { fast: 3e6, standard: 5e6, fine: 9e6 };

  // Ground size of one image pixel in metres, from its coordinate system.
  function groundPixelSize({ kind, rx, ry, miny, maxy, unitFactor = 1 }) {
    rx = Math.abs(rx); ry = Math.abs(ry);
    if (kind === "geographic") {
      const lat0 = ((miny + maxy) / 2) * Math.PI / 180;
      return { px: rx * 111320 * Math.cos(lat0), py: ry * 110574 };
    }
    if (kind === "webmercator") {
      const k = Math.cos(2 * Math.atan(Math.exp((miny + maxy) / 2 / 6378137)) - Math.PI / 2);
      return { px: rx * k, py: ry * k };
    }
    return { px: rx * unitFactor, py: ry * unitFactor };
  }

  // Analysis grid: at most `budget` pixels on the field (Detail setting).
  function gridSize(width, height, px, py, fieldFraction, budget) {
    const total = Math.min(budget / Math.max(0.05, fieldFraction), 16e6);
    const native = Math.sqrt(px * py);
    const pixel = Math.max(native, Math.sqrt((width * px * height * py) / total), 0.03);
    return {
      W: Math.max(64, Math.round((width * px) / pixel)),
      H: Math.max(64, Math.round((height * py) / pixel)),
      pixel,
    };
  }

  // Field fraction of the image: share of pixels with alpha > 127.
  function countField(alpha) {
    let inside = 0;
    for (let i = 0; i < alpha.length; i++) if (alpha[i] > 127) inside++;
    return inside;
  }

  // Averages full-resolution rows onto the W x H grid. bands = [R, G, B] (+ A).
  function createGrid(width, height, W, H, { hasAlpha = false, nodata = null, bits = 8 } = {}) {
    const n = W * H;
    const sum = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
    const good = new Uint32Array(n), all = new Uint32Array(n);
    const colOf = new Int32Array(width);
    for (let c = 0; c < width; c++) colOf[c] = Math.min(W - 1, Math.floor(((c + 0.5) * W) / width));
    const nd = nodata === null || nodata === undefined || Number.isNaN(Number(nodata)) ? null : Number(nodata);
    return {
      addRows(r0, rows, bands) {
        const [R, G, B, A] = bands;
        for (let k = 0; k < rows; k++) {
          const r = r0 + k;
          const o = Math.min(H - 1, Math.floor(((r + 0.5) * H) / height)) * W;
          for (let c = 0; c < width; c++) {
            const s = k * width + c, i = o + colOf[c];
            all[i]++;
            let ok = hasAlpha ? A[s] > 127 : true;
            if (ok && !hasAlpha && nd !== null && R[s] === nd && G[s] === nd && B[s] === nd) ok = false;
            if (!ok) continue;
            good[i]++;
            sum[0][i] += R[s]; sum[1][i] += G[s]; sum[2][i] += B[s];
          }
        }
      },
      finish() {
        const rgba = new Uint8ClampedArray(n * 4);
        const mean = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
        for (let i = 0; i < n; i++) {
          if (good[i]) for (let c = 0; c < 3; c++) mean[c][i] = sum[c][i] / good[i];
          rgba[4 * i + 3] = good[i] * 2 > all[i] ? 255 : 0;
        }
        if (bits > 8) {  // 16-bit: stretch the 1st-99th percentiles to 0-255
          for (let c = 0; c < 3; c++) {
            const vals = [];
            for (let i = 0; i < n; i += 7) if (rgba[4 * i + 3]) vals.push(mean[c][i]);
            vals.sort((p, q) => p - q);
            const lo = vals[Math.floor(0.01 * (vals.length - 1))] || 0, hi = vals[Math.floor(0.99 * (vals.length - 1))] || 1;
            for (let i = 0; i < n; i++) mean[c][i] = ((mean[c][i] - lo) / Math.max(hi - lo, 1e-9)) * 255;
          }
        }
        for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) rgba[4 * i + c] = Math.round(mean[c][i]);
        return rgba;
      },
    };
  }

  // Elevation at the centre of every grid cell (same coordinate system as
  // the image). img/dsm: { left, top, resX, resY } (resY < 0 for north-up);
  // dsm also { data, width, height, nodata }.
  function sampleSurface(dsm, img, width, height, W, H) {
    const spanX = (width * img.resX) / W, spanY = (height * img.resY) / H;
    const nd = dsm.nodata === null || dsm.nodata === undefined ? null : Number(dsm.nodata);
    const surface = new Float32Array(W * H);
    let inside = 0;
    for (let y = 0; y < H; y++) {
      const r = Math.floor((img.top + (y + 0.5) * spanY - dsm.top) / dsm.resY);
      for (let x = 0; x < W; x++) {
        const c = Math.floor((img.left + (x + 0.5) * spanX - dsm.left) / dsm.resX);
        let v = NaN;
        if (r >= 0 && c >= 0 && r < dsm.height && c < dsm.width) {
          v = dsm.data[r * dsm.width + c];
          if (!Number.isFinite(v) || (nd !== null && v === nd) || v < -1e5 || v > 1e5) v = NaN;
        }
        surface[y * W + x] = v;
        if (Number.isFinite(v)) inside++;
      }
    }
    return { surface, inside };
  }

  // ------------------------------------------------------------------ main
  async function analyze(input, opts = {}, progress = () => {}) {
    const step = async (text) => { progress(text); await new Promise((r) => setTimeout(r, 0)); };
    const { rgba, width: w, height: h, pixel } = input;
    const n = w * h, area = pixel * pixel;
    const design = opts.designFt === undefined ? [20, 40] : opts.designFt;
    const hint = opts.directionHintDeg || 114;

    // Field = mask pixels (alpha) or, without a mask, the image minus empty
    // black / white borders touching the edge.
    await step("Finding the field …");
    let valid = new Uint8Array(n);
    let hasMask = false;
    for (let i = 0; i < n; i++) { valid[i] = rgba[4 * i + 3] > 0 ? 1 : 0; if (!valid[i]) hasMask = true; }
    if (!hasMask || input.noMask) {
      const flat = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        const R = rgba[4 * i], G = rgba[4 * i + 1], B = rgba[4 * i + 2];
        flat[i] = (R <= 3 && G <= 3 && B <= 3) || (R >= 252 && G >= 252 && B >= 252) ? 1 : 0;
      }
      const { labels, count } = label(flat, w, h);
      const edge = new Uint8Array(count + 1);
      for (let x = 0; x < w; x++) { edge[labels[x]] = 1; edge[labels[(h - 1) * w + x]] = 1; }
      for (let y = 0; y < h; y++) { edge[labels[y * w]] = 1; edge[labels[y * w + w - 1]] = 1; }
      for (let i = 0; i < n; i++) valid[i] = valid[i] && !(labels[i] && edge[labels[i]]) ? 1 : 0;
    }

    await step("Colour and texture of every pixel …");
    const F = features(rgba, valid, w, h, pixel);

    await step("Learning what pecan canopy looks like in this image …");
    const { seeds, ownExg, thresholds } = harvest(F, valid, w, h, pixel);
    const model = trainQda(F, seeds);

    await step("Finding the pecan canopy (no grass, soil or shadow) …");
    const skip = new Uint8Array(n);
    for (let i = 0; i < n; i++) skip[i] = ownExg[i] < 0.03 && F.Y[i] > thresholds.brightY && F.t050[i] < thresholds.tValley ? 1 : 0;
    let P = predictCanopy(model, valid, skip, n);
    P = gauss(P, w, h, 0.2 / pixel);
    let mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) mask[i] = valid[i] && P[i] >= 0.5 ? 1 : 0;
    // Young trees: their canopy pieces are smaller than 2 m² and lower than
    // 1 m in the DSM, so the steps below drop them. They are kept aside and
    // count only where they stand on an empty planting position.
    const small = opts.youngTrees === false ? null : closing(mask, w, h, 0.25 / pixel);
    mask = closing(mask, w, h, 0.25 / pixel);
    mask = opening(mask, w, h, 0.2 / pixel);
    mask = fillSmallHoles(mask, w, h, Math.round(4.0 / area));
    mask = dropSmall(mask, w, h, Math.round(2.0 / area));
    for (let i = 0; i < n; i++) if (!valid[i]) mask[i] = 0;
    for (const k of FEATURES) F[k] = null;
    const rgbCanopyM2 = mask.reduce((s, v) => s + v, 0) * area;
    if (opts.debug) opts.debug("rgbmask", mask);   // testing only

    // Elevation (DSM): only canopy that stands at least 1 m tall is a tree.
    let chm = null, elevation = { used: false, reason: "no elevation file" };
    if (input.surface) {
      await step("Elevation (DSM): tree heights, removing grass and weeds …");
      const refined = elevationRefine(input.surface, mask, P, valid, w, h, pixel);
      mask = refined.mask; chm = refined.chm; elevation = refined.info;
    }
    P = null;

    let shadowRemoved = 0;
    const trimmedAway = new Uint8Array(n);
    if (opts.removeShadow !== false) {
      await step("Trimming shadow from the crown edges …");
      const trimmed = trimShadow(rgba, mask, valid, chm, w, h, pixel, thresholds.darkY);
      for (let i = 0; i < n; i++) trimmedAway[i] = mask[i] && !trimmed.mask[i] ? 1 : 0;
      mask = trimmed.mask; shadowRemoved = trimmed.removed_m2;
    }

    await step("Splitting the canopy into crowns …");
    let score, md, peakMin;
    if (chm) {
      // Treetops = local maxima of the smoothed canopy height model.
      const height = new Float32Array(n);
      for (let i = 0; i < n; i++) height[i] = mask[i] && Number.isFinite(chm[i]) ? chm[i] : 0;
      score = gauss(height, w, h, 0.5 / pixel);
      md = Math.max(1, Math.round(2.5 / pixel));
      peakMin = 1.0;
    } else {
      const dist = edt(mask, w, h);
      for (let i = 0; i < n; i++) dist[i] *= pixel;
      score = gauss(dist, w, h, 0.35 / pixel);
      md = Math.max(1, Math.round(3.0 / pixel));
      peakMin = 0.35;
    }
    const mx = maxFilter(score, w, h, md);
    let peaks = [];
    for (let i = 0; i < n; i++) if (mask[i] && score[i] >= mx[i] && score[i] > peakMin) peaks.push(i);
    peaks.sort((p, q) => score[q] - score[p]);
    const taken = new Map(), cell = md;
    const kept = [];
    for (const i of peaks) {
      const x = i % w, y = (i / w) | 0, cx = Math.floor(x / cell), cy = Math.floor(y / cell);
      let ok = true;
      for (let dx = -1; dx <= 1 && ok; dx++) for (let dy = -1; dy <= 1 && ok; dy++) {
        for (const j of taken.get((cx + dx) + "," + (cy + dy)) || []) {
          const jx = j % w, jy = (j / w) | 0;
          if ((jx - x) ** 2 + (jy - y) ** 2 < md * md) { ok = false; break; }
        }
      }
      if (!ok) continue;
      kept.push(i);
      const key = cx + "," + cy;
      if (!taken.has(key)) taken.set(key, []);
      taken.get(key).push(i);
    }
    const markers = new Int32Array(n);
    kept.forEach((i, k) => { markers[i] = k + 1; });
    const comp = label(mask, w, h);
    const seeded = new Uint8Array(comp.count + 1), bestAt = new Int32Array(comp.count + 1).fill(-1);
    for (let i = 0; i < n; i++) {
      const c = comp.labels[i];
      if (!c) continue;
      if (markers[i]) seeded[c] = 1;
      if (bestAt[c] < 0 || score[i] > score[bestAt[c]]) bestAt[c] = i;
    }
    const tops = kept.slice();
    for (let c = 1; c <= comp.count; c++) if (!seeded[c] && bestAt[c] >= 0) { tops.push(bestAt[c]); markers[bestAt[c]] = tops.length; }
    let labels = watershed(score, markers, mask, w, h);

    // Crown table.
    const pts = tops.map((i) => [((i % w) + 0.5) * pixel, (h - ((i / w) | 0) - 0.5) * pixel]);
    const scores = tops.map((i) => score[i]);
    const count = tops.length;
    const sizes = new Int32Array(count + 1);
    for (let i = 0; i < n; i++) sizes[labels[i]]++;

    await step("Measuring the planting grid …");
    const eligible = Array.from({ length: count }, (_, t) => sizes[t + 1] * area >= 2.0);
    const gridPts = pts.filter((_, t) => eligible[t]);
    const gridIdx = pts.map((_, t) => t).filter((t) => eligible[t]);
    const { a, b } = measureGrid(gridPts, design, { checkRows: opts.checkRows, debug: opts.debugGrid });
    const gp = gridPositions(gridPts, a, b);
    const pos = Array.from({ length: count }, () => null);
    gridIdx.forEach((t, k) => { if (gp.placed[k]) pos[t] = gp.pos[k]; });
    // A tree whose top is a little off its planting position (a big or
    // leaning crown) takes the nearest empty position, as its placed
    // neighbours see it, instead of being merged into a neighbour.
    {
      const I0 = inv2(a, b), taken0 = new Set();
      const placedT = [];
      for (let t = 0; t < count; t++) if (pos[t]) { taken0.add(pos[t][0] + "," + pos[t][1]); placedT.push(t); }
      const medSize = percentile(placedT.map((t) => sizes[t + 1]), 50) || 1;
      const cand = [];
      for (let t = 0; t < count; t++) {
        if (pos[t] || !eligible[t] || sizes[t + 1] < 0.25 * medSize) continue;
        const near = placedT.map((s) => [Math.hypot(pts[s][0] - pts[t][0], pts[s][1] - pts[t][1]), s]).sort((p, q) => p[0] - q[0]).slice(0, 6);
        if (near.length < 2 || near[0][0] > 3 * len(b)) continue;
        const us = [], vs = [];
        for (const [, s] of near) {
          const ex = pts[t][0] - pts[s][0], ey = pts[t][1] - pts[s][1];
          us.push(pos[s][0] + I0[0][0] * ex + I0[0][1] * ey);
          vs.push(pos[s][1] + I0[1][0] * ex + I0[1][1] * ey);
        }
        const u = percentile(us, 50), v = percentile(vs, 50), p = Math.round(u), q = Math.round(v);
        if (Math.abs(v - q) > 0.3 || Math.abs(u - p) > 0.5) continue;
        cand.push({ t, p, q, err: Math.abs(u - p) });
      }
      cand.sort((x, y) => x.err - y.err);
      for (const c of cand) {
        const key = c.p + "," + c.q;
        if (taken0.has(key)) continue;
        taken0.add(key);
        pos[c.t] = [c.p, c.q];
      }
    }

    // Label helpers.
    const relabel = (from, to) => {
      const map = new Int32Array(count + 1);
      for (let l = 0; l <= count; l++) map[l] = l;
      for (const [f, t] of from.map((f, k) => [f, to[k]])) map[f] = t;
      for (let i = 0; i < n; i++) if (labels[i]) labels[i] = map[labels[i]];
    };

    // One crown per planting position: crowns on the same site merge.
    await step("One crown per planting position …");
    const siteOf = (p) => p[0] + "," + p[1];
    const sites = new Map();
    for (let t = 0; t < count; t++) if (pos[t]) {
      const k = siteOf(pos[t]);
      if (!sites.has(k)) sites.set(k, []);
      sites.get(k).push(t);
    }
    const alive = new Uint8Array(count).fill(1);
    const from = [], to = [];
    const pieces = new Int32Array(count).fill(1);
    for (const list of sites.values()) {
      if (list.length < 2) continue;
      const keep = list.reduce((p, q) => (scores[q] > scores[p] ? q : p));
      for (const t of list) if (t !== keep) { from.push(t + 1); to.push(keep + 1); alive[t] = 0; pieces[keep]++; }
    }
    // Off-grid pieces touching a tree of the grid within one tree spacing join it.
    const inRow = len(a);
    const touching = new Map();
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, l = labels[i];
      if (!l) continue;
      for (const j of [x < w - 1 ? i + 1 : -1, y < h - 1 ? i + w : -1]) {
        if (j < 0) continue;
        const m = labels[j];
        if (!m || m === l) continue;
        for (const [p, q] of [[l, m], [m, l]]) {
          if (!touching.has(p)) touching.set(p, new Set());
          touching.get(p).add(q);
        }
      }
    }
    const resolve = new Int32Array(count + 1);
    for (let l = 0; l <= count; l++) resolve[l] = l;
    from.forEach((f, k) => { resolve[f] = to[k]; });
    for (let t = 0; t < count; t++) {
      if (pos[t] || !alive[t] || !eligible[t]) continue;
      const near = [...(touching.get(t + 1) || [])].map((l) => resolve[l] - 1).filter((s) => s >= 0 && pos[s] && alive[s]
        && Math.hypot(pts[s][0] - pts[t][0], pts[s][1] - pts[t][1]) < inRow);
      if (!near.length) continue;
      const keep = near.reduce((p, q) => (Math.hypot(pts[q][0] - pts[t][0], pts[q][1] - pts[t][1]) < Math.hypot(pts[p][0] - pts[t][0], pts[p][1] - pts[t][1]) ? q : p));
      from.push(t + 1); to.push(keep + 1); alive[t] = 0; pieces[keep]++;
    }
    relabel(from, to);

    // Empty planting positions covered by canopy: the canopy closest to the
    // position becomes its crown (so touching crowns alternate EVEN / ODD).
    const knots = pts.map((p) => p.slice());
    const site = new Map();
    for (let t = 0; t < count; t++) if (pos[t] && alive[t]) site.set(siteOf(pos[t]), t);
    const extra = [];
    const radius = Math.ceil(inRow / pixel);
    const minArea = 0.2 * percentile(Array.from({ length: count }, (_, t) => sizes[t + 1]).filter((s, t) => alive[t] && eligible[t]), 50);
    for (const t of [...site.values()]) {
      const [i, j] = pos[t];
      for (let gap = 1; gap <= 3; gap++) {
        if (!site.has((i + gap + 1) + "," + j)) continue;
        let empty = true;
        for (let q = 1; q <= gap; q++) if (site.has((i + q) + "," + j)) empty = false;
        if (!empty) continue;
        const right = site.get((i + gap + 1) + "," + j);
        for (let q = 1; q <= gap; q++) {
          const f = q / (gap + 1);
          const spot = [pts[t][0] + f * (pts[right][0] - pts[t][0]), pts[t][1] + f * (pts[right][1] - pts[t][1])];
          const sc = Math.floor(spot[0] / pixel), sr = h - 1 - Math.floor(spot[1] / pixel);
          if (sc < 0 || sr < 0 || sc >= w || sr >= h) continue;
          const at = labels[sr * w + sc];
          const left = q === 1 ? t : null;
          const owners = [t + 1, right + 1].filter((l) => l === at);
          if (!owners.length) continue;  // bare: missing tree
          const r0 = Math.max(0, sr - radius), r1 = Math.min(h - 1, sr + radius);
          const c0 = Math.max(0, sc - radius), c1 = Math.min(w - 1, sc + radius);
          const take = [];
          for (let y = r0; y <= r1; y++) for (let x = c0; x <= c1; x++) {
            const l = labels[y * w + x];
            if (l !== t + 1 && l !== right + 1) continue;
            const own = l === t + 1 ? pts[t] : pts[right];
            const px = (x + 0.5) * pixel, py = (h - y - 0.5) * pixel;
            if ((px - spot[0]) ** 2 + (py - spot[1]) ** 2 < (px - own[0]) ** 2 + (py - own[1]) ** 2) take.push(y * w + x);
          }
          if (take.length * area < 0.0001 || take.length < minArea) continue;
          // keep the piece connected to the spot
          const set = new Set(take), seen = new Set([sr * w + sc]), stack = [sr * w + sc];
          if (!set.has(sr * w + sc)) continue;
          while (stack.length) {
            const u = stack.pop(), ux = u % w;
            for (const v of [ux > 0 ? u - 1 : -1, ux < w - 1 ? u + 1 : -1, u - w, u + w]) if (v >= 0 && set.has(v) && !seen.has(v)) { seen.add(v); stack.push(v); }
          }
          if (seen.size < minArea) continue;
          const id = count + extra.length + 1;
          for (const u of seen) labels[u] = id;
          // A neighbour must stay in one piece: canopy of it cut off on the
          // far side of the new crown joins the new crown (never red, blue,
          // red for one tree along the row).
          const R0 = Math.max(0, sr - 2 * radius), R1 = Math.min(h - 1, sr + 2 * radius);
          const C0 = Math.max(0, sc - 2 * radius), C1 = Math.min(w - 1, sc + 2 * radius);
          for (const donor of [t, right]) {
            const L = donor + 1;
            const pieceOf = new Map();
            let parts = 0;
            const sizesOf = [];
            for (let y = R0; y <= R1; y++) for (let x = C0; x <= C1; x++) {
              const s0 = y * w + x;
              if (labels[s0] !== L || pieceOf.has(s0)) continue;
              parts++;
              let size = 0;
              const stack2 = [s0];
              pieceOf.set(s0, parts);
              while (stack2.length) {
                const u = stack2.pop(), ux = u % w, uy = (u / w) | 0;
                size++;
                for (const v of [ux > C0 ? u - 1 : -1, ux < C1 ? u + 1 : -1, uy > R0 ? u - w : -1, uy < R1 ? u + w : -1]) {
                  if (v >= 0 && labels[v] === L && !pieceOf.has(v)) { pieceOf.set(v, parts); stack2.push(v); }
                }
              }
              sizesOf[parts] = size;
            }
            if (parts < 2) continue;
            const rc = Math.floor(pts[donor][0] / pixel), rr = h - 1 - Math.floor(pts[donor][1] / pixel);
            let main = pieceOf.get(rr * w + rc) || 0;
            if (!main) for (let k = 1; k <= parts; k++) if (!main || sizesOf[k] > sizesOf[main]) main = k;
            const touches = new Uint8Array(parts + 1);
            for (const [u, k] of pieceOf) {
              if (k === main) continue;
              const ux = u % w;
              for (const v of [ux > 0 ? u - 1 : -1, ux < w - 1 ? u + 1 : -1, u - w, u + w]) if (v >= 0 && v < n && labels[v] === id) touches[k] = 1;
            }
            for (const [u, k] of pieceOf) if (k !== main && touches[k]) labels[u] = id;
          }
          extra.push({ pos: [i + q, j], spot, size: seen.size });
          void left;
        }
      }
    }

    // Young trees (replants): a compact green canopy piece that the size or
    // height rules dropped, standing on an EMPTY planting position of a row,
    // is a young tree. It becomes a crown (flag YOUNG_TREE), so it is counted
    // and takes its place in the EVEN / ODD alternation.
    let gridSearch = null, fineSearch = null, rowWalk = null, darkSearch = null, darkWalk = null, midWalk = null;
    if (small) {
      await step("Young trees on empty planting positions …");
      const I = inv2(a, b);
      const occupied = new Set();
      const rowRange = new Map();
      const refs = [];
      for (let t = 0; t < count; t++) {
        if (!pos[t] || !alive[t]) continue;
        occupied.add(siteOf(pos[t]));
        refs.push(t);
        const [i, j] = pos[t], r = rowRange.get(j);
        rowRange.set(j, r ? [Math.min(r[0], i), Math.max(r[1], i)] : [i, i]);
      }
      for (const e of extra) occupied.add(siteOf(e.pos));
      const cellM = 2 * Math.max(inRow, len(b));
      const buckets = new Map();
      for (const t of refs) {
        const k = Math.floor(pts[t][0] / cellM) + "," + Math.floor(pts[t][1] / cellM);
        if (!buckets.has(k)) buckets.set(k, []);
        buckets.get(k).push(t);
      }
      const exgAt = (i) => { const R = rgba[4 * i], G = rgba[4 * i + 1], B = rgba[4 * i + 2]; return (2 * G - R - B) / (R + G + B + 1); };
      const lumaAt = (i) => 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2];
      // Green of sunlit pecan leaves in this image (lower quartile).
      const leaves = [];
      for (let i = 0; i < n; i += 7) if (mask[i] && lumaAt(i) >= thresholds.darkY) leaves.push(exgAt(i));
      const leafGreen = leaves.length ? percentile(leaves, 25) : 0.1;
      const cand = new Uint8Array(n);
      for (let i = 0; i < n; i++) cand[i] = small[i] && valid[i] && !mask[i] && !labels[i] && !trimmedAway[i] ? 1 : 0;
      const comp = label(cand, w, h);
      const m = comp.count;
      const cnt = new Int32Array(m + 1), sumX = new Float64Array(m + 1), sumY = new Float64Array(m + 1), sumL = new Float64Array(m + 1);
      for (let i = 0; i < n; i++) {
        const l = comp.labels[i];
        if (!l) continue;
        cnt[l]++; sumX[l] += i % w; sumY[l] += (i / w) | 0; sumL[l] += lumaAt(i);
      }
      const crownAreas = refs.map((t) => sizes[t + 1] * area);
      const maxArea = Math.min(30, Math.max(4, 0.5 * percentile(crownAreas, 50)));
      // Tree-sized and sunlit (a piece in the cast shadow of a big tree is
      // shadow over grass, not a young tree).
      const keepComp = new Uint8Array(m + 1);
      for (let l = 1; l <= m; l++) keepComp[l] = cnt[l] * area >= 0.6 && cnt[l] * area <= maxArea && sumL[l] / cnt[l] >= thresholds.darkY ? 1 : 0;
      const toEdge = edt(valid, w, h);
      const reach = new Float64Array(m + 1), edgeMin = new Float64Array(m + 1).fill(Infinity);
      const border = new Int32Array(m + 1), touch = new Int32Array(m + 1);
      const greens = new Map(), pix = new Map();
      const ringSeen = new Set();
      for (let i = 0; i < n; i++) {
        const l = comp.labels[i];
        if (!l || !keepComp[l]) continue;
        const x = i % w, y = (i / w) | 0;
        const d2 = (x - sumX[l] / cnt[l]) ** 2 + (y - sumY[l] / cnt[l]) ** 2;
        if (d2 > reach[l]) reach[l] = d2;
        if (toEdge[i] < edgeMin[l]) edgeMin[l] = toEdge[i];
        if (!greens.has(l)) { greens.set(l, []); pix.set(l, []); }
        greens.get(l).push(exgAt(i));
        pix.get(l).push(i);
        for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]) {
          if (j < 0 || comp.labels[j] === l || ringSeen.has(j)) continue;
          ringSeen.add(j);
          border[l]++;
          if (labels[j]) touch[l]++;
        }
      }
      // Green of the open ground (no crown) within rM metres around a piece.
      const ringGreens = (list, rM = 1.0) => {
        const r = Math.max(1, Math.round(rM / pixel));
        let x0 = w, y0 = h, x1 = 0, y1 = 0;
        for (const i of list) { const x = i % w, y = (i / w) | 0; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
        x0 = Math.max(0, x0 - r - 1); y0 = Math.max(0, y0 - r - 1); x1 = Math.min(w - 1, x1 + r + 1); y1 = Math.min(h - 1, y1 + r + 1);
        const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
        const own = new Uint8Array(bw * bh);
        for (const i of list) own[(((i / w) | 0) - y0) * bw + (i % w) - x0] = 1;
        const grown = dilate(own, bw, bh, r);
        const out = [];
        out.luma = [];
        for (let k = 0; k < bw * bh; k++) {
          if (!grown[k] || own[k]) continue;
          const i = (((k / bw) | 0) + y0) * w + (k % bw) + x0;
          if (valid[i] && !labels[i]) { out.push(exgAt(i)); out.luma.push(lumaAt(i)); }
        }
        return out;
      };
      const best = new Map();
      for (let l = 1; l <= m; l++) {
        if (!keepComp[l]) continue;
        const A = cnt[l] * area;
        if (Math.sqrt(reach[l]) * pixel > 0.45 * inRow) continue;       // compact, not a strip
        if (cnt[l] / (Math.PI * Math.max(reach[l], 1)) < 0.35) continue; // round like a crown
        if (touch[l] > 0.15 * border[l]) continue;                      // free-standing, not the edge of a big crown
        if (edgeMin[l] * pixel < 0.5 * inRow) continue;                // not on the field edge
        if (percentile(greens.get(l), 75) < leafGreen) continue;       // as green as pecan leaves
        // A young tree stands out from the bare tree strip around it; a
        // greener spot inside grass does not (ground 1 m around it).
        const ring = ringGreens(pix.get(l));
        if (ring.length) {
          const own25 = percentile(greens.get(l), 25), own75 = percentile(greens.get(l), 75);
          const asGreen = ring.filter((v) => v >= own25).length / ring.length;
          if (asGreen > 0.12 || own75 - percentile(ring, 50) < 0.12) continue;
          // Never ground: a young tree stands on clean bare ground, or it
          // casts its own shadow beside it (a plant that stands up).
          const ownShadow = ring.luma.filter((v) => v < 0.6 * (sumL[l] / cnt[l])).length / ring.length;
          if (asGreen > 0.05 && ownShadow < 0.05) continue;
        } else continue;
        const spot = [(sumX[l] / cnt[l] + 0.5) * pixel, (h - sumY[l] / cnt[l] - 0.5) * pixel];
        const near = [];
        const bx = Math.floor(spot[0] / cellM), by = Math.floor(spot[1] / cellM);
        for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
          for (const t of buckets.get((bx + dx) + "," + (by + dy)) || []) {
            const d = Math.hypot(pts[t][0] - spot[0], pts[t][1] - spot[1]);
            if (d < 1.5 * cellM) near.push([d, t]);
          }
        }
        if (near.length < 2) continue;
        near.sort((p, q) => p[0] - q[0]);
        const us = [], vs = [];
        for (const [, t] of near.slice(0, 6)) {
          const ex = spot[0] - pts[t][0], ey = spot[1] - pts[t][1];
          us.push(pos[t][0] + I[0][0] * ex + I[0][1] * ey);
          vs.push(pos[t][1] + I[1][0] * ex + I[1][1] * ey);
        }
        const u = percentile(us, 50), v = percentile(vs, 50);
        const p = Math.round(u), q = Math.round(v);
        if (Math.abs(u - p) > 0.3 || Math.abs(v - q) > 0.25) continue;  // on a planting position
        const range = rowRange.get(q);
        if (!range || p < range[0] - 1 || p > range[1] + 1) continue;   // in a planted row
        const key = p + "," + q;
        if (occupied.has(key)) continue;                                // that position is empty
        const prev = best.get(key);
        if (!prev || A > prev.A) best.set(key, { l, A, pos: [p, q], spot });
      }
      const idOf = new Int32Array(m + 1);
      for (const c of best.values()) {
        idOf[c.l] = count + extra.length + 1;
        extra.push({ pos: c.pos, spot: c.spot, size: cnt[c.l], young: true });
      }
      for (let i = 0; i < n; i++) {
        const id = idOf[comp.labels[i]];
        if (id) { labels[i] = id; mask[i] = 1; }
      }

      // Every empty planting position of the 40 × 20 ft grid is searched for
      // a young tree (a sapling is often too small or too low for the steps
      // above). Where the tree should stand comes from its planted
      // neighbours. A young tree is a spot of leaves clearly greener than
      // the ground around it, with little green right beside it: an island
      // of leaves on the tree strip, never grass, soil or shadow.
      // The grid also checks the image: the same search is run halfway
      // between planting positions, where nothing was planted. Saplings are
      // found on the planting positions; weeds turn up just as often in
      // between, and then none of the spots is counted.
      for (const c of best.values()) occupied.add(siteOf(c.pos));
      const placed = new Map();
      for (const t of refs) placed.set(siteOf(pos[t]), pts[t]);
      for (const e of extra) placed.set(siteOf(e.pos), e.spot);
      const rowSp = len(b);
      // search radius: a quarter of the tree spacing; a third where the
      // trees stand irregularly along their rows (positions less exact)
      const fitErr = percentile(Array.from(gp.fitM).filter(Number.isFinite), 50) || 0;
      const rIn = (fitErr > 0.12 * inRow ? 0.35 : 0.25) * inRow, rOut = rIn + 1.5, R = Math.ceil(rOut / pixel), bw = 2 * R + 1;
      const free = (k) => valid[k] && !labels[k] && !trimmedAway[k];
      const compOf = new Int32Array(bw * bw);
      // where a tree at grid position (p, q) stands (p may be fractional)
      // Rows searched for young trees: every row of the block, along its
      // whole length (as long as the rows beside it), also a row with few
      // or no big trees (a row of replants). Spots outside the field are
      // skipped by the search itself.
      const searchRows = new Map();
      {
        const qs = [...rowRange.keys()];
        const qLo = Math.min(...qs), qHi = Math.max(...qs);
        for (let q = qLo - 1; q <= qHi + 1; q++) {
          let lo = Infinity, hi = -Infinity;
          for (let dq = -2; dq <= 2; dq++) { const r = rowRange.get(q + dq); if (r) { lo = Math.min(lo, r[0]); hi = Math.max(hi, r[1]); } }
          if (lo <= hi) searchRows.set(q, [lo, hi]);
        }
      }
      const acr = [-a[1] / inRow, a[0] / inRow];   // unit vector across the rows
      const spotAt = (p, q) => {
        const est = [], p0 = Math.round(p);
        // nearest planted trees; further out where the row has few of them
        for (const [rq, rp] of [[1, 3], [2, 6]]) {
          for (let dq = -rq; dq <= rq; dq++) for (let dp = -rp; dp <= rp; dp++) {
            const s = placed.get((p0 + dp) + "," + (q + dq));
            if (!s) continue;
            const fp = p - (p0 + dp), fq = -dq;
            est.push([Math.abs(fp) * inRow + Math.abs(fq) * rowSp, s[0] + fp * a[0] + fq * b[0], s[1] + fp * a[1] + fq * b[1]]);
          }
          if (est.length >= 2) break;
          est.length = 0;
        }
        if (est.length < 2) return null;
        est.sort((e1, e2) => e1[0] - e2[0]);
        const use = est.slice(0, 6);
        let sx = percentile(use.map((e) => e[1]), 50), sy = percentile(use.map((e) => e[2]), 50);
        // Across the row the position follows the trees of its own row (the
        // rows beside it may not line up exactly with it).
        const own = [];
        for (let dp = 1; dp <= 8 && own.length < 4; dp++) {
          for (const pp of [p0 - dp, p0 + dp, dp === 1 ? p0 : null]) {
            if (pp === null) continue;
            const s = placed.get(pp + "," + q);
            if (s) own.push(s[0] * acr[0] + s[1] * acr[1]);
          }
        }
        if (own.length) {
          const shift = percentile(own, 50) - (sx * acr[0] + sy * acr[1]);
          if (Math.abs(shift) < 0.4 * rowSp) { sx += shift * acr[0]; sy += shift * acr[1]; }
        }
        return [sx, sy];
      };
      // Direction in which shadows fall in this image, from the big crowns:
      // the dark ground 2 m from their edges (null when shadows are not clear).
      const shadowDir = (() => {
        const ys = [];
        for (let i = 0; i < n; i += 13) if (valid[i] && !labels[i]) ys.push(lumaAt(i));
        if (ys.length < 100) return null;
        const groundY = percentile(ys, 50), s2 = 2.0 / pixel;
        const darkFrac = (ang) => {
          const dx = Math.cos(ang), dy = -Math.sin(ang);
          let hit = 0, all = 0;
          for (let y = 0; y < h; y += 3) for (let x = 0; x < w; x += 3) {
            const l = labels[y * w + x];
            if (!l || l > count) continue;
            const tx = Math.round(x + dx * s2), ty = Math.round(y + dy * s2);
            if (tx < 0 || ty < 0 || tx >= w || ty >= h) continue;
            const k = ty * w + tx;
            all++;
            if (valid[k] && !labels[k] && lumaAt(k) < 0.6 * groundY) hit++;
          }
          return all ? hit / all : 0;
        };
        let best = 0, bestAng = 0;
        for (let deg = 0; deg < 360; deg += 10) { const f = darkFrac((deg * Math.PI) / 180); if (f > best) { best = f; bestAng = deg; } }
        const opposite = darkFrac(((bestAng + 180) * Math.PI) / 180);
        if (best < 0.05 || best < 1.8 * opposite) return null;
        const ang = (bestAng * Math.PI) / 180;
        return [Math.cos(ang), -Math.sin(ang)];
      })();
      if (opts.debugFine) opts.debugFine("shadowDir", shadowDir, null, 0, false, null);   // testing only
      // A young tree stands up: right beside it, on the side shadows fall,
      // lies its own compact shadow (not part of a big tree's shadow).
      const ownShadow = (inList, mx2, my2, bgYmed) => {
        if (!shadowDir) return false;
        const rad = Math.sqrt(inList.size / Math.PI), reach = rad + 1.5 / pixel, box = Math.ceil(4.0 / pixel);
        const isDark = (k) => valid[k] && !labels[k] && !inList.has(k) && lumaAt(k) < 0.6 * bgYmed;
        const seeds = [];
        for (let y = Math.max(0, Math.floor(my2 - reach)); y <= Math.min(h - 1, Math.ceil(my2 + reach)); y++) {
          for (let x = Math.max(0, Math.floor(mx2 - reach)); x <= Math.min(w - 1, Math.ceil(mx2 + reach)); x++) {
            const ddx = x - mx2, ddy = y - my2, d = Math.hypot(ddx, ddy);
            if (d > reach || d < 0.5 * rad) continue;
            if ((ddx * shadowDir[0] + ddy * shadowDir[1]) / Math.max(d, 1e-6) < Math.cos((40 * Math.PI) / 180)) continue;
            if (isDark(y * w + x)) seeds.push(y * w + x);
          }
        }
        if (!seeds.length) return false;
        const seen = new Set(seeds), stack = seeds.slice();
        let touchesCrown = false;
        while (stack.length) {
          const u = stack.pop(), ux = u % w, uy = (u / w) | 0;
          if (Math.abs(ux - mx2) > box || Math.abs(uy - my2) > box) return false; // runs on: a big shadow
          for (const v of [ux > 0 ? u - 1 : -1, ux < w - 1 ? u + 1 : -1, u - w, u + w]) {
            if (v < 0 || v >= n || seen.has(v)) continue;
            if (labels[v]) { touchesCrown = true; continue; }
            if (isDark(v)) { seen.add(v); stack.push(v); }
          }
        }
        const A = seen.size * area;
        return !touchesCrown && A >= Math.max(0.15, 0.3 * inList.size * area) && A <= 12;
      };
      // the sapling searched within rIn of a spot (null: none, or not
      // searchable); L = the rules
      // Neighbours of a leaf pixel. On a sharp image (under 12 cm) the
      // leaves of one young tree are parted by bits of soil: leaves up to
      // 15 cm apart belong together.
      const hop = [];
      {
        if (pixel > 0.12) hop.push([-1, 0], [1, 0], [0, -1], [0, 1]);
        else for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if (dx || dy) hop.push([dx, dy]);  // touching corners too
      }
      // distance (pixels) from every open pixel to the nearest crown
      const toCrown = (() => { const open = new Uint8Array(n); for (let i = 0; i < n; i++) open[i] = labels[i] ? 0 : 1; return edt(open, w, h); })();
      const findSapling = (sx, sy, L) => {
          const cc = Math.floor(sx / pixel), rr = h - 1 - Math.floor(sy / pixel);
          if (cc < R || rr < R || cc >= w - R || rr >= h - R) return null;
          if (!valid[rr * w + cc] || toEdge[rr * w + cc] * pixel < 2.0) return null;
          // the ground 1.5 m around the search circle
          const bg = [], bgY = [];
          let annulus = 0;
          for (let y = rr - R; y <= rr + R; y++) for (let x = cc - R; x <= cc + R; x++) {
            const d = Math.hypot(x - cc, y - rr) * pixel;
            if (d <= rIn || d > rOut) continue;
            annulus++;
            const k = y * w + x;
            if (free(k)) { bg.push(exgAt(k)); bgY.push(lumaAt(k)); }
          }
          if (bg.length < 0.4 * annulus) return null;                     // mostly crowns around
          const bgYmed = percentile(bgY, 50);
          if (bgYmed < thresholds.darkY) return null;                     // in a big tree's shadow
          const bg50 = percentile(bg, 50);
          // leaf pixels: greener than most of the ground (capped, so that
          // lush grass beside a narrow tree strip does not set the bar)
          const thr = Math.max(Math.min(percentile(bg, 75) + L.above, bg50 + 0.1), L.leafMin * leafGreen);
          // (a young crown with its own shadow may also be pale: lighter
          // than the ground and at least as green)
          const isLeaf = L.needShadow
            ? (k) => free(k) && (exgAt(k) >= thr || (lumaAt(k) >= 1.12 * bgYmed && exgAt(k) >= bg50 + 0.02))
            : (k) => free(k) && exgAt(k) >= thr;
          compOf.fill(0);
          let id2 = 0, pick = null;
          for (let y = rr - R; y <= rr + R; y++) for (let x = cc - R; x <= cc + R; x++) {
            const k0 = (y - rr + R) * bw + (x - cc + R);
            if (compOf[k0] || Math.hypot(x - cc, y - rr) * pixel > rIn || !isLeaf(y * w + x)) continue;
            compOf[k0] = ++id2;
            // leaves within 0.6 m of the search circle; green going on
            // beyond that is a band of grass or weeds, not a sapling
            const list = [], stack = [y * w + x];
            let leak = 0;
            while (stack.length) {
              const u = stack.pop(), ux = u % w, uy = (u / w) | 0;
              list.push(u);
              for (const [vx, vy] of hop.map(([dx, dy]) => [ux + dx, uy + dy])) {
                const v = vy * w + vx;
                if (vx < cc - R || vx > cc + R || vy < rr - R || vy > rr + R) { leak++; continue; }
                const kk = (vy - rr + R) * bw + (vx - cc + R);
                if (compOf[kk] || !isLeaf(v)) continue;
                if (Math.hypot(vx - cc, vy - rr) * pixel > rIn + 0.6) { leak++; continue; }
                compOf[kk] = id2;
                stack.push(v);
              }
            }
            const A = list.length * area;
            const why = opts.debugWhy && A >= 0.15 ? (reason) => opts.debugWhy(cc, rr, reason, A, list) : () => {};   // testing only
            if (A < Math.max(L.minArea, 2 * area) || A > maxArea) { why("size"); continue; } // sapling-sized
            const g = list.map(exgAt);
            const gmax = Math.max(...g);
            if (gmax - bg50 < L.contrast) { why("contrast"); continue; }   // much greener than the ground
            if (percentile(g, 75) < L.green * leafGreen) { why("green"); continue; } // as green as pecan leaves
            let mx2 = 0, my2 = 0;
            for (const u of list) { mx2 += u % w; my2 += (u / w) | 0; }
            mx2 /= list.length; my2 /= list.length;
            let reach2 = 0, edge2 = 0, touch2 = 0;
            const inList = new Set(list);
            for (const u of list) {
              const ux = u % w, uy = (u / w) | 0;
              reach2 = Math.max(reach2, (ux - mx2) ** 2 + (uy - my2) ** 2);
              for (const v of [u - 1, u + 1, u - w, u + w]) {
                if (inList.has(v)) continue;
                edge2++;
                if (labels[v]) touch2++;
              }
            }
            if (Math.sqrt(reach2) * pixel > 0.45 * inRow) { why("strip"); continue; } // compact, not a strip
            if (list.length / (Math.PI * Math.max(reach2, 1)) < (hop.length > 4 ? 0.15 : L.round)) { why("round"); continue; } // (leaves with soil between: less filled)
            if (touch2 > 0.15 * edge2) { why("touch"); continue; }        // not the edge of a big crown
            let gapC = Infinity, lum = 0;
            for (const u of list) { if (toCrown[u] < gapC) gapC = toCrown[u]; lum += lumaAt(u); }
            if (gapC * pixel < 0.4) { why("nearcrown"); continue; }       // stands apart from the big crowns
            lum /= list.length;
            if (lum < (L.darkMin ?? 1) * thresholds.darkY || lum < (L.sunlit ?? 0.7) * bgYmed) { why("dark " + Math.round(lum) + "/" + Math.round(bgYmed) + "/" + Math.round(thresholds.darkY)); continue; } // sunlit, not leaves in a shadow
            if (leak > (L.leak * edge2 * hop.length) / 4) { why("leak"); continue; } // not part of a band of grass
            const ring = ringGreens(list, 0.6);
            if (!ring.length) { why("noring"); continue; }
            // on mostly bare ground: a greener patch with grass all round is grass
            const own50 = percentile(g, 50);
            if (ring.filter((v) => v >= 0.6 * own50).length > 0.5 * ring.length) { why("grassring"); continue; }
            const own25 = percentile(g, 25);
            const asGreen = ring.filter((v) => v >= own25).length / ring.length;
            if (asGreen > L.asGreen) { why("asgreen " + asGreen.toFixed(2)); continue; } // an island of leaves
            if (L.needShadow && !ownShadow(inList, mx2, my2, bgYmed)) { why("noshadow"); continue; } // it stands: its own shadow
            why("OK");
            const score = gmax - bg50;
            if (!pick || score > pick.score) pick = { list, score, spot: [(mx2 + 0.5) * pixel, (h - my2 - 0.5) * pixel] };
          }
          return { pick, near: !!pick && Math.hypot(pick.spot[0] - sx, pick.spot[1] - sy) <= rIn };
      };
      // (Looser rules were tried on Barber and on Fine detail: the spots
      // they add turn up nearly as often between positions, i.e. weeds.)
      const L = { above: 0.04, leafMin: 0.5, minArea: 0.25, contrast: 0.14, green: 0.75, round: 0.3, leak: 0.15, asGreen: 0.15 };
      // second chance: a fainter or paler young crown that casts its own shadow
      const L2 = { above: 0.03, leafMin: 0.3, minArea: 0.2, contrast: 0.07, green: 0.4, round: 0.3, leak: 0.15, asGreen: 0.15, needShadow: true };
      const search = (sx, sy) => {
        const r = findSapling(sx, sy, L);
        if (!r || r.pick || !shadowDir) return r;
        const r2 = findSapling(sx, sy, L2);
        return r2 && r2.pick ? r2 : r;
      };
      const found = [];
      const check = { on: 0, onFound: 0, between: 0, betweenFound: 0 };
      const betweenPicks = new Map(), searchedBetween = [];
      // Only where the measured grid is the planting design (within 25 %):
      // otherwise the "planting positions" are not where trees were planted.
      const gridOk = !Array.isArray(design) || (Math.abs(inRow / (design[0] * 0.3048) - 1) <= 0.25 && Math.abs(rowSp / (design[1] * 0.3048) - 1) <= 0.4);
      for (const [q, [lo, hi]] of gridOk ? searchRows : []) {
        for (let p = lo - 1; p <= hi + 1; p++) {
          if (occupied.has(p + "," + q)) continue;
          const s = spotAt(p, q);
          const r = s && search(s[0], s[1]);
          if (!r) continue;
          check.on++;
          if (r.near) check.onFound++;
          if (r.pick) found.push({ pos: [p, q], pick: r.pick });
          if (opts.debugYoung) opts.debugYoung("spot", s, s, 0, !!r.pick);
        }
        for (let p = lo - 1; p <= hi; p++) {
          const s = spotAt(p + 0.5, q);
          const r = s && search(s[0], s[1]);
          if (!r) continue;
          searchedBetween.push([p, q]);
          if (r.near) betweenPicks.set(p + "," + q, { p, q, pick: r.pick });
          if (opts.debugYoung && r.pick) opts.debugYoung("between", s, r.pick.spot, r.pick.list.length * area, r.near);
        }
      }
      if (opts.debugYoung) for (const f of found) opts.debugYoung("on", spotAt(f.pos[0], f.pos[1]), f.pick.spot, f.pick.list.length * area, Math.hypot(f.pick.spot[0] - spotAt(f.pos[0], f.pos[1])[0], f.pick.spot[1] - spotAt(f.pos[0], f.pos[1])[1]) <= rIn);
      // Young trees one tree spacing apart halfway between the predicted
      // positions are a row planted half a spacing off the grid of the old
      // trees (a run of young trees), not weeds: weeds lie at random.
      const inRun = (b2) => betweenPicks.has((b2.p - 1) + "," + b2.q) || betweenPicks.has((b2.p + 1) + "," + b2.q);
      const runs = [...betweenPicks.values()].filter(inRun);
      // Weed evidence only between two positions that both have a tree: next
      // to an empty position, green there is that position's young tree
      // planted a little off its spot.
      const hasTree = (p, q) => occupied.has(p + "," + q);
      const closed = (p, q) => hasTree(p, q) && hasTree(p + 1, q);
      check.between = searchedBetween.filter(([p, q]) => closed(p, q)).length;
      check.betweenFound = [...betweenPicks.values()].filter((b2) => !inRun(b2) && closed(b2.p, b2.q)).length;
      const rateOn = check.onFound / Math.max(1, check.on), rateBetween = check.betweenFound / Math.max(1, check.between);
      gridSearch = { ...check, gridOk, runs: runs.length, counted: gridOk && check.between >= 10 && check.onFound >= 3 && rateBetween <= 0.25 * rateOn, added: 0 };
      if (gridSearch.counted) {
        const taken = new Set(occupied);
        const commit = (pos2, pick) => {
          if (pick.list.some((u) => labels[u])) return;
          const id = count + extra.length + 1;
          for (const u of pick.list) { labels[u] = id; mask[u] = 1; }
          extra.push({ pos: pos2, spot: pick.spot, size: pick.list.length, young: true });
          taken.add(siteOf(pos2));
          gridSearch.added++;
        };
        for (const f of found) commit(f.pos, f.pick);
        // a run's young tree takes the nearer free planting position
        for (const b2 of runs.sort((x, y) => x.q - y.q || x.p - y.p)) {
          const p2 = [b2.p, b2.p + 1].find((pp) => !taken.has(pp + "," + b2.q));
          if (p2 !== undefined) commit([p2, b2.q], b2.pick);
        }
        // a young tree planted a little off its spot (sizeable, standing
        // apart) takes the nearer of the two planting positions beside it
        // when that one is still empty
        for (const b2 of betweenPicks.values()) {
          if (b2.pick.list.length * area < 0.6) continue;
          const sL = spotAt(b2.p, b2.q), sR = spotAt(b2.p + 1, b2.q);
          if (!sL || !sR) continue;
          const dL = Math.hypot(b2.pick.spot[0] - sL[0], b2.pick.spot[1] - sL[1]), dR = Math.hypot(b2.pick.spot[0] - sR[0], b2.pick.spot[1] - sR[1]);
          const p2 = dL <= dR ? b2.p : b2.p + 1;
          if (!taken.has(p2 + "," + b2.q)) commit([p2, b2.q], b2.pick);
        }
      }

      // Dark young trees: in green weeds a young pecan is no greener than
      // the weeds, but it is a dark, round, compact bush (its own shade and
      // shadow) standing alone on the strip. Looked for at every empty
      // planting position; the 40 × 20 ft spacing is the check: counted
      // only when such bushes stand on the planting positions far more
      // often than halfway between them (weeds and grass lie anywhere).
      let findDark = null;
      if (gridOk) {
        await step("Young trees: dark young crowns on the planting positions …");
        const occ2 = new Set(occupied);
        for (const e of extra) if (e.pos) occ2.add(siteOf(e.pos));
        const rD = (opts.darkRadius || 0.45) * inRow;   // search radius for dark crowns (checked halfway between with the same radius)
        findDark = (sx, sy) => {
          const cc = Math.floor(sx / pixel), rr = h - 1 - Math.floor(sy / pixel);
          if (cc < R || rr < R || cc >= w - R || rr >= h - R) return null;
          if (!valid[rr * w + cc] || toEdge[rr * w + cc] * pixel < 2.0) return null;
          const gYs = [];
          let ann = 0;
          for (let y = rr - R; y <= rr + R; y++) for (let x = cc - R; x <= cc + R; x++) {
            const d = Math.hypot(x - cc, y - rr) * pixel;
            if (d <= rD || d > rOut) continue;
            ann++;
            const k = y * w + x;
            if (free(k)) gYs.push(lumaAt(k));
          }
          if (gYs.length < 0.4 * ann) return null;
          const gY = percentile(gYs, 50);
          if (gY < thresholds.darkY) return null;
          const isD = (k) => free(k) && lumaAt(k) < 0.7 * gY && exgAt(k) >= 0.03;
          compOf.fill(0);
          let id2 = 0, pick = null;
          for (let y = rr - R; y <= rr + R; y++) for (let x = cc - R; x <= cc + R; x++) {
            const k0 = (y - rr + R) * bw + (x - cc + R);
            if (compOf[k0] || Math.hypot(x - cc, y - rr) * pixel > rD || !isD(y * w + x)) continue;
            compOf[k0] = ++id2;
            const list = [], stack = [y * w + x];
            let leak = 0;
            while (stack.length) {
              const u = stack.pop(), ux = u % w, uy = (u / w) | 0;
              list.push(u);
              for (const [vx, vy] of [[ux - 1, uy], [ux + 1, uy], [ux, uy - 1], [ux, uy + 1]]) {
                if (vx < cc - R || vx > cc + R || vy < rr - R || vy > rr + R) { leak++; continue; }
                const kk = (vy - rr + R) * bw + (vx - cc + R), v = vy * w + vx;
                if (compOf[kk] || !isD(v)) continue;
                if (Math.hypot(vx - cc, vy - rr) * pixel > rD + 0.6) { leak++; continue; }
                compOf[kk] = id2;
                stack.push(v);
              }
            }
            const A = list.length * area;
            if (A < 0.25 || A > 3.0) continue;                                   // bush-sized
            let mx2 = 0, my2 = 0, lum = 0, eg = 0, clear = Infinity;
            for (const u of list) { mx2 += u % w; my2 += (u / w) | 0; lum += lumaAt(u); eg += exgAt(u); if (toCrown[u] < clear) clear = toCrown[u]; }
            mx2 /= list.length; my2 /= list.length; lum /= list.length; eg /= list.length;
            if (clear * pixel < 2.0) continue;                                    // well away from big trees and their shadows
            if (lum > 0.6 * gY || eg < 0.04) continue;                            // clearly dark, green (not mud)
            {
              // its dark area (any colour) stays small and never reaches a big
              // tree: otherwise it is a piece of that tree's cast shadow
              const seenD = new Set(list), stD = list.slice();
              let partOfShadow = false;
              while (stD.length && !partOfShadow) {
                const u = stD.pop(), ux = u % w, uy = (u / w) | 0;
                for (const [vx, vy] of [[ux - 1, uy], [ux + 1, uy], [ux, uy - 1], [ux, uy + 1]]) {
                  if (vx < 0 || vy < 0 || vx >= w || vy >= h) continue;
                  const v = vy * w + vx;
                  if (seenD.has(v) || !valid[v]) continue;
                  if (labels[v]) { partOfShadow = true; break; }
                  if (lumaAt(v) >= 0.7 * gY) continue;
                  seenD.add(v);
                  if (seenD.size * area > 4.0) { partOfShadow = true; break; }
                  stD.push(v);
                }
              }
              if (partOfShadow) continue;
            }
            let reach2 = 0, edge2 = 0, touch2 = 0;
            const inList = new Set(list);
            for (const u of list) {
              const ux = u % w, uy = (u / w) | 0;
              reach2 = Math.max(reach2, (ux - mx2) ** 2 + (uy - my2) ** 2);
              for (const v of [u - 1, u + 1, u - w, u + w]) { if (inList.has(v)) continue; edge2++; if (labels[v]) touch2++; }
            }
            if (Math.sqrt(reach2) * pixel > 1.0) continue;                       // compact
            if (list.length / (Math.PI * Math.max(reach2, 1)) < 0.35) continue;   // round
            if (touch2 > 0.1 * edge2 || leak > 0.15 * edge2) continue;            // standing alone
            const ring = ringGreens(list, 0.5);
            if (!ring.length || ring.luma.filter((v) => v < 0.7 * gY).length > 0.25 * ring.length) continue; // not a dark patch in tall grass
            const spot = [(mx2 + 0.5) * pixel, (h - my2 - 0.5) * pixel];
            if (!pick || A > pick.A) pick = { list, spot, A };
          }
          return { pick, near: !!pick && Math.hypot(pick.spot[0] - sx, pick.spot[1] - sy) <= rD };
        };
        const foundD = [], chk = { on: 0, onFound: 0, between: 0, betweenFound: 0 };
        for (const [q, [lo, hi]] of searchRows) {
          for (let p = lo - 1; p <= hi + 1; p++) {
            if (occ2.has(p + "," + q)) continue;
            const s = spotAt(p, q);
            const r = s && findDark(s[0], s[1]);
            if (!r) continue;
            chk.on++;
            if (r.near) { chk.onFound++; foundD.push({ pos: [p, q], pick: r.pick }); }
            if (opts.debugYoung && r.pick) opts.debugYoung("don", s, r.pick.spot, r.pick.A, r.near);
          }
          for (let p = lo - 1; p <= hi; p++) {
            if (!(occ2.has(p + "," + q) && occ2.has((p + 1) + "," + q))) continue;   // between two trees
            const s = spotAt(p + 0.5, q);
            const r = s && findDark(s[0], s[1]);
            if (!r) continue;
            chk.between++;
            if (r.near) chk.betweenFound++;
            if (opts.debugYoung && r.pick) opts.debugYoung("dbetween", s, r.pick.spot, r.pick.A, r.near);
          }
        }
        const rOn = chk.onFound / Math.max(1, chk.on), rBe = chk.betweenFound / Math.max(1, chk.between);
        darkSearch = { ...chk, counted: chk.between >= 10 && chk.onFound >= 3 && rBe <= 0.25 * rOn, added: 0 };
        if (darkSearch.counted) {
          for (const f of foundD) {
            if (f.pick.list.some((u) => labels[u]) || occ2.has(siteOf(f.pos))) continue;
            const id = count + extra.length + 1;
            for (const u of f.pick.list) { labels[u] = id; mask[u] = 1; }
            extra.push({ pos: f.pos, spot: f.pick.spot, size: f.pick.list.length, young: true });
            occ2.add(siteOf(f.pos));
            darkSearch.added++;
          }
        }
      }

      // Full-resolution look. A young crown is only a few pixels on the
      // analysis grid, so at every planting position still empty the
      // original image (about 5 cm) is read around the spot, and a young
      // crown is a cluster of leaf pixels standing alone on the ground (not
      // a band of grass, not shadow, not soil). Checked against the
      // positions in between like the search above, and added to it.
      const C = opts.chip;
      if (C && C.read && gridOk && shadowDir && C.width && C.height) {
        await step("Young trees: full-resolution look at the empty planting positions …");
        const sxI = C.width / w, syI = C.height / h;               // image pixels per grid pixel
        const native = pixel / Math.sqrt(sxI * syI);
        const kf = Math.max(1, Math.round(0.05 / native)), pf = native * kf;
        const half = rIn + 2.0;
        const done = new Set(occupied);
        for (const e of extra) if (e.pos) done.add(siteOf(e.pos));
        for (const e of extra) if (e.pos) placed.set(siteOf(e.pos), e.spot);
        const look = async (sx, sy) => {
          const gc = sx / pixel, gr = h - sy / pixel;
          const ci = Math.floor(gc), ri = Math.floor(gr);
          if (ci < 0 || ri < 0 || ci >= w || ri >= h || !valid[ri * w + ci] || toEdge[ri * w + ci] * pixel < 2.0) return null;
          const halfI = half / native;
          let x0 = Math.max(0, Math.floor(gc * sxI - halfI)), y0 = Math.max(0, Math.floor(gr * syI - halfI));
          let x1 = Math.min(C.width, Math.ceil(gc * sxI + halfI)), y1 = Math.min(C.height, Math.ceil(gr * syI + halfI));
          const fw = Math.floor((x1 - x0) / kf), fh = Math.floor((y1 - y0) / kf);
          if (fw < 8 || fh < 8) return null;
          x1 = x0 + fw * kf; y1 = y0 + fh * kf;
          const bands = await C.read(x0, y0, x1, y1);
          const fg = createGrid(x1 - x0, y1 - y0, fw, fh, { hasAlpha: !!C.hasAlpha, nodata: C.nodata, bits: 8 });
          fg.addRows(0, y1 - y0, bands);
          const F = fg.finish();
          const nf = fw * fh;
          const gIdx = new Int32Array(nf), mx = new Float64Array(nf), my = new Float64Array(nf);
          const free2 = new Uint8Array(nf), ex = new Float32Array(nf), lu = new Float32Array(nf);
          for (let fy = 0; fy < fh; fy++) for (let fx = 0; fx < fw; fx++) {
            const k = fy * fw + fx, ix = x0 + (fx + 0.5) * kf, iy = y0 + (fy + 0.5) * kf;
            const gx = Math.min(w - 1, Math.floor(ix / sxI)), gy = Math.min(h - 1, Math.floor(iy / syI)), g = gy * w + gx;
            gIdx[k] = g; mx[k] = (ix / sxI) * pixel; my[k] = (h - iy / syI) * pixel;
            const R = F[4 * k], G = F[4 * k + 1], B = F[4 * k + 2];
            ex[k] = (2 * G - R - B) / (R + G + B + 1); lu[k] = 0.299 * R + 0.587 * G + 0.114 * B;
            free2[k] = F[4 * k + 3] && valid[g] && !labels[g] && !trimmedAway[g] ? 1 : 0;
          }
          // leaf greenness smoothed over 3 x 3 pixels (single grass blades fade)
          const exS = new Float32Array(nf);
          for (let fy = 0; fy < fh; fy++) for (let fx = 0; fx < fw; fx++) {
            let s = 0, c = 0;
            for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
              const xx = fx + dx, yy = fy + dy;
              if (xx < 0 || yy < 0 || xx >= fw || yy >= fh) continue;
              s += ex[yy * fw + xx]; c++;
            }
            exS[fy * fw + fx] = s / c;
          }
          const dist = (k) => Math.hypot(mx[k] - sx, my[k] - sy);
          const bg = [], bgY = [];
          let ann = 0;
          for (let k = 0; k < nf; k++) {
            const d = dist(k);
            if (d <= rIn || d > rIn + 1.0) continue;
            ann++;
            if (free2[k]) { bg.push(exS[k]); bgY.push(lu[k]); }
          }
          if (bg.length < 0.4 * ann) return null;
          const bgYmed = percentile(bgY, 50);
          if (bgYmed < thresholds.darkY) return null;
          const bg50 = percentile(bg, 50);
          const thr = Math.max(Math.min(percentile(bg, 75) + 0.05, bg50 + 0.12), 0.5 * leafGreen);
          const leaf = (k) => free2[k] && exS[k] >= thr;
          const comp = new Int32Array(nf);
          let idc = 0, pick = null, pickB = null;
          const ringR = Math.max(1, Math.round(0.4 / pf));
          for (let k0 = 0; k0 < nf; k0++) {
            if (comp[k0] || !leaf(k0) || dist(k0) > rIn) continue;
            comp[k0] = ++idc;
            const list = [], stack = [k0];
            let leak = 0;
            while (stack.length) {
              const u = stack.pop(), ux = u % fw, uy = (u / fw) | 0;
              list.push(u);
              for (const [vx, vy] of [[ux - 1, uy], [ux + 1, uy], [ux, uy - 1], [ux, uy + 1]]) {
                if (vx < 0 || vy < 0 || vx >= fw || vy >= fh) { leak++; continue; }
                const v = vy * fw + vx;
                if (comp[v] || !leaf(v)) continue;
                if (dist(v) > rIn + 0.6) { leak++; continue; }
                comp[v] = idc;
                stack.push(v);
              }
            }
            const A = list.length * pf * pf;
            if (A < 0.15 || A > maxArea) continue;                              // young-crown sized
            const small = A < (opts.fineMinArea || 0.3);                        // (smaller: like a weed, unless it has a stem)
            const g = list.map((u) => exS[u]);
            const gmax = Math.max(...g);
            if (gmax - bg50 < 0.12) continue;                                   // much greener than the ground
            if (percentile(list.map((u) => ex[u]), 75) < 0.6 * leafGreen) continue; // leaf green
            let cx = 0, cy = 0;
            for (const u of list) { cx += u % fw; cy += (u / fw) | 0; }
            cx /= list.length; cy /= list.length;
            let reach2 = 0, edge = 0, touch = 0;
            const inL = new Set(list);
            let bx0 = fw, by0 = fh, bx1 = 0, by1 = 0;
            for (const u of list) {
              const ux = u % fw, uy = (u / fw) | 0;
              reach2 = Math.max(reach2, (ux - cx) ** 2 + (uy - cy) ** 2);
              if (ux < bx0) bx0 = ux; if (ux > bx1) bx1 = ux; if (uy < by0) by0 = uy; if (uy > by1) by1 = uy;
              for (const [vx, vy] of [[ux - 1, uy], [ux + 1, uy], [ux, uy - 1], [ux, uy + 1]]) {
                if (vx < 0 || vy < 0 || vx >= fw || vy >= fh) continue;
                const v = vy * fw + vx;
                if (inL.has(v)) continue;
                edge++;
                if (labels[gIdx[v]]) touch++;
              }
            }
            if (Math.sqrt(reach2) * pf > 0.45 * inRow) continue;               // compact, not a strip
            if (list.length / (Math.PI * Math.max(reach2, 1)) < 0.2) continue;
            if (touch > 0.15 * edge || leak > 0.15 * edge) continue;            // not a crown edge, not a band of grass
            // an island of leaves: little leaf-green ground within 0.4 m
            let ring = 0, ringGreen = 0, ringSoil = 0;
            for (let yy = Math.max(0, by0 - ringR); yy <= Math.min(fh - 1, by1 + ringR); yy++) {
              for (let xx = Math.max(0, bx0 - ringR); xx <= Math.min(fw - 1, bx1 + ringR); xx++) {
                const v = yy * fw + xx;
                if (inL.has(v) || !free2[v]) continue;
                let near2 = false;
                for (const u of list) { const ux = u % fw, uy = (u / fw) | 0; if ((ux - xx) ** 2 + (uy - yy) ** 2 <= ringR * ringR) { near2 = true; break; } }
                if (!near2) continue;
                ring++;
                if (exS[v] >= thr) ringGreen++;
                if (ex[v] < 0.05) ringSoil++;                                 // bare soil (sunlit or in its shadow), not green
              }
            }
            if (!ring || ringGreen > 0.2 * ring) continue;
            // On clean bare soil (a weeded tree strip): a lone green crown
            // there is a young tree (checked below against the positions in
            // between, separately from the rule with the shadow).
            const bare = ringSoil >= 0.6 * ring && list.length / (Math.PI * Math.max(reach2, 1)) >= 0.3;
            if (opts.debugFine) opts.debugFine("ring", [list.reduce((s, u) => s + mx[u], 0) / list.length, list.reduce((s, u) => s + my[u], 0) / list.length], null, A, bare, { soil: ringSoil / ring, round: list.length / (Math.PI * Math.max(reach2, 1)), green: ringGreen / ring });   // testing only
            // It stands up: its own compact shadow lies right beside it, on
            // the side the big trees' shadows fall (weeds and grass lie flat).
            const stands = (() => {
              const sdx = shadowDir[0], sdy = shadowDir[1];
              const rad = Math.sqrt(list.length / Math.PI), reachS = rad + 1.2 / pf;
              const isDark = (v) => free2[v] && !inL.has(v) && lu[v] < 0.55 * bgYmed;
              const seeds = [];
              for (let yy = Math.max(0, Math.floor(cy - reachS)); yy <= Math.min(fh - 1, Math.ceil(cy + reachS)); yy++) {
                for (let xx = Math.max(0, Math.floor(cx - reachS)); xx <= Math.min(fw - 1, Math.ceil(cx + reachS)); xx++) {
                  const ddx = xx - cx, ddy = yy - cy, d = Math.hypot(ddx, ddy);
                  if (d > reachS || d < 0.3 * rad) continue;
                  if ((ddx * sdx + ddy * sdy) / Math.max(d, 1e-6) < 0.7) continue;
                  const v = yy * fw + xx;
                  if (isDark(v)) seeds.push(v);
                }
              }
              if (!seeds.length) return false;
              const seen = new Set(seeds), st = seeds.slice();
              let bad = false;
              while (st.length && !bad) {
                const u = st.pop(), ux = u % fw, uy = (u / fw) | 0;
                for (const [vx, vy] of [[ux - 1, uy], [ux + 1, uy], [ux, uy - 1], [ux, uy + 1]]) {
                  if (vx < 0 || vy < 0 || vx >= fw || vy >= fh) { bad = true; break; }  // runs on: a big shadow
                  const v = vy * fw + vx;
                  if (seen.has(v) || inL.has(v)) continue;
                  if (labels[gIdx[v]]) { bad = true; break; }                          // a big tree's shadow
                  if (isDark(v)) { seen.add(v); st.push(v); }
                }
              }
              // a real tree shadow: dark, about the size of the crown (0.5 to
              // 2.5 times), and right next to it (measured on young pecans)
              const As = seen.size * pf * pf;
              if (bad) return false;
              const crownShadow = As >= 0.5 * A && As <= 2.5 * A && As <= 12 &&
                percentile([...seen].map((v) => lu[v]), 50) <= 0.45 * bgYmed &&
                [...seen].reduce((s, v) => s + ((v % fw) - cx) * sdx + (((v / fw) | 0) - cy) * sdy, 0) / seen.size / Math.max(rad, 1) <= 4;
              return crownShadow && !small;
            })();
            if (!stands && !bare) continue;
            const spot = [list.reduce((s, u) => s + mx[u], 0) / list.length, list.reduce((s, u) => s + my[u], 0) / list.length];
            const score = A * (gmax - bg50);
            const cand = { score, spot, cells: list.map((u) => gIdx[u]), A };
            if (stands && (!pick || score > pick.score)) pick = cand;
            if (bare && (!pickB || score > pickB.score)) pickB = cand;
          }
          const nearOf = (pk) => !!pk && Math.hypot(pk.spot[0] - sx, pk.spot[1] - sy) <= rIn;
          return { pick, near: nearOf(pick), pickB, nearB: nearOf(pickB) };
        };
        const found2 = [], picks2 = new Map(), found3 = [], picks3 = new Map();
        const check2 = { on: 0, onFound: 0, between: 0, betweenFound: 0 };
        const check3 = { on: 0, onFound: 0, between: 0, betweenFound: 0 };
        for (const [q, [lo, hi]] of searchRows) {
          for (let p = lo - 1; p <= hi + 1; p++) {
            if (done.has(p + "," + q)) continue;
            const s = spotAt(p, q);
            const r = s && await look(s[0], s[1]);
            if (!r) continue;
            check2.on++; check3.on++;
            if (r.near) check2.onFound++;
            if (r.nearB) check3.onFound++;
            if (r.pick) found2.push({ pos: [p, q], pick: r.pick });
            if (r.pickB) found3.push({ pos: [p, q], pick: r.pickB });
          }
          for (let p = lo - 1; p <= hi; p++) {
            if (betweenPicks.has(p + "," + q)) continue;     // green already found there by the search above
            const s = spotAt(p + 0.5, q);
            const r = s && await look(s[0], s[1]);
            if (!r) continue;
            check2.between++; check3.between++;
            if (r.near) picks2.set(p + "," + q, { p, q, pick: r.pick });
            if (r.nearB) picks3.set(p + "," + q, { p, q, pick: r.pickB });
          }
        }
        const inRun2 = (b2) => picks2.has((b2.p - 1) + "," + b2.q) || picks2.has((b2.p + 1) + "," + b2.q);
        const runs2 = [...picks2.values()].filter(inRun2);
        check2.betweenFound = picks2.size - runs2.length;
        const rate2On = check2.onFound / Math.max(1, check2.on), rate2Between = check2.betweenFound / Math.max(1, check2.between);
        fineSearch = { ...check2, runs: runs2.length, counted: check2.between >= 20 && check2.onFound >= 3 && rate2Between <= 0.25 * rate2On, added: 0 };
        // the lone green crowns on clean bare soil: their own check
        const inRun3 = (b3) => picks3.has((b3.p - 1) + "," + b3.q) || picks3.has((b3.p + 1) + "," + b3.q);
        const runs3 = [...picks3.values()].filter(inRun3);
        check3.betweenFound = picks3.size - runs3.length;
        const rate3On = check3.onFound / Math.max(1, check3.on), rate3Between = check3.betweenFound / Math.max(1, check3.between);
        fineSearch.bare_soil = { found_on_positions: check3.onFound, found_between: check3.betweenFound, counted: check3.between >= 20 && check3.onFound >= 3 && rate3Between <= 0.25 * rate3On, added: 0 };
        {
          const commit2 = (pos2, pick) => {
            if (done.has(siteOf(pos2))) return;
            const cover = new Map();
            for (const g of pick.cells) cover.set(g, (cover.get(g) || 0) + 1);
            const cells = [...cover.entries()].filter(([g, c]) => !labels[g] && c * pf * pf >= 0.3 * area).map(([g]) => g);
            if (!cells.length) {
              const gc = Math.floor(pick.spot[0] / pixel), gr = h - 1 - Math.floor(pick.spot[1] / pixel), g = gr * w + gc;
              if (gc < 0 || gr < 0 || gc >= w || gr >= h || labels[g]) return;
              cells.push(g);
            }
            const id = count + extra.length + 1;
            for (const g of cells) { labels[g] = id; mask[g] = 1; }
            extra.push({ pos: pos2, spot: pick.spot, size: cells.length, young: true });
            done.add(siteOf(pos2));
            fineSearch.added++;
            return true;
          };
          if (fineSearch.counted) {
            for (const f of found2) commit2(f.pos, f.pick);
            for (const b2 of runs2.sort((x, y) => x.q - y.q || x.p - y.p)) {
              const p2 = [b2.p, b2.p + 1].find((pp) => !done.has(pp + "," + b2.q));
              if (p2 !== undefined) commit2([p2, b2.q], b2.pick);
            }
          }
          if (fineSearch.bare_soil.counted) {
            for (const f of found3) if (commit2(f.pos, f.pick)) fineSearch.bare_soil.added++;
            for (const b3 of runs3.sort((x, y) => x.q - y.q || x.p - y.p)) {
              const p3 = [b3.p, b3.p + 1].find((pp) => !done.has(pp + "," + b3.q));
              if (p3 !== undefined && commit2([p3, b3.q], b3.pick)) fineSearch.bare_soil.added++;
            }
          }
        }
      }

      // Row walk: along every row of trees (its own line, on to the end of
      // its tree strip) young trees are looked for between and beyond the
      // trees, also where the trees stand at another spacing than the grid
      // (a block planted differently; young trees replanted at their own
      // spacing). Only where the grid is the planting design, only young
      // crowns of at least 0.3 m², each at least about a tree spacing from
      // every tree of its row (so never a weed beside a tree), and only when
      // the same walk along the grass alleys finds almost nothing (below).
      if (gridSearch && (gridSearch.gridOk || opts.forceRowWalk)) {
        await step("Young trees: walking along every row …");
        const ah = [a[0] / inRow, a[1] / inRow];
        const Uf = (p2) => p2[0] * ah[0] + p2[1] * ah[1], Vf = (p2) => p2[0] * acr[0] + p2[1] * acr[1];
        const rows2 = new Map(), occ = new Set();
        const addPt = (q, pt, ps) => { if (!rows2.has(q)) rows2.set(q, []); rows2.get(q).push({ u: Uf(pt), v: Vf(pt), pos: ps, pt }); };
        for (let t = 0; t < count; t++) if (pos[t] && alive[t]) { addPt(pos[t][1], pts[t], pos[t]); occ.add(siteOf(pos[t])); }
        for (const e of extra) if (e.pos) { addPt(e.pos[1], e.spot, e.pos); occ.add(siteOf(e.pos)); }
        const minGap = 0.55 * inRow;
        rowWalk = { rows: 0, onRows: 0, inAlleys: 0, counted: false, added: 0 };
        const cands = [];
        // walk one line (the row, or the alley beside it: lineShift = 0 or
        // half a row spacing); young crowns found, at least minGap apart
        const walk = (list, lineShift, keep, finder = (x, y) => findSapling(x, y, L), minA = 0.45) => {
          const nearOwn = (u, k) => list.map((o) => [Math.abs(o.u - u), o]).sort((x, y) => x[0] - y[0]).slice(0, k).map((x) => x[1]);
          const vAt = (u) => percentile(nearOwn(u, 3).map((o) => o.v), 50) + lineShift;
          const picks = [];
          // three lanes along the line (a big tree's top may sit off the strip centre)
          for (let u = list[0].u - 3 * inRow; u <= list[list.length - 1].u + 3 * inRow; u += rIn) for (const lane of [0, -0.8, 0.8]) {
            const v = vAt(u) + lane * rIn;
            const r = finder(u * ah[0] + v * acr[0], u * ah[1] + v * acr[1]);
            if (!r || !r.pick || r.pick.list.length * area < minA || r.pick.list.length * area > 3.5) continue;  // young-crown sized
            const pk = r.pick, pu = Uf(pk.spot);
            if (Math.abs(Vf(pk.spot) - vAt(pu)) > 1.6 * rIn) continue;                    // on the line
            let clear = Infinity;
            for (const u2 of pk.list) if (toCrown[u2] < clear) clear = toCrown[u2];
            if (clear * pixel < 1.0) continue;                                             // standing clear of big crowns
            if (picks.some((o) => Math.hypot(o.spot[0] - pk.spot[0], o.spot[1] - pk.spot[1]) < 0.5)) continue;
            picks.push(pk);
          }
          picks.sort((x, y) => y.list.length - x.list.length);
          const taken = lineShift ? [] : list.map((o) => o.u), out = [];
          for (const pk of picks) {
            const pu = Uf(pk.spot);
            if (taken.some((tu) => Math.abs(tu - pu) < minGap)) continue;
            taken.push(pu);
            out.push(pk);
          }
          if (keep) for (const pk of out) keep.push({ pk, list, nearOwn });
          return out.length;
        };
        for (const list of rows2.values()) {
          if (list.length < 2) continue;
          rowWalk.rows++;
          list.sort((p1, p2) => p1.u - p2.u);
          rowWalk.onRows += walk(list, 0, cands);
          // the alley beside the row, where nothing was planted: the check
          rowWalk.inAlleys += walk(list, 0.5 * rowSp, null);
        }
        rowWalk.counted = rowWalk.onRows >= 3 && rowWalk.inAlleys <= 0.15 * rowWalk.onRows + 1;
        for (const { pk, nearOwn } of rowWalk.counted ? cands : []) {
          const pu = Uf(pk.spot);
          {
            if (pk.list.some((u2) => labels[u2])) continue;
            // its planting position when that is free and close, else off the grid
            let ps = null;
            const o = nearOwn(pu, 1)[0];
            if (o && o.pos) {
              const ex = pk.spot[0] - o.pt[0], ey = pk.spot[1] - o.pt[1];
              const uu = o.pos[0] + I[0][0] * ex + I[0][1] * ey, vv = o.pos[1] + I[1][0] * ex + I[1][1] * ey;
              const pp = Math.round(uu), qq = Math.round(vv);
              if (Math.abs(uu - pp) <= 0.35 && Math.abs(vv - qq) <= 0.3 && !occ.has(pp + "," + qq)) ps = [pp, qq];
            }
            if (ps) occ.add(siteOf(ps));
            const id = count + extra.length + 1;
            for (const u2 of pk.list) { labels[u2] = id; mask[u2] = 1; }
            extra.push({ pos: ps, spot: pk.spot, size: pk.list.length, young: true });
            rowWalk.added++;
            const lst = cands.find((c) => c.pk === pk).list;
            lst.push({ u: pu, v: Vf(pk.spot), pos: ps, pt: pk.spot });
          }
        }

        // The same walk for dark young crowns (young pecans in weeds that
        // stand a little off the exact spacing), with the same alley check.
        if (findDark) {
          for (const e of extra) if (e.young && e.pos) {
            const lst = rows2.get(e.pos[1]);
            if (lst && !lst.some((o) => Math.abs(o.u - Uf(e.spot)) < 0.1)) lst.push({ u: Uf(e.spot), v: Vf(e.spot), pos: e.pos, pt: e.spot });
          }
          const cands3 = [];
          darkWalk = { onRows: 0, inAlleys: 0, counted: false, added: 0 };
          for (const list of rows2.values()) {
            if (list.length < 2) continue;
            list.sort((p1, p2) => p1.u - p2.u);
            darkWalk.onRows += walk(list, 0, cands3, findDark, 0.25);
            darkWalk.inAlleys += walk(list, 0.5 * rowSp, null, findDark, 0.25);
          }
          darkWalk.counted = darkWalk.onRows >= 3 && darkWalk.inAlleys <= 0.15 * darkWalk.onRows + 1;
          for (const { pk, nearOwn } of darkWalk.counted ? cands3 : []) {
            if (pk.list.some((u2) => labels[u2])) continue;
            const pu = Uf(pk.spot);
            let ps = null;
            const o = nearOwn(pu, 1)[0];
            if (o && o.pos) {
              const ex = pk.spot[0] - o.pt[0], ey = pk.spot[1] - o.pt[1];
              const uu = o.pos[0] + I[0][0] * ex + I[0][1] * ey, vv = o.pos[1] + I[1][0] * ex + I[1][1] * ey;
              const pp = Math.round(uu), qq = Math.round(vv);
              if (Math.abs(uu - pp) <= 0.35 && Math.abs(vv - qq) <= 0.3 && !occ.has(pp + "," + qq)) ps = [pp, qq];
            }
            if (ps) occ.add(siteOf(ps));
            const id = count + extra.length + 1;
            for (const u2 of pk.list) { labels[u2] = id; mask[u2] = 1; }
            extra.push({ pos: ps, spot: pk.spot, size: pk.list.length, young: true });
            darkWalk.added++;
          }
        }

        // Young crowns in half shade or with darker leaves (between the
        // sunlit and the dark rules: 55 to 70 % as bright as the ground
        // around them, a little less green than the ground's best): one
        // more walk along the rows, with its own check along the alleys.
        if (opts.midWalk !== false) {
          const Lm = { ...L, sunlit: 0.55, darkMin: 0.85, contrast: 0.08 };
          for (const e of extra) if (e.young && e.pos) {
            const lst = rows2.get(e.pos[1]);
            if (lst && !lst.some((o) => Math.abs(o.u - Uf(e.spot)) < 0.1)) lst.push({ u: Uf(e.spot), v: Vf(e.spot), pos: e.pos, pt: e.spot });
          }
          const candsM = [];
          midWalk = { onRows: 0, inAlleys: 0, counted: false, added: 0 };
          const finderM = (x, y) => findSapling(x, y, Lm);
          for (const list of rows2.values()) {
            if (list.length < 2) continue;
            list.sort((p1, p2) => p1.u - p2.u);
            midWalk.onRows += walk(list, 0, candsM, finderM);
            midWalk.inAlleys += walk(list, 0.5 * rowSp, null, finderM);
          }
          midWalk.counted = midWalk.onRows >= 3 && midWalk.inAlleys <= 0.15 * midWalk.onRows + 1;
          for (const { pk, nearOwn } of midWalk.counted ? candsM : []) {
            if (pk.list.some((u2) => labels[u2])) continue;
            const pu = Uf(pk.spot);
            let ps = null;
            const o = nearOwn(pu, 1)[0];
            if (o && o.pos) {
              const ex = pk.spot[0] - o.pt[0], ey = pk.spot[1] - o.pt[1];
              const uu = o.pos[0] + I[0][0] * ex + I[0][1] * ey, vv = o.pos[1] + I[1][0] * ex + I[1][1] * ey;
              const pp = Math.round(uu), qq = Math.round(vv);
              if (Math.abs(uu - pp) <= 0.35 && Math.abs(vv - qq) <= 0.3 && !occ.has(pp + "," + qq)) ps = [pp, qq];
            }
            if (ps) occ.add(siteOf(ps));
            const id = count + extra.length + 1;
            for (const u2 of pk.list) { labels[u2] = id; mask[u2] = 1; }
            extra.push({ pos: ps, spot: pk.spot, size: pk.list.length, young: true });
            midWalk.added++;
          }
        }
      }
    }
    const total = count + extra.length;
    extra.forEach((e, k) => {
      pts.push(e.spot); knots.push(e.spot); scores.push(0); pos.push(e.pos); eligible.push(true);
      pieces[count + k] = 1;
    });
    const aliveAll = new Uint8Array(total);
    for (let t = 0; t < count; t++) aliveAll[t] = alive[t];
    for (let t = count; t < total; t++) aliveAll[t] = 1;
    const inferred = new Uint8Array(total), young = new Uint8Array(total);
    for (let t = count; t < total; t++) { if (extra[t - count].young) young[t] = 1; else inferred[t] = 1; }

    // Crown boxes and areas (after merges and splits).
    const area2 = new Float64Array(total + 1), box = Array.from({ length: total + 1 }, () => [Infinity, Infinity, -Infinity, -Infinity]);
    const sumCX = new Float64Array(total + 1), sumCY = new Float64Array(total + 1);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const l = labels[y * w + x];
      if (!l) continue;
      area2[l]++; sumCX[l] += x; sumCY[l] += y;
      const bb = box[l];
      if (x < bb[0]) bb[0] = x; if (y < bb[1]) bb[1] = y; if (x > bb[2]) bb[2] = x; if (y > bb[3]) bb[3] = y;
    }
    // Crowns below 2 m² are not shown, so they take no part in the colours
    // (young trees on their planting position excepted).
    for (let t = 0; t < total; t++) if (area2[t + 1] * area < 2.0 && !young[t]) aliveAll[t] = 0;
    // The middle of each crown (centre of its outline): the diagonal lines
    // run through it. Where the centre falls outside the crown (a bent
    // shape), the treetop is used.
    const mids = knots.map((k, t) => {
      const l = t + 1;
      if (!area2[l]) return k;
      const cx = sumCX[l] / area2[l], cy = sumCY[l] / area2[l];
      const ci = Math.round(cx), ri = Math.round(cy);
      if (ci < 0 || ri < 0 || ci >= w || ri >= h || labels[ri * w + ci] !== l) return k;
      return [(cx + 0.5) * pixel, (h - cy - 0.5) * pixel];
    });

    // ---------------- diagonals
    await step("Drawing the EVEN / ODD diagonals …");
    const angle = (v) => ((Math.atan2(-v[1], v[0]) * 180) / Math.PI + 360) % 180;
    const gapTo = (deg) => { const g = Math.abs(deg - hint) % 180; return Math.min(g, 180 - g); };
    // The diagonal joins a tree to the next tree in the next row (true
    // diagonal, at about 63° to rows 40 ft apart with trees 20 ft apart),
    // or to the tree two places along in the next row ("oblique", about
    // 45°: the same trees are then removed in every row). Automatic: the
    // oblique lines where the rows run east-west (within 20°: there the
    // true diagonal looks nearly vertical on the map), else the true one.
    const diagOptions = (m) => [[[m * a[0] + b[0], m * a[1] + b[1]], [m, 1]], [[m * a[0] - b[0], m * a[1] - b[1]], [m, -1]]].sort((p, q) => gapTo(angle(p[0])) - gapTo(angle(q[0])));
    let diagStyle = opts.diagonalStyle || "auto";
    if (diagStyle === "auto") { const ra = angle(a); diagStyle = Math.min(ra, 180 - ra) < 20 ? "oblique" : "true"; }
    const options = diagOptions(diagStyle === "oblique" ? 2 : 1);
    const [vector, stepDir] = options[0];
    const stepM = len(vector);
    const pitch = Math.abs(a[0] * b[1] - a[1] * b[0]) / stepM;
    const dir = [vector[0] / stepM, vector[1] / stepM];
    let normal = [-dir[1], dir[0]];
    if (normal[0] < 0) normal = [-normal[0], -normal[1]];
    // Numbering from the planting grid (D01 / D02 stay at the same corner);
    // each crown's diagonal from its position after the row alternation.
    const baseIndex = pos.map((p) => (p ? p[0] * stepDir[1] - p[1] * stepDir[0] : null));
    const alt = opts.alternateRows === false
      ? { shifted: pos, moved: new Uint8Array(total), extraSteps: 0 }
      : alternateAlongRows(pos, aliveAll, stepDir, opts.coordinateMoves !== false);
    const index = alt.shifted.map((p) => (p ? p[0] * stepDir[1] - p[1] * stepDir[0] : null));
    const keys = [...new Set(baseIndex.filter((k, t) => k !== null && aliveAll[t]))].sort((p, q) => p - q);
    const offset = (t) => pts[t][0] * normal[0] + pts[t][1] * normal[1];
    // D01 = outermost diagonal on the side the normal points to.
    const meanOff = keys.map((k) => { let s = 0, c = 0; baseIndex.forEach((kk, t) => { if (kk === k && aliveAll[t]) { s += offset(t); c++; } }); return s / c; });
    let trend = 1;
    if (keys.length >= 2) {
      const mk = keys.reduce((s, k) => s + k, 0) / keys.length, mo = meanOff.reduce((s, o) => s + o, 0) / keys.length;
      trend = keys.reduce((s, k, i) => s + (k - mk) * (meanOff[i] - mo), 0);
    }
    const sign = trend > 0 ? 1 : -1;
    const first = sign > 0 ? keys[keys.length - 1] : keys[0];
    const numberOf = (k) => (first - k) * sign + 1;
    const diagonal = new Int32Array(total);
    for (let t = 0; t < total; t++) diagonal[t] = index[t] !== null ? numberOf(index[t]) : 0;

    // Grid frame: u along rows, v across rows.
    const rowHat = [a[0] / inRow, a[1] / inRow];
    let across = [-rowHat[1], rowHat[0]];
    if (across[0] * b[0] + across[1] * b[1] < 0) across = [-across[0], -across[1]];
    const rowSpacing = Math.abs(a[0] * b[1] - a[1] * b[0]) / inRow;
    const slope = (vector[0] * rowHat[0] + vector[1] * rowHat[1]) / (vector[0] * across[0] + vector[1] * across[1]);
    const U = (p) => p[0] * rowHat[0] + p[1] * rowHat[1];
    const V = (p) => p[0] * across[0] + p[1] * across[1];
    const toXY = (u, v) => [u * rowHat[0] + v * across[0], u * rowHat[1] + v * across[1]];
    const labelAt = (x, y) => {
      const c = Math.floor(x / pixel), r = h - 1 - Math.floor(y / pixel);
      return c < 0 || r < 0 || c >= w || r >= h ? -1 : labels[r * w + c];
    };
    const validAt = (x, y) => {
      const c = Math.floor(x / pixel), r = h - 1 - Math.floor(y / pixel);
      return c >= 0 && r >= 0 && c < w && r < h && valid[r * w + c] === 1;
    };
    const diagOfLabel = new Int32Array(total + 1);
    for (let t = 0; t < total; t++) if (aliveAll[t]) diagOfLabel[t + 1] = diagonal[t];
    // How deep (pixels) a point lies inside the canopy of each colour: a
    // line crossing the other colour keeps to the crown edges.
    const edgeTol = opts.edgeTolerance ?? 0.25;
    const depthIn = [0, 1].map((f) => {
      const m2 = new Uint8Array(n);
      for (let i = 0; i < n; i++) { const l = labels[i]; m2[i] = l > 0 && diagOfLabel[l] !== 0 && Math.abs(diagOfLabel[l]) % 2 === f ? 1 : 0; }
      return edt(m2, w, h);
    });

    const extent = (t) => {
      const u = U(mids[t]), v = V(mids[t]);
      let lo = v, hi = v;
      for (let s = pixel / 2; s < 2 * rowSpacing; s += pixel / 2) { const [x, y] = toXY(u, v - s); if (labelAt(x, y) !== t + 1) break; lo = v - s; }
      for (let s = pixel / 2; s < 2 * rowSpacing; s += pixel / 2) { const [x, y] = toXY(u, v + s); if (labelAt(x, y) !== t + 1) break; hi = v + s; }
      return [lo, hi];
    };
    const ease = (t) => { t = Math.max(0, Math.min(1, t)); return t ** 4 * (35 + t * (-84 + t * (70 - 20 * t))); };

    const crossingFor = (number) => (uGuess, v) => {
      const half = 0.35 * rowSpacing, reach = 0.7 * inRow;
      let best = null, bestU = uGuess;
      for (let u = uGuess - reach; u <= uGuess + reach + 1e-9; u += 0.15) {
        let other = 0;
        for (let s = -half; s <= half; s += pixel) {
          const [x, y] = toXY(u, v + s);
          const l = labelAt(x, y);
          if (l > 0 && diagOfLabel[l] !== number) other++;
        }
        // least canopy of the other colour, but each metre sideways counts
        // too (else neighbouring lines all crowd into the same gap)
        const cost = other * pixel + (opts.crossPenalty ?? 0.5) * Math.abs(u - uGuess);
        if (!best || cost < best[0] - 1e-9) { best = [cost]; bestU = u; }
      }
      return bestU;
    };

    function wavy(number, members, ext, crossing, separate) {
      const groups = [];
      for (const t of members.slice().sort((p, q) => V(mids[p]) - V(mids[q]))) {
        const [lo, hi] = ext.get(t);
        const last = groups[groups.length - 1];
        if (last && !separate.has(t) && !last.members.some((m) => separate.has(m)) && lo <= last.hi && Math.abs(V(mids[t]) - last.v) < 0.3 * rowSpacing) {
          last.members.push(t);
          last.u = last.members.reduce((s, m) => s + U(mids[m]), 0) / last.members.length;
          last.v = last.members.reduce((s, m) => s + V(mids[m]), 0) / last.members.length;
          last.lo = Math.min(last.lo, lo); last.hi = Math.max(last.hi, hi);
        } else groups.push({ members: [t], u: U(mids[t]), v: V(mids[t]), lo, hi });
      }
      let ks = [groups[0]];
      for (const g of groups.slice(1)) {
        const prev = ks[ks.length - 1];
        const missing = Math.round((g.v - prev.v) / rowSpacing) - 1;
        for (let q = 1; q <= missing; q++) {
          const f = q / (missing + 1);
          const v = prev.v + f * (g.v - prev.v), u = prev.u + f * (g.u - prev.u);
          ks.push({ u: crossing(u, v), v });
        }
        ks.push(g);
      }
      const gap = 0.45 * rowSpacing;
      const clusters = [[ks[0]]];
      for (const k of ks.slice(1)) {
        const c = clusters[clusters.length - 1];
        if (k.v - c[c.length - 1].v < gap) c.push(k); else clusters.push([k]);
      }
      ks = [];
      for (let c of clusters) {
        if (c.length > 1) {
          const mid = c.reduce((s, k) => s + k.v, 0) / c.length;
          c = c.slice().sort((p, q) => (p.u - q.u) * (slope >= 0 ? 1 : -1)).map((k, i, all) => {
            let v = mid - gap / 2 + (gap * i) / (all.length - 1);
            // a small crown (young tree): the line passes it inside the crown
            if (k.members && k.hi - k.lo < 2.5) v = Math.min(k.hi, Math.max(k.lo, v));
            return { ...k, v };
          });
        }
        ks.push(...c);
      }
      const begin = Math.min(Math.min(...groups.map((g) => g.lo)), ks[0].v) - 1.5;
      const end = Math.max(Math.max(...groups.map((g) => g.hi)), ks[ks.length - 1].v) + 1.5;
      const uk = ks.map((k) => k.u), vk = ks.map((k) => k.v);
      for (let q = 1; q < vk.length; q++) vk[q] = Math.max(vk[q], vk[q - 1] + 0.05);
      // A line goes through the middle of each of its crowns: each knot may
      // slide along the row only a little around the crown's centre (in a
      // row without a tree of this line: inside the open gap there), and the
      // knots are then drawn as straight along the diagonal as those bands
      // allow (iterated smoothing, kept in the band).
      if (opts.straightLines !== false && vk.length >= 2) {
        const otherFam = (l) => l > 0 && diagOfLabel[l] !== 0 && Math.abs(diagOfLabel[l]) % 2 !== Math.abs(number) % 2;
        const chord = (u0, v0, inside) => {
          let lo = u0, hi = u0;
          for (let s = pixel / 2; s <= inRow; s += pixel / 2) { const [x, y] = toXY(u0 - s, v0); if (!inside(labelAt(x, y))) break; lo = u0 - s; }
          for (let s = pixel / 2; s <= inRow; s += pixel / 2) { const [x, y] = toXY(u0 + s, v0); if (!inside(labelAt(x, y))) break; hi = u0 + s; }
          return [lo, hi];
        };
        const bandLo = [], bandHi = [];
        ks.forEach((k, q) => {
          let lo, hi;
          if (k.members) {
            const own = new Set(k.members.map((t) => t + 1));
            // through the middle of the crown: at most 0.6 m (or 15 % of the
            // crown's width) from its centre
            [lo, hi] = chord(uk[q], vk[q], (l) => own.has(l));
            const half = Math.min(opts.centreBand ?? 0.6, 0.15 * (hi - lo));
            lo = uk[q] - half; hi = uk[q] + half;
          } else {
            [lo, hi] = chord(uk[q], vk[q], (l) => !otherFam(l));
            const cut = 0.2 * (hi - lo);
            lo = Math.max(uk[q] - 0.5 * inRow, Math.min(uk[q], lo + cut)); hi = Math.min(uk[q] + 0.5 * inRow, Math.max(uk[q], hi - cut));
          }
          bandLo.push(lo - slope * vk[q]); bandHi.push(hi - slope * vk[q]);
        });
        const sig = (opts.straightSigma ?? 1.0) * rowSpacing, nk0 = vk.length;
        let r = vk.map((v, q) => uk[q] - slope * v);
        const nbr = vk.map((v) => { const list = []; for (let j = 0; j < nk0; j++) { const dv = vk[j] - v; if (Math.abs(dv) <= 3 * sig) list.push([j, Math.exp(-0.5 * (dv / sig) ** 2)]); } return list; });
        for (let it = 0; it < (opts.straightIters ?? 120); it++) {
          const nr = r.map((_, q) => { let s = 0, ws = 0; for (const [j, wj] of nbr[q]) { s += wj * r[j]; ws += wj; } return s / ws; });
          r = nr.map((x, q) => Math.min(bandHi[q], Math.max(bandLo[q], x)));
        }
        for (let q = 0; q < nk0; q++) uk[q] = r[q] + slope * vk[q];
      }
      // Past its first and last crown the line runs on in the direction of
      // the diagonal itself (after a smooth bend inside the crown), so it
      // ends, and its arrow points, along the diagonal.
      const tail = (s, length) => { const s0 = Math.min(1.5, 0.5 * length); return s <= s0 ? (s * s) / (2 * s0) : s - s0 / 2; };
      // ... and on to the edge of the field (not ending in the middle)
      const toEdgeAlong = (u0, v0, dir) => {
        let d = 0;
        for (let s = 0.5; s < 2000; s += 0.5) {
          const [x, y] = toXY(u0 + dir * slope * s, v0 + dir * s);
          if (!validAt(x, y)) break;
          d = s;
        }
        return d + 0.5;
      };
      const lenB = Math.max(vk[0] - begin, toEdgeAlong(uk[0], vk[0], -1));
      const lenE = Math.max(end - vk[vk.length - 1], toEdgeAlong(uk[uk.length - 1], vk[vk.length - 1], 1));
      const line = [];
      // Clean diagonal: a smooth curve through the line's trees (no S-bend
      // at every tree), straight along the diagonal where the trees are,
      // and straight on along the diagonal past its end trees.
      const nk = vk.length, sec = [], tan = new Array(nk);
      for (let k = 0; k < nk - 1; k++) sec.push((uk[k + 1] - uk[k]) / (vk[k + 1] - vk[k]));
      for (let k = 0; k < nk; k++) {
        if (k === 0 || k === nk - 1) { tan[k] = slope; continue; }
        const s0 = sec[k - 1], s1 = sec[k];
        if (s0 * s1 <= 0) { tan[k] = 0; continue; }
        tan[k] = (s0 + s1) / 2;
        const lim = 3 * Math.min(Math.abs(s0), Math.abs(s1));                 // no overshoot
        if (Math.abs(tan[k]) > lim) tan[k] = Math.sign(tan[k]) * lim;
      }
      const curve = (q) => (v) => {
        const v0 = vk[q], v1 = vk[q + 1], H = v1 - v0, t = Math.max(0, Math.min(1, (v - v0) / H));
        const t2 = t * t, t3 = t2 * t;
        return (2 * t3 - 3 * t2 + 1) * uk[q] + (t3 - 2 * t2 + t) * H * tan[q] + (-2 * t3 + 3 * t2) * uk[q + 1] + (t3 - t2) * H * tan[q + 1];
      };
      const nb = Math.max(3, Math.ceil(lenB / 0.25));
      for (let s = 0; s < nb; s++) { const d = lenB * (1 - s / nb); line.push([uk[0] - slope * d, vk[0] - d]); }
      line.push([uk[0], vk[0]]);
      for (let q = 0; q < nk - 1; q++) {
        const cq = curve(q);
        const seg = opts.avoidOther === false ? null : route(number, uk[q], vk[q], uk[q + 1], vk[q + 1], cq);
        if (seg) { line.push(...seg); continue; }
        const v0 = vk[q], v1 = vk[q + 1];
        const steps = Math.max(3, Math.ceil((v1 - v0) / 0.25));
        for (let s = 1; s <= steps; s++) {
          const v = v0 + ((v1 - v0) * s) / steps;
          line.push([cq(v), v]);
        }
      }
      const ne = Math.max(3, Math.ceil(lenE / 0.25)), uL = uk[uk.length - 1], vL = vk[vk.length - 1];
      for (let s = 1; s <= ne; s++) { const d = (lenE * s) / ne; line.push([uL + slope * d, vL + d]); }
      void tail;
      // where the line must pass through its own crowns (for the last step)
      line.knots = ks.map((k, q) => (k.members ? [uk[q], vk[q], k.hi - k.lo < 2.5] : null)).filter(Boolean);
      return line;
    }

    // From one knot of a line to the next: the path that runs least over
    // canopy of the other colour (dynamic programming over the steps across
    // the rows, sideways moves cost a little, and it stays near the usual
    // smooth wave where nothing is in the way). null: too short to route.
    function route(number, u0, v0, u1, v1, curveFn) {
      const dv = 0.25, du = 0.25;
      const n = Math.round((v1 - v0) / dv);
      if (n < 4 || Math.abs(u1 - u0) > 2 * du * n) return null;
      const uLo = Math.min(u0, u1) - 0.75 * inRow, m = Math.ceil((Math.max(u0, u1) + 0.75 * inRow - uLo) / du) + 1;
      const fam2 = Math.abs(number) % 2, depthOther = depthIn[1 - fam2];
      const ideal = curveFn || ((v) => u0 + (u1 - u0) * ease((v - v0) / (v1 - v0)));
      // the clean curve itself when it keeps off the other colour
      let blocked = false;
      for (let i = 1; i < n && !blocked; i++) {
        const v = v0 + ((v1 - v0) * i) / n, [x, y] = toXY(ideal(v), v);
        const c = Math.floor(x / pixel), r = h - 1 - Math.floor(y / pixel);
        if (c >= 0 && r >= 0 && c < w && r < h && depthOther[r * w + c] * pixel > edgeTol) blocked = true;
      }
      if (!blocked) return null;
      const cost = new Float64Array((n + 1) * m).fill(Infinity), back = new Int16Array((n + 1) * m);
      const j0 = Math.round((u0 - uLo) / du), j1 = Math.round((u1 - uLo) / du);
      cost[j0] = 0;
      for (let i = 1; i <= n; i++) {
        const v = v0 + ((v1 - v0) * i) / n;
        const uEase = ideal(v);
        for (let j = 0; j < m; j++) {
          const u = uLo + j * du;
          const [x, y] = toXY(u, v);
          const c = Math.floor(x / pixel), r = h - 1 - Math.floor(y / pixel);
          const dIn = c < 0 || r < 0 || c >= w || r >= h ? 0 : depthOther[r * w + c] * pixel - edgeTol;  // metres beyond the tolerated edge
          const here = (dIn > 0 ? 10 + 20 * dIn : 0) + 0.3 * Math.abs(u - uEase);
          let best = Infinity, arg = 0;
          for (let k = Math.max(0, j - 2); k <= Math.min(m - 1, j + 2); k++) {
            const c = cost[(i - 1) * m + k] + 0.5 * ((j - k) * du) ** 2;
            if (c < best) { best = c; arg = k; }
          }
          cost[i * m + j] = best + here;
          back[i * m + j] = arg;
        }
      }
      if (!Number.isFinite(cost[n * m + j1])) return null;
      const us = new Float64Array(n + 1);
      for (let i = n, j = j1; i >= 0; i--) { us[i] = uLo + j * du; j = back[i * m + j]; }
      us[0] = u0; us[n] = u1;
      // light smoothing (ends fixed)
      const sm = Float64Array.from(us);
      for (let i = 1; i < n; i++) {
        let s = 0, c = 0;
        for (let k = Math.max(0, i - 2); k <= Math.min(n, i + 2); k++) { s += us[k]; c++; }
        sm[i] = s / c;
      }
      // a detour only when it really clears the other colour (in closed
      // canopy it would not, and would only make the line untidy)
      const overOther = (uf) => {
        let o = 0;
        for (let i = 1; i < n; i++) {
          const v = v0 + ((v1 - v0) * i) / n, [x, y] = toXY(uf(i, v), v);
          const c = Math.floor(x / pixel), r = h - 1 - Math.floor(y / pixel);
          if (c >= 0 && r >= 0 && c < w && r < h && depthOther[r * w + c] * pixel > edgeTol) o++;
        }
        return o * dv;
      };
      const oCurve = overOther((i, v) => ideal(v)), oPath = overOther((i) => sm[i]);
      if (oPath > 0.3 * oCurve || oCurve - oPath < 0.5) return null;
      const out = [];
      for (let i = 1; i <= n; i++) out.push([sm[i], v0 + ((v1 - v0) * i) / n]);
      return out;
    }

    // Crowns off the planting grid (young trees replanted at their own
    // spacing, trees planted a little off) join a diagonal BEFORE the lines
    // are drawn, so their line goes through them too: the diagonal at their
    // own place in the grid, of the colour opposite to the crowns next to
    // them along the row.
    const preAssigned = new Uint8Array(total);
    if (opts.offGridOnLines !== false) {
      const Ig = inv2(a, b);
      const placedT = [];
      for (let t = 0; t < total; t++) if (aliveAll[t] && pos[t] && alt.shifted[t]) placedT.push(t);
      const cellG = 2 * Math.max(inRow, rowSpacing), bucketsG = new Map();
      for (const s of placedT) {
        const k = Math.floor(pts[s][0] / cellG) + "," + Math.floor(pts[s][1] / cellG);
        if (!bucketsG.has(k)) bucketsG.set(k, []);
        bucketsG.get(k).push(s);
      }
      const parityOf = (d) => Math.abs(d) % 2;
      // the trees of each diagonal (across, along the rows), for where its line runs
      const diagTrees = new Map();
      for (const s of placedT) {
        if (!diagTrees.has(diagonal[s])) diagTrees.set(diagonal[s], []);
        diagTrees.get(diagonal[s]).push([V(mids[s]), U(mids[s])]);
      }
      for (let t = 0; t < total; t++) {
        if (!aliveAll[t] || pos[t]) continue;
        const bx = Math.floor(pts[t][0] / cellG), by = Math.floor(pts[t][1] / cellG);
        let near = -1, nd = Infinity;
        for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (const s of bucketsG.get((bx + dx) + "," + (by + dy)) || []) {
          const d = Math.hypot(pts[s][0] - pts[t][0], pts[s][1] - pts[t][1]);
          if (d < nd) { nd = d; near = s; }
        }
        if (near < 0) continue;
        const ex = pts[t][0] - pts[near][0], ey = pts[t][1] - pts[near][1];
        const pf = alt.shifted[near][0] + Ig[0][0] * ex + Ig[0][1] * ey, qf = alt.shifted[near][1] + Ig[1][0] * ex + Ig[1][1] * ey;
        const kf = pf * stepDir[1] - qf * stepDir[0];
        // colours of the crowns next to it along its row
        const ut = U(mids[t]), vt = V(mids[t]);
        let left = -1, right = -1, dl = Infinity, dr = Infinity;
        for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (const s of bucketsG.get((bx + dx) + "," + (by + dy)) || []) {
          const du = U(mids[s]) - ut, dv = V(mids[s]) - vt;
          if (Math.abs(dv) >= 0.4 * rowSpacing || du === 0 || Math.abs(du) > 1.5 * inRow) continue;
          if (du > 0 && du < dr) { dr = du; right = s; }
          if (du < 0 && -du < dl) { dl = -du; left = s; }
        }
        const sides = [left, right].filter((s) => s >= 0).map((s) => parityOf(diagonal[s]));
        const avoid = sides.length && sides.every((p) => p === sides[0]) ? sides[0] : -1;
        // a diagonal that already has a tree in this row would have two
        // there (its line could not go through both): the free one, usually
        // the empty planting position the young tree stands at
        // Which line will run nearest to it: each nearby diagonal's line
        // follows its own trees, so where it crosses this crown's row comes
        // from its trees in the rows just before and after (or in this row).
        // The nearest one takes it (the colour opposite to its row
        // neighbours preferred, by up to 1 m); its line then needs the
        // smallest bend and no other line has to give way.
        let bestK = Math.round(kf), bestC = Infinity;
        for (let k = Math.round(kf) - 3; k <= Math.round(kf) + 3; k++) {
          const list = diagTrees.get(numberOf(k));
          if (!list) continue;
          let below = null, above = null;
          for (const e of list) {
            if (e[0] <= vt && (!below || e[0] > below[0])) below = e;
            if (e[0] > vt && (!above || e[0] < above[0])) above = e;
          }
          let uAt = null;
          if (below && above && above[0] - below[0] < 3 * rowSpacing) uAt = below[1] + ((above[1] - below[1]) * (vt - below[0])) / (above[0] - below[0]);
          else { const e = below && vt - below[0] < 1.5 * rowSpacing ? below : above && above[0] - vt < 1.5 * rowSpacing ? above : null; if (e) uAt = e[1] + slope * (vt - e[0]); }
          if (uAt === null) continue;
          const c = Math.abs(uAt - ut) + (avoid >= 0 && parityOf(numberOf(k)) === avoid ? 1.0 : 0);
          if (c < bestC) { bestC = c; bestK = k; }
        }
        diagonal[t] = numberOf(bestK);
        diagOfLabel[t + 1] = diagonal[t];
        preAssigned[t] = 1;
      }
    }
    const byDiag = new Map();
    for (let t = 0; t < total; t++) if (aliveAll[t] && (pos[t] || preAssigned[t])) {
      if (!byDiag.has(diagonal[t])) byDiag.set(diagonal[t], []);
      byDiag.get(diagonal[t]).push(t);
    }
    const ext = new Map();
    for (const list of byDiag.values()) for (const t of list) ext.set(t, extent(t));

    // One unbroken line per diagonal, from its first to its last crown, also
    // across white areas (no image) inside the field.
    const uvLines = new Map();
    for (const [number, members] of [...byDiag.entries()].sort((p, q) => p[0] - q[0])) {
      const crossing = crossingFor(number);
      let separate = new Set();
      let uv = [];
      for (let attempt = 0; attempt < 2; attempt++) {
        uv = wavy(number, members, ext, crossing, separate);
        const hit = new Set();
        for (const [u, v] of uv) { const [x, y] = toXY(u, v); const l = labelAt(x, y); if (l > 0) hit.add(l); }
        separate = new Set(members.filter((t) => !hit.has(t + 1)));
        if (!separate.size) break;
      }
      uvLines.set(number, uv);
    }
    if (opts.debugLines) opts.debugLines("raw", [...uvLines].map(([nb, uv]) => [nb, uv.map(([u, v]) => toXY(u, v))]));   // testing only

    // Lines never cross: at every point across the rows the lines keep the
    // order of their numbers, at least 0.3 tree spacings apart. Where two
    // would cross, both give way the least possible (weighted isotonic
    // regression; a line running through its own crown gives way less).
    const STEP = 0.25, gapU = 0.2 * inRow;
    const samples = new Map();
    let kMin = Infinity, kMax = -Infinity;
    for (const [number, uv] of uvLines) {
      const k0 = Math.ceil(uv[0][1] / STEP), k1 = Math.floor(uv[uv.length - 1][1] / STEP);
      if (k1 < k0) continue;
      const u = new Float64Array(k1 - k0 + 1), wt = new Float64Array(k1 - k0 + 1);
      let j = 0;
      for (let k = k0; k <= k1; k++) {
        const v = k * STEP;
        while (j < uv.length - 2 && uv[j + 1][1] < v) j++;
        const [ua, va] = uv[j], [ub, vb] = uv[Math.min(j + 1, uv.length - 1)];
        const f = vb > va ? Math.min(1, Math.max(0, (v - va) / (vb - va))) : 0;
        u[k - k0] = ua + f * (ub - ua);
        const [x, y] = toXY(u[k - k0], v);
        const l = labelAt(x, y);
        wt[k - k0] = l > 0 && diagOfLabel[l] === number ? 10 : 1;
      }
      samples.set(number, { k0, k1, u, wt });
      kMin = Math.min(kMin, k0); kMax = Math.max(kMax, k1);
    }
    const order = [...samples.keys()].sort((p, q) => p - q);
    let trendSum = 0;
    for (let k = kMin; k <= kMax; k += 4) {
      let prev = null;
      for (const number of order) {
        const s = samples.get(number);
        if (k < s.k0 || k > s.k1) continue;
        const u = s.u[k - s.k0];
        if (prev) trendSum += Math.sign(u - prev[1]) / (number - prev[0]);
        prev = [number, u];
      }
    }
    const sgn = trendSum < 0 ? -1 : 1;
    // one pass: at every point across the rows, the order is restored with
    // the least (weighted) movement (pool adjacent violators on
    // y = sgn·u − gap·number, which must not decrease)
    const untangle = (useWeights) => {
      for (let k = kMin; k <= kMax; k++) {
        const at = [];
        for (const number of order) { const s = samples.get(number); if (k >= s.k0 && k <= s.k1) at.push([number, s]); }
        if (at.length < 2) continue;
        const blocks = [];
        for (const [number, s] of at) {
          const i = k - s.k0, wi = useWeights ? s.w[i] : 1;
          blocks.push({ sum: wi * (sgn * s.u[i] - gapU * number), w: wi, n: 1 });
          while (blocks.length > 1 && blocks[blocks.length - 2].sum / blocks[blocks.length - 2].w > blocks[blocks.length - 1].sum / blocks[blocks.length - 1].w) {
            const top = blocks.pop(), below = blocks[blocks.length - 1];
            below.sum += top.sum; below.w += top.w; below.n += top.n;
          }
        }
        let idx = 0;
        for (const bl of blocks) {
          const y = bl.sum / bl.w;
          for (let q = 0; q < bl.n; q++, idx++) { const [number, s] = at[idx]; s.u[k - s.k0] = sgn * (y + gapU * number); }
        }
      }
    };
    // moving average over ±1 m along the line
    const smooth = (arr) => {
      const out = new Float64Array(arr.length), r = 4;
      for (let i = 0; i < arr.length; i++) {
        let sum = 0, c = 0;
        for (let j = Math.max(0, i - r); j <= Math.min(arr.length - 1, i + r); j++) { sum += arr[j]; c++; }
        out[i] = sum / c;
      }
      return out;
    };
    for (const s of samples.values()) { s.w = smooth(s.wt); s.u0 = Float64Array.from(s.u); }
    untangle(true);
    // the shifts are smoothed along each line (gentle bends, no notches),
    // then a last pass guarantees the order again
    for (const s of samples.values()) {
      const delta = smooth(s.u.map((u, i) => u - s.u0[i]));
      for (let i = 0; i < s.u.length; i++) s.u[i] = s.u0[i] + delta[i];
    }
    untangle(false);
    // A line keeps off the canopy of the other colour (a red line never
    // over a blue crown, a blue line never over a red one): where it is on
    // such a crown it moves sideways (along the row) to the nearest point
    // that is not, then the bend is smoothed; the order of the lines (no
    // crossing) is restored last and always wins.
    // (a line may just touch the edge of such a crown: only where it would
    // run more than edgeTol inside it does it give way, so the lines stay
    // smooth instead of dodging every ragged crown edge)
    const otherAt = (number, u, v) => {
      const [x, y] = toXY(u, v);
      const c = Math.floor(x / pixel), r = h - 1 - Math.floor(y / pixel);
      if (c < 0 || r < 0 || c >= w || r >= h) return false;
      return depthIn[1 - (Math.abs(number) % 2)][r * w + c] * pixel > edgeTol;
    };
    const push = (number, s, margin) => {
      let moved = 0;
      for (let i = 0; i < s.u.length; i++) {
        const v = (s.k0 + i) * STEP, u = s.u[i];
        if (!otherAt(number, u, v)) continue;
        for (let d = pixel / 2; d <= 0.6 * inRow; d += pixel / 2) {
          const free1 = !otherAt(number, u - d, v), free2 = !otherAt(number, u + d, v);
          if (free1 || free2) { s.u[i] = free1 ? u - d - margin : u + d + margin; moved++; break; }
        }
      }
      return moved;
    };
    for (let round = 0; round < (opts.avoidOther === false ? 0 : 2); round++) {
      for (const [number, s] of samples) {
        const before = Float64Array.from(s.u);
        if (!push(number, s, pixel)) continue;
        const delta = smooth(s.u.map((u, i) => u - before[i]));
        for (let i = 0; i < s.u.length; i++) s.u[i] = before[i] + delta[i];
      }
    }
    untangle(false);
    // gentle smoothing of every line (no kinks), order kept
    for (let round = 0; round < 2; round++) {
      for (const s of samples.values()) s.u = smooth(s.u);
      untangle(false);
    }
    // Last word: off the canopy of the other colour. A point on such a crown
    // moves sideways to just past its edge (the nearer side), the move is
    // spread smoothly over 1.5 m on each side, and where lines must give way
    // to each other (no crossing), the moved points hold their place.
    for (const s of samples.values()) s.w = new Float64Array(s.u.length).fill(1);
    for (let round = 0; round < (opts.avoidOther === false ? 0 : 4); round++) {
      let anyMoved = false;
      for (const [number, s] of samples) {
        const shift = new Float64Array(s.u.length);
        for (let i = 0; i < s.u.length; i++) {
          const v = (s.k0 + i) * STEP, u = s.u[i];
          if (!otherAt(number, u, v)) continue;
          // (no free path within 2 m: closed canopy, the line stays)
          for (let d = pixel / 2; d <= Math.min(2.0, 0.35 * inRow); d += pixel / 2) {
            const freeL = !otherAt(number, u - d, v), freeR = !otherAt(number, u + d, v);
            if (!freeL && !freeR) continue;
            const sgn = freeL ? -1 : 1;
            const extra2 = !otherAt(number, u + sgn * (d + 0.3), v) ? 0.3 : 0;
            shift[i] = sgn * (d + extra2);
            break;
          }
        }
        let moved = false;
        const total2 = new Float64Array(s.u.length);
        for (let i = 0; i < s.u.length; i++) {
          if (!shift[i]) continue;
          moved = true;
          const spreadK = Math.round(Math.max(opts.bendMin ?? 3, (opts.bendFactor ?? 3) * Math.abs(shift[i])) / STEP);   // gradual bends (over at least 3 m)
          for (let j = Math.max(0, i - spreadK); j <= Math.min(s.u.length - 1, i + spreadK); j++) {
            const f = shift[i] * (1 - Math.abs(j - i) / (spreadK + 1));
            if (Math.abs(f) > Math.abs(total2[j])) total2[j] = f;
          }
          s.w[i] = 20;
        }
        if (!moved) continue;
        // keep the move only where it clears most of the other colour
        const onOther = (arr) => { let o = 0; for (let i = 0; i < arr.length; i++) if (otherAt(number, arr[i], (s.k0 + i) * STEP)) o++; return o; };
        const tried = s.u.map((u, i) => u + total2[i]);
        if (onOther(tried) > 0.5 * onOther(s.u)) { for (let i = 0; i < s.u.length; i++) if (shift[i]) s.w[i] = 1; continue; }
        anyMoved = true;
        for (let i = 0; i < s.u.length; i++) s.u[i] = tried[i];
      }
      untangle(true);
      if (!anyMoved) break;
    }
    // smooth out small steps, but never onto the other colour
    for (let round = 0; round < 2; round++) {
      for (const [number, s] of samples) {
        const sm = smooth(s.u);
        for (let i = 0; i < s.u.length; i++) {
          const v = (s.k0 + i) * STEP;
          if (!otherAt(number, sm[i], v) || otherAt(number, s.u[i], v)) s.u[i] = sm[i];
        }
      }
      untangle(true);
    }
    // no sharp kinks (where crowns of both colours close the way)
    for (let round = 0; round < 3; round++) {
      for (const s of samples.values()) {
        const sm = smooth(s.u);
        for (let i = 1; i < s.u.length - 1; i++) {
          if (Math.abs(s.u[i - 1] - 2 * s.u[i] + s.u[i + 1]) > 0.15) for (let j = Math.max(0, i - 4); j <= Math.min(s.u.length - 1, i + 4); j++) s.u[j] = sm[j];
        }
      }
      untangle(false);
    }
    // presentable: a wide final smoothing (about ±2 m), where it does not
    // put the line on more canopy of the other colour; order kept
    const wide = (arr, r) => {
      const out = new Float64Array(arr.length);
      for (let i = 0; i < arr.length; i++) {
        let s2 = 0, c = 0;
        for (let j = Math.max(0, i - r); j <= Math.min(arr.length - 1, i + r); j++) { s2 += arr[j]; c++; }
        out[i] = s2 / c;
      }
      return out;
    };
    for (let round = 0; round < 3; round++) {
      for (const [number, s] of samples) {
        const sm = wide(wide(s.u, 8), 8);
        let before = 0, after = 0;
        for (let i = 0; i < s.u.length; i++) {
          const v = (s.k0 + i) * STEP;
          if (otherAt(number, s.u[i], v)) before++;
          if (otherAt(number, sm[i], v)) after++;
        }
        if (after <= 1.15 * before + 0.01 * s.u.length) for (let i = 0; i < s.u.length; i++) s.u[i] = sm[i];
      }
      untangle(false);
    }
    // Last of all: every line runs through the middle of each of its own
    // crowns. Where the steps above moved it off, it is pulled back (the
    // pull spread smoothly from tree to tree, fading out past the end trees),
    // and in keeping the lines apart those points hold their place.
    if (opts.pinCentres !== false) {
      for (let round = 0; round < 3; round++) {
        for (const [number, s] of samples) {
          const kn = (uvLines.get(number) || {}).knots || [];
          const at = [];
          // (a crown far from where the line runs, e.g. a young tree beside
          // its planting position, is not worth dragging the line and its
          // neighbours over: only pulls up to pinMax)
          // (a small young crown: up to 0.7 of a tree spacing, else it is
          // missed altogether)
          const pinMax = (opts.pinMax ?? 0.4) * inRow, pinSmall = 0.7 * inRow;
          for (const [ku, kv, small] of kn) {
            const i = Math.round(kv / STEP) - s.k0;
            if (i >= 0 && i < s.u.length && Math.abs(ku - s.u[i]) <= (small ? pinSmall : pinMax)) at.push([i, ku - s.u[i]]);
          }
          s.w = new Float64Array(s.u.length).fill(1);
          if (!at.length) continue;
          at.sort((p, q) => p[0] - q[0]);
          const corr = new Float64Array(s.u.length), fade = Math.round(3 / STEP);
          for (let i = 0; i < s.u.length; i++) {
            if (i <= at[0][0]) corr[i] = at[0][1] * Math.max(0, 1 - (at[0][0] - i) / fade);
            else if (i >= at[at.length - 1][0]) corr[i] = at[at.length - 1][1] * Math.max(0, 1 - (i - at[at.length - 1][0]) / fade);
          }
          // eased from tree to tree (flat at each tree: no corner there)
          const easeS = (x) => x * x * (3 - 2 * x);
          for (let k = 0; k + 1 < at.length; k++) {
            const [i0, d0] = at[k], [i1, d1] = at[k + 1];
            for (let i = i0; i <= i1; i++) corr[i] = i1 > i0 ? d0 + (d1 - d0) * easeS((i - i0) / (i1 - i0)) : d0;
          }
          for (let i = 0; i < s.u.length; i++) {
            if (i < at[0][0]) corr[i] = at[0][1] * easeS(Math.max(0, 1 - (at[0][0] - i) / fade));
            else if (i > at[at.length - 1][0]) corr[i] = at[at.length - 1][1] * easeS(Math.max(0, 1 - (i - at[at.length - 1][0]) / fade));
          }
          const sm = smooth(corr);
          for (let i = 0; i < s.u.length; i++) s.u[i] += sm[i];
          for (const [i] of at) for (let j = Math.max(0, i - 3); j <= Math.min(s.u.length - 1, i + 3); j++) s.w[j] = 30;
        }
        untangle(true);
      }
      // round off the corners left where lines gave way to each other (the
      // points at the crowns hold their place)
      for (let round = 0; round < 3; round++) {
        for (const s of samples.values()) {
          if (!s.w) continue;
          const sm = smooth(s.u);
          for (let i = 0; i < s.u.length; i++) if (s.w[i] < 30) s.u[i] = sm[i];
        }
        untangle(true);
      }
    }
    let linesMoved = 0;
    for (const s of samples.values()) for (let i = 0; i < s.u.length; i++) if (Math.abs(s.u[i] - s.u0[i]) > 0.05) linesMoved++;
    // Only the ends of a line are cut at the edge of the field.
    const lines = new Map();
    for (const [number, uv] of uvLines) {
      const s = samples.get(number);
      const pts2 = s ? Array.from(s.u, (u, i) => toXY(u, (s.k0 + i) * STEP)) : uv.map(([u, v]) => toXY(u, v));
      let i0 = 0, i1 = pts2.length - 1;
      while (i0 <= i1 && !validAt(pts2[i0][0], pts2[i0][1])) i0++;
      while (i1 >= i0 && !validAt(pts2[i1][0], pts2[i1][1])) i1--;
      // Past its first and last own crown, a line runs on to the field edge
      // (it never starts or ends in the middle of the field). Only with
      // opts.trimTails it stops before a crown of the other colour instead.
      const kind = (p) => {
        const l = labelAt(p[0], p[1]);
        if (!(l > 0) || diagOfLabel[l] === 0) return 0;
        return Math.abs(diagOfLabel[l]) % 2 === Math.abs(number) % 2 ? 1 : -1;
      };
      let f0 = i0;
      while (f0 <= i1 && kind(pts2[f0]) !== 1) f0++;
      let f1 = i1;
      while (f1 >= i0 && kind(pts2[f1]) !== 1) f1--;
      if (opts.trimTails && f0 <= f1) {
        for (let j = f0 - 1; j >= i0; j--) if (kind(pts2[j]) === -1) { i0 = Math.min(f0, j + 3); break; }
        for (let j = f1 + 1; j <= i1; j++) if (kind(pts2[j]) === -1) { i1 = Math.max(f1, j - 3); break; }
      }
      lines.set(number, i1 > i0 ? [pts2.slice(i0, i1 + 1)] : []);
    }

    // Off-grid crowns take the diagonal whose line runs through them.
    const hits = new Map();
    for (const [number, list] of lines) for (const piece of list) for (const [x, y] of piece) {
      const l = labelAt(x, y);
      if (l <= 0) continue;
      const m = hits.get(l) || new Map();
      m.set(number, (m.get(number) || 0) + 1);
      hits.set(l, m);
    }
    for (let t = 0; t < total; t++) {
      if (!aliveAll[t] || pos[t]) continue;
      if (preAssigned[t]) {
        // joined its line before it was drawn; only if that line was pushed
        // off it (to keep the lines apart) and another runs through it, it
        // takes that one
        const m = hits.get(t + 1);
        if (m && m.size && !m.has(diagonal[t])) { diagonal[t] = [...m.entries()].reduce((p, q) => (q[1] > p[1] ? q : p))[0]; preAssigned[t] = 2; }
        continue;
      }
      const m = hits.get(t + 1);
      if (m && m.size) { diagonal[t] = [...m.entries()].reduce((p, q) => (q[1] > p[1] ? q : p))[0]; continue; }
      let best = 0, bestD = Infinity;
      for (const [number, list] of lines) for (const piece of list) for (let s = 0; s < piece.length; s += 4) {
        const d = Math.hypot(piece[s][0] - pts[t][0], piece[s][1] - pts[t][1]);
        if (d < bestD) { bestD = d; best = number; }
      }
      diagonal[t] = best;
    }

    // Off-grid crowns take the colour opposite to the crowns next to them
    // along the row (nearest crown on each side within 1.5 tree spacings).
    const parity = (d) => Math.abs(d) % 2;
    let recoloured = 0;
    if (opts.alternateRows !== false) {
      const drawnLines = [...lines.entries()].filter(([, list]) => list.length);
      for (let t = 0; t < total; t++) {
        if (!aliveAll[t] || pos[t] || !diagonal[t] || preAssigned[t]) continue;
        const ut = U(knots[t]), vt = V(knots[t]);
        let left = -1, right = -1, dl = Infinity, dr = Infinity;
        for (let s = 0; s < total; s++) {
          if (s === t || !aliveAll[s] || !diagonal[s]) continue;
          const du = U(knots[s]) - ut, dv = V(knots[s]) - vt;
          if (Math.abs(dv) >= 0.4 * rowSpacing || du === 0 || Math.abs(du) > 1.5 * inRow) continue;
          if (du > 0 && du < dr) { dr = du; right = s; }
          if (du < 0 && -du < dl) { dl = -du; left = s; }
        }
        const sides = [left, right].filter((s) => s >= 0).map((s) => parity(diagonal[s]));
        if (!sides.length || sides.some((p) => p !== sides[0]) || parity(diagonal[t]) !== sides[0]) continue;
        let best = 0, bestD = Infinity;
        for (const [number, list] of drawnLines) {
          if (parity(number) === sides[0]) continue;
          for (const piece of list) for (let s = 0; s < piece.length; s += 2) {
            const d = Math.hypot(piece[s][0] - pts[t][0], piece[s][1] - pts[t][1]);
            if (d < bestD) { bestD = d; best = number; }
          }
        }
        if (best) { diagonal[t] = best; recoloured++; }
      }
    }

    // Check (the planting rule): neighbouring crowns along every row must
    // have opposite colours.
    let rowPairs = 0, rowSame = 0;
    const byRow = new Map();
    for (let t = 0; t < total; t++) if (aliveAll[t] && pos[t]) {
      if (!byRow.has(pos[t][1])) byRow.set(pos[t][1], []);
      byRow.get(pos[t][1]).push(t);
    }
    for (const list of byRow.values()) {
      list.sort((p, q) => pos[p][0] - pos[q][0]);
      for (let k = 1; k < list.length; k++) {
        rowPairs++;
        if (parity(diagonal[list[k]]) === parity(diagonal[list[k - 1]])) rowSame++;
      }
    }
    let crossedOwn = 0, onGrid = 0;
    for (let t = 0; t < total; t++) {
      if (!aliveAll[t] || !pos[t]) continue;
      onGrid++;
      if ((hits.get(t + 1) || new Map()).has(diagonal[t])) crossedOwn++;
    }

    // Tree height (DSM) of every crown.
    let heights = null;
    if (chm) {
      await step("Tree height of every crown …");
      const values = Array.from({ length: total + 1 }, () => []);
      for (let i = 0; i < n; i++) { const l = labels[i]; if (l && Number.isFinite(chm[i])) values[l].push(chm[i]); }
      heights = values.map((v) => (v.length ? {
        median: percentile(v, 50), p90: percentile(v, 90), top: percentile(v, 99),
        mean: v.reduce((s, x) => s + x, 0) / v.length,
      } : null));
    }

    // ---------------- polygons
    await step("Outlining the crowns …");
    const crowns = [];
    const minI = Math.min(...pos.filter(Boolean).map((p) => p[0])), minJ = Math.min(...pos.filter(Boolean).map((p) => p[1]));
    for (let t = 0; t < total; t++) {
      if (!aliveAll[t] || !(area2[t + 1] * area >= 2.0 || (young[t] && area2[t + 1] > 0))) continue;
      const polys = tracePolygons(labels, w, t + 1, box[t + 1]).map((poly) =>
        poly.map((ring) => {
          const s = simplifyRing(ring.concat([ring[0]]), 0.6);
          return s.map(([cx, cy]) => [cx * pixel, (h - cy) * pixel]);
        }));
      if (!polys.length) continue;
      const flags = [];
      if (!pos[t]) flags.push("OFF_GRID");
      if (inferred[t]) flags.push("INFERRED_FROM_SPACING");
      if (young[t]) flags.push("YOUNG_TREE");
      if (pieces[t] > 1) flags.push("MERGED_SPLIT_CROWN");
      if (alt.moved[t]) flags.push("MOVED_ACROSS_GAP");
      const hgt = heights && heights[t + 1];
      crowns.push({
        id: "C" + String(t + 1).padStart(5, "0"),
        polys,
        area: area2[t + 1] * area,
        diagonal: diagonal[t],
        top: knots[t],
        grid_row: pos[t] ? pos[t][1] - minJ + 1 : null,
        grid_tree: pos[t] ? pos[t][0] - minI + 1 : null,
        flags: flags.join(";"),
        height_median_m: hgt ? hgt.median : null,
        height_p90_m: hgt ? hgt.p90 : null,
        height_top_m: hgt ? hgt.top : null,
        height_mean_m: hgt ? hgt.mean : null,
      });
    }
    let fieldPixels = 0, canopyPixels = 0;
    for (let i = 0; i < n; i++) { fieldPixels += valid[i]; canopyPixels += mask[i]; }
    const bearing = (v) => ((Math.atan2(v[0], v[1]) * 180) / Math.PI + 360) % 180;
    return {
      crowns,
      diagonals: [...lines.entries()].map(([number, pieces3]) => ({ number, pieces: pieces3 })),
      settings: {
        in_row_spacing_m: inRow,
        grid_vectors_m: { in_row: a, across: b },
        row_spacing_m: rowSpacing,
        row_bearing_deg: bearing(a),
        diagonal_bearing_deg: bearing(vector),
        diagonal_image_angle_deg: angle(vector),
        diagonal_spacing_m: pitch,
        diagonal_count: lines.size,
        diagonal_style: diagStyle,
        alternate_along_rows: opts.alternateRows !== false,
        crowns_moved_across_gap: alt.moved.reduce((s, v, t) => s + (v && aliveAll[t] ? 1 : 0), 0),
        diagonal_extra_steps_from_moves: alt.extraSteps,
        off_grid_crowns_recoloured_to_alternate: recoloured,
        young_trees_added: crowns.filter((c) => /YOUNG_TREE/.test(c.flags)).length,
        // planting-position search: spots found on / halfway between positions
        young_grid_search: gridSearch && {
          grid_matches_design: gridSearch.gridOk, positions_searched: gridSearch.on, found_on_positions: gridSearch.onFound,
          between_searched: gridSearch.between, found_between: gridSearch.betweenFound, offset_row_trees: gridSearch.runs,
          counted: gridSearch.counted, young_trees_added: gridSearch.added,
        },
        young_row_walk: rowWalk,
        young_dark_search: darkSearch,
        young_dark_walk: darkWalk,
        young_half_shade_walk: midWalk,
        young_full_resolution: fineSearch && {
          positions_searched: fineSearch.on, found_on_positions: fineSearch.onFound,
          between_searched: fineSearch.between, found_between: fineSearch.betweenFound, offset_row_trees: fineSearch.runs,
          counted: fineSearch.counted, young_trees_added: fineSearch.added, bare_soil: fineSearch.bare_soil,
        },
        line_points_moved_to_avoid_crossing: linesMoved,
        row_neighbours: rowPairs,
        row_neighbours_same_colour: rowSame,
        crowns_crossed_by_their_diagonal: onGrid ? crossedOwn / onGrid : null,
      },
      segmentation: {
        rgb_canopy_m2: rgbCanopyM2,
        elevation,
        cast_shadow: { used: opts.removeShadow !== false, shadow_removed_m2: shadowRemoved },
      },
      field_area_m2: fieldPixels * area,
      canopy_area_m2: canopyPixels * area,
      grid: { width: w, height: h, pixel },
      valid,
      mask: opts.returnRasters ? mask : undefined,
      labels: opts.returnRasters ? labels : undefined,
      heights: opts.returnRasters && chm ? chm : undefined,
    };
  }

  window.PecanAnalysis = {
    analyze, groundPixelSize, gridSize, countField, createGrid, sampleSurface, DETAIL_PIXELS,
    VERSION: "2026-11-20",
  };
})();
