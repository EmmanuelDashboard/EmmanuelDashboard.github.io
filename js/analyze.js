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
 *      the same search finds far fewer spots halfway between positions
 *      (planted saplings, not weeds);
 *   8. EVEN / ODD diagonals as wavy lines through the crowns, ending in the
 *      direction of the diagonal; along every row neighbouring crowns always
 *      alternate (1 blue, 2 red ...), also across a missing tree.
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

  function measureGrid(points, designFt) {
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
  function alternateAlongRows(pos, alive, stepDir) {
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
      rowIds.forEach((row, k) => {
        for (const [t, s] of choices[k][picked[k]].shift) if (s) { shifted[t][0] += s; moved[t] = 1; }
        if (k && rowIds[k] - rowIds[k - 1] === 1) extraSteps += between(choices[k - 1][picked[k - 1]], choices[k][picked[k]]);
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
    const { a, b } = measureGrid(gridPts, design);
    const gp = gridPositions(gridPts, a, b);
    const pos = Array.from({ length: count }, () => null);
    gridIdx.forEach((t, k) => { if (gp.placed[k]) pos[t] = gp.pos[k]; });

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
    let gridSearch = null;
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
      const rIn = 0.25 * inRow, rOut = rIn + 1.5, R = Math.ceil(rOut / pixel), bw = 2 * R + 1;
      const free = (k) => valid[k] && !labels[k] && !trimmedAway[k];
      const compOf = new Int32Array(bw * bw);
      // where a tree at grid position (p, q) stands (p may be fractional)
      const spotAt = (p, q) => {
        const est = [], p0 = Math.round(p);
        for (let dq = -1; dq <= 1; dq++) for (let dp = -3; dp <= 3; dp++) {
          const s = placed.get((p0 + dp) + "," + (q + dq));
          if (!s) continue;
          const fp = p - (p0 + dp), fq = -dq;
          est.push([Math.abs(fp) * inRow + Math.abs(fq) * rowSp, s[0] + fp * a[0] + fq * b[0], s[1] + fp * a[1] + fq * b[1]]);
        }
        if (est.length < 2) return null;
        est.sort((e1, e2) => e1[0] - e2[0]);
        const use = est.slice(0, 6);
        return [percentile(use.map((e) => e[1]), 50), percentile(use.map((e) => e[2]), 50)];
      };
      // the sapling searched within rIn of a spot (null: none, or not searchable)
      const findSapling = (sx, sy) => {
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
          if (percentile(bgY, 50) < thresholds.darkY) return null;        // in a big tree's shadow
          const bg50 = percentile(bg, 50);
          const thr = Math.max(percentile(bg, 75) + 0.04, 0.5 * leafGreen);
          const isLeaf = (k) => free(k) && exgAt(k) >= thr;
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
              for (const [vx, vy] of [[ux - 1, uy], [ux + 1, uy], [ux, uy - 1], [ux, uy + 1]]) {
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
            if (A < 0.25 || A > maxArea) continue;                         // sapling-sized
            const g = list.map(exgAt);
            const gmax = Math.max(...g);
            if (gmax - bg50 < 0.14) continue;                            // much greener than the ground
            if (percentile(g, 75) < 0.75 * leafGreen) continue;           // as green as pecan leaves
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
            if (Math.sqrt(reach2) * pixel > 0.45 * inRow) continue;       // compact, not a strip
            if (list.length / (Math.PI * Math.max(reach2, 1)) < 0.3) continue;
            if (touch2 > 0.15 * edge2) continue;                          // not the edge of a big crown
            if (leak > 0.15 * edge2) continue;                            // not part of a band of grass
            const ring = ringGreens(list, 0.6);
            if (!ring.length) continue;
            const own25 = percentile(g, 25);
            const asGreen = ring.filter((v) => v >= own25).length / ring.length;
            if (asGreen > 0.15) continue;                                 // an island of leaves
            const score = gmax - bg50;
            if (!pick || score > pick.score) pick = { list, score, spot: [(mx2 + 0.5) * pixel, (h - my2 - 0.5) * pixel] };
          }
          return { pick, near: !!pick && Math.hypot(pick.spot[0] - sx, pick.spot[1] - sy) <= rIn };
      };
      const found = [];
      const check = { on: 0, onFound: 0, between: 0, betweenFound: 0 };
      for (const [q, [lo, hi]] of rowRange) {
        for (let p = lo - 1; p <= hi + 1; p++) {
          if (occupied.has(p + "," + q)) continue;
          const s = spotAt(p, q);
          const r = s && findSapling(s[0], s[1]);
          if (!r) continue;
          check.on++;
          if (r.near) check.onFound++;
          if (r.pick) found.push({ pos: [p, q], pick: r.pick });
        }
        for (let p = lo; p < hi; p++) {
          const s = spotAt(p + 0.5, q);
          const r = s && findSapling(s[0], s[1]);
          if (!r) continue;
          check.between++;
          if (r.near) check.betweenFound++;
        }
      }
      const rateOn = check.onFound / Math.max(1, check.on), rateBetween = check.betweenFound / Math.max(1, check.between);
      gridSearch = { ...check, counted: check.between >= 20 && check.onFound >= 5 && rateBetween <= 0.25 * rateOn };
      if (gridSearch.counted) {
        for (const f of found) {
          if (f.pick.list.some((u) => labels[u])) continue;
          const id = count + extra.length + 1;
          for (const u of f.pick.list) { labels[u] = id; mask[u] = 1; }
          extra.push({ pos: f.pos, spot: f.pick.spot, size: f.pick.list.length, young: true });
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
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const l = labels[y * w + x];
      if (!l) continue;
      area2[l]++;
      const bb = box[l];
      if (x < bb[0]) bb[0] = x; if (y < bb[1]) bb[1] = y; if (x > bb[2]) bb[2] = x; if (y > bb[3]) bb[3] = y;
    }
    // Crowns below 2 m² are not shown, so they take no part in the colours
    // (young trees on their planting position excepted).
    for (let t = 0; t < total; t++) if (area2[t + 1] * area < 2.0 && !young[t]) aliveAll[t] = 0;

    // ---------------- diagonals
    await step("Drawing the EVEN / ODD diagonals …");
    const angle = (v) => ((Math.atan2(-v[1], v[0]) * 180) / Math.PI + 360) % 180;
    const gapTo = (deg) => { const g = Math.abs(deg - hint) % 180; return Math.min(g, 180 - g); };
    const options = [[[a[0] + b[0], a[1] + b[1]], [1, 1]], [[a[0] - b[0], a[1] - b[1]], [1, -1]]].sort((p, q) => gapTo(angle(p[0])) - gapTo(angle(q[0])));
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
      : alternateAlongRows(pos, aliveAll, stepDir);
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

    const extent = (t) => {
      const u = U(knots[t]), v = V(knots[t]);
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
        const key = [Math.round(other * pixel * 10) / 10, Math.abs(u - uGuess)];
        if (!best || key[0] < best[0] || (key[0] === best[0] && key[1] < best[1])) { best = key; bestU = u; }
      }
      return bestU;
    };

    function wavy(members, ext, crossing, separate) {
      const groups = [];
      for (const t of members.slice().sort((p, q) => V(knots[p]) - V(knots[q]))) {
        const [lo, hi] = ext.get(t);
        const last = groups[groups.length - 1];
        if (last && !separate.has(t) && !last.members.some((m) => separate.has(m)) && lo <= last.hi && Math.abs(V(knots[t]) - last.v) < 0.3 * rowSpacing) {
          last.members.push(t);
          last.u = last.members.reduce((s, m) => s + U(knots[m]), 0) / last.members.length;
          last.v = last.members.reduce((s, m) => s + V(knots[m]), 0) / last.members.length;
          last.lo = Math.min(last.lo, lo); last.hi = Math.max(last.hi, hi);
        } else groups.push({ members: [t], u: U(knots[t]), v: V(knots[t]), lo, hi });
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
          c = c.slice().sort((p, q) => (p.u - q.u) * (slope >= 0 ? 1 : -1)).map((k, i, all) => ({ ...k, v: mid - gap / 2 + (gap * i) / (all.length - 1) }));
        }
        ks.push(...c);
      }
      const begin = Math.min(Math.min(...groups.map((g) => g.lo)), ks[0].v) - 1.5;
      const end = Math.max(Math.max(...groups.map((g) => g.hi)), ks[ks.length - 1].v) + 1.5;
      const uk = ks.map((k) => k.u), vk = ks.map((k) => k.v);
      for (let q = 1; q < vk.length; q++) vk[q] = Math.max(vk[q], vk[q - 1] + 0.05);
      // Past its first and last crown the line runs on in the direction of
      // the diagonal itself (after a smooth bend inside the crown), so it
      // ends, and its arrow points, along the diagonal.
      const tail = (s, length) => { const s0 = Math.min(1.5, 0.5 * length); return s <= s0 ? (s * s) / (2 * s0) : s - s0 / 2; };
      const lenB = vk[0] - begin, lenE = end - vk[vk.length - 1];
      const line = [];
      const nb = Math.max(3, Math.ceil(lenB / 0.25));
      for (let s = 0; s < nb; s++) { const d = lenB * (1 - s / nb); line.push([uk[0] - slope * tail(d, lenB), vk[0] - d]); }
      line.push([uk[0], vk[0]]);
      for (let q = 0; q < vk.length - 1; q++) {
        const v0 = vk[q], v1 = vk[q + 1];
        const steps = Math.max(3, Math.ceil((v1 - v0) / 0.25));
        for (let s = 1; s <= steps; s++) {
          const v = v0 + ((v1 - v0) * s) / steps;
          line.push([uk[q] + (uk[q + 1] - uk[q]) * ease((v - v0) / (v1 - v0)), v]);
        }
      }
      const ne = Math.max(3, Math.ceil(lenE / 0.25)), uL = uk[uk.length - 1], vL = vk[vk.length - 1];
      for (let s = 1; s <= ne; s++) { const d = (lenE * s) / ne; line.push([uL + slope * tail(d, lenE), vL + d]); }
      return line;
    }

    const byDiag = new Map();
    for (let t = 0; t < total; t++) if (aliveAll[t] && pos[t]) {
      if (!byDiag.has(diagonal[t])) byDiag.set(diagonal[t], []);
      byDiag.get(diagonal[t]).push(t);
    }
    const ext = new Map();
    for (const list of byDiag.values()) for (const t of list) ext.set(t, extent(t));

    const lines = new Map();
    for (const [number, members] of [...byDiag.entries()].sort((p, q) => p[0] - q[0])) {
      const crossing = crossingFor(number);
      let separate = new Set();
      let pieces2 = [];
      for (let attempt = 0; attempt < 2; attempt++) {
        const uv = wavy(members, ext, crossing, separate);
        // inside the field, each piece from its first to its last own crown
        pieces2 = [];
        let run = [];
        for (const [u, v] of uv) {
          const xy = toXY(u, v);
          if (validAt(xy[0], xy[1])) run.push([u, v, xy]);
          else { if (run.length > 1) pieces2.push(run); run = []; }
        }
        if (run.length > 1) pieces2.push(run);
        pieces2 = pieces2.map((piece) => {
          const v0 = piece[0][1], v1 = piece[piece.length - 1][1];
          const own = members.filter((t) => V(knots[t]) >= v0 && V(knots[t]) <= v1);
          if (!own.length) return null;
          const lo = Math.min(...own.map((t) => ext.get(t)[0])) - 1.5, hi = Math.max(...own.map((t) => ext.get(t)[1])) + 1.5;
          const kept2 = piece.filter(([, v]) => v >= lo && v <= hi).map(([, , xy]) => xy);
          return kept2.length > 1 ? kept2 : null;
        }).filter(Boolean);
        const hit = new Set();
        for (const piece of pieces2) for (const [x, y] of piece) { const l = labelAt(x, y); if (l > 0) hit.add(l); }
        separate = new Set(members.filter((t) => !hit.has(t + 1)));
        if (!separate.size) break;
      }
      lines.set(number, pieces2);
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
        if (!aliveAll[t] || pos[t] || !diagonal[t]) continue;
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
        row_spacing_m: rowSpacing,
        row_bearing_deg: bearing(a),
        diagonal_bearing_deg: bearing(vector),
        diagonal_image_angle_deg: angle(vector),
        diagonal_spacing_m: pitch,
        diagonal_count: lines.size,
        alternate_along_rows: opts.alternateRows !== false,
        crowns_moved_across_gap: alt.moved.reduce((s, v, t) => s + (v && aliveAll[t] ? 1 : 0), 0),
        diagonal_extra_steps_from_moves: alt.extraSteps,
        off_grid_crowns_recoloured_to_alternate: recoloured,
        young_trees_added: crowns.filter((c) => /YOUNG_TREE/.test(c.flags)).length,
        // planting-position search: spots found on / halfway between positions
        young_grid_search: gridSearch && {
          positions_searched: gridSearch.on, found_on_positions: gridSearch.onFound,
          between_searched: gridSearch.between, found_between: gridSearch.betweenFound, counted: gridSearch.counted,
        },
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
