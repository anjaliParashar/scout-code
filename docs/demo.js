/* Interactive 2-D demo of support-aware MI followed by a local control variate. */
(function () {
  const C1 = [1.35, 1.35], R1 = 0.65;
  const C2 = [-1.35, 1.35], R2 = 0.55;
  const N = 72;
  const BUDGET = 16;
  const N_INIT = 6;
  const LS = 0.55;
  const NOISE = 0.12;

  const MAGMA = [
    [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
    [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191],
  ];

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function diamond(x, c, r) {
    return r - (Math.abs(x[0] - c[0]) + Math.abs(x[1] - c[1]));
  }
  function realMean(x) {
    const base = Math.max(diamond(x, C1, R1), diamond(x, C2, R2));
    return 1.15 * Math.tanh(2.4 * base) + 0.10 * Math.sin(1.1 * x[0]) - 0.08 * Math.cos(0.9 * x[1]);
  }
  function simMean(x) {
    const xs = [x[0] + 0.18, x[1] - 0.12];
    const base = Math.max(diamond(xs, C1, R1 * 1.05), 0.1 * diamond(xs, C2, R2 * 0.95));
    return 1.15 * Math.tanh(2.4 * base) + 0.10 * Math.sin(1.1 * x[0]) - 0.08 * Math.cos(0.9 * x[1]);
  }
  function dist(a, b) {
    const dx = a[0] - b[0], dy = a[1] - b[1];
    return Math.hypot(dx, dy);
  }

  function color(t) {
    const u = Math.min(1, Math.max(0, t)) * (MAGMA.length - 1);
    const i = Math.floor(u);
    const f = u - i;
    const a = MAGMA[i], b = MAGMA[Math.min(i + 1, MAGMA.length - 1)];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }

  function solve(A, b) {
    const n = b.length;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let k = 0; k < n; k++) {
      let piv = k;
      for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[piv][k])) piv = i;
      const tmp = M[k]; M[k] = M[piv]; M[piv] = tmp;
      const diag = M[k][k] || 1e-8;
      for (let j = k; j <= n; j++) M[k][j] /= diag;
      for (let i = 0; i < n; i++) {
        if (i === k) continue;
        const f = M[i][k];
        for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j];
      }
    }
    return M.map((row) => row[n]);
  }

  function gpMean(trainX, trainY, query) {
    const n = trainX.length;
    if (!n) return query.map(() => 0);
    const mu = trainY.reduce((s, v) => s + v, 0) / n;
    const y = trainY.map((v) => v - mu);
    const K = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const d = dist(trainX[i], trainX[j]);
        K[i][j] = Math.exp(-0.5 * d * d / (LS * LS));
      }
      K[i][i] += NOISE * NOISE + 1e-5;
    }
    const alpha = solve(K, y);
    return query.map((q) => {
      let s = mu;
      for (let i = 0; i < n; i++) {
        const d = dist(q, trainX[i]);
        s += Math.exp(-0.5 * d * d / (LS * LS)) * alpha[i];
      }
      return s;
    });
  }

  function makeWorld(seed) {
    const rnd = mulberry32(seed);
    const pool = [];
    for (let i = 0; i < 380; i++) pool.push([-2.6 + rnd() * 5.2, -2.6 + rnd() * 5.2]);
    const labeled = [];
    while (labeled.length < N_INIT) {
      const i = Math.floor(rnd() * pool.length);
      if (!labeled.includes(i)) labeled.push(i);
    }
    const grid = [];
    for (let iy = 0; iy < N; iy++) {
      for (let ix = 0; ix < N; ix++) {
        const x = -3 + (6 * ix) / (N - 1);
        const y = -3 + (6 * iy) / (N - 1);
        grid.push([x, y]);
      }
    }
    let vmin = Infinity, vmax = -Infinity;
    const simGrid = grid.map(simMean);
    const realGrid = grid.map(realMean);
    for (const v of simGrid.concat(realGrid)) {
      vmin = Math.min(vmin, v);
      vmax = Math.max(vmax, v);
    }
    return { seed, pool, labeled, y: labeled.map((i) => realMean(pool[i])), grid, simGrid, realGrid, vmin, vmax, shortlist: [], chosen: -1 };
  }

  function minDists(pool, labeled, rest) {
    return rest.map((i) => {
      let m = Infinity;
      for (const j of labeled) m = Math.min(m, dist(pool[i], pool[j]));
      return m;
    });
  }

  function choose(world, method) {
    const { pool, labeled, y } = world;
    const rest = [];
    const taken = new Set(labeled);
    for (let i = 0; i < pool.length; i++) if (!taken.has(i)) rest.push(i);
    const dmin = minDists(pool, labeled, rest);
    if (method === "random") return { shortlist: [], chosen: rest[Math.floor(Math.random() * rest.length)] };
    if (method === "mi") {
      const order = rest.map((idx, k) => ({ idx, d: dmin[k] })).sort((a, b) => b.d - a.d);
      return { shortlist: order.slice(0, 18).map((o) => o.idx), chosen: order[0].idx };
    }
    const eligible = [];
    for (let i = 0; i < rest.length; i++) if (dmin[i] > 0.28) eligible.push(i);
    const use = eligible.length ? eligible : rest.map((_, i) => i);
    if (method === "proxy") {
      let best = use[0];
      for (const i of use) if (simMean(pool[rest[i]]) > simMean(pool[rest[best]])) best = i;
      return { shortlist: [], chosen: rest[best] };
    }
    const band = [];
    for (const i of use) if (dmin[i] < 1.15) band.push(i);
    const base = band.length >= 15 ? band : use;
    const sims = base.map((i) => simMean(pool[rest[i]]));
    const novs = base.map((i) => dmin[i]);
    const sMin = Math.min(...sims), sMax = Math.max(...sims);
    const nMin = Math.min(...novs), nMax = Math.max(...novs);
    const ranked = base.map((i, k) => ({
      i,
      score: 0.55 * ((sims[k] - sMin) / (sMax - sMin + 1e-9)) + 0.45 * ((novs[k] - nMin) / (nMax - nMin + 1e-9)),
    })).sort((a, b) => b.score - a.score).slice(0, 28);
    const shortIdx = ranked.map((r) => rest[r.i]);
    const trainX = labeled.map((i) => pool[i]);
    const fhat = gpMean(trainX, y, shortIdx.map((i) => pool[i]));
    const g = shortIdx.map((i) => simMean(pool[i]));
    const gs = trainX.map(simMean);
    const gMean = gs.reduce((s, v) => s + v, 0) / gs.length;
    const yMean = y.reduce((s, v) => s + v, 0) / y.length;
    let cov = 0, varg = 0;
    for (let i = 0; i < gs.length; i++) {
      cov += (gs[i] - gMean) * (y[i] - yMean);
      varg += (gs[i] - gMean) ** 2;
    }
    const beta = Math.min(1, Math.max(0, cov / (varg + 1e-6)));
    let best = 0;
    for (let i = 1; i < shortIdx.length; i++) {
      const si = (1 - 0.35 * beta) * fhat[i] + 0.35 * beta * g[i];
      const sb = (1 - 0.35 * beta) * fhat[best] + 0.35 * beta * g[best];
      if (si > sb) best = i;
    }
    return { shortlist: shortIdx, chosen: shortIdx[best] };
  }

  function counts(world) {
    let fails = 0, left = 0, right = 0;
    for (let k = N_INIT; k < world.labeled.length; k++) {
      const x = world.pool[world.labeled[k]];
      const r = realMean(x);
      if (r > 0.2) fails += 1;
      if (diamond(x, C2, R2) >= 0) left += 1;
      if (diamond(x, C1, R1) >= 0) right += 1;
    }
    return { fails, left, right, k: world.labeled.length - N_INIT };
  }

  function outline(ctx, c, r, scale) {
    ctx.beginPath();
    [[r, 0], [0, r], [-r, 0], [0, -r], [r, 0]].forEach(([dx, dy], i) => {
      const px = scale.px(c[0] + dx), py = scale.py(c[1] + dy);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = "rgba(0,0,0,0.85)";
    ctx.lineWidth = 1.4;
    ctx.stroke();
  }

  function paint(canvas, values, world, points) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    const img = ctx.createImageData(w, h);
    const scale = {
      px: (x) => ((x + 3) / 6) * (w - 1),
      py: (y) => (1 - (y + 3) / 6) * (h - 1),
    };
    for (let py = 0; py < h; py++) {
      const gy = Math.min(N - 1, Math.floor((1 - py / (h - 1)) * (N - 1)));
      for (let px = 0; px < w; px++) {
        const gx = Math.min(N - 1, Math.floor((px / (w - 1)) * (N - 1)));
        const v = values ? values[gy * N + gx] : null;
        const rgb = v == null ? [236, 232, 227] : color((v - world.vmin) / (world.vmax - world.vmin));
        const o = (py * w + px) * 4;
        img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    outline(ctx, C1, R1, scale);
    outline(ctx, C2, R2, scale);
    function dot(p, radius, fill) {
      ctx.beginPath();
      ctx.arc(scale.px(p[0]), scale.py(p[1]), radius, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = "#111";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    if (points.shortlist) {
      for (const p of points.shortlist) dot(p, 3.2, "#f5d76e");
    }
    if (points.labeled) {
      for (const p of points.labeled) dot(p, 4.2, "white");
    }
    if (points.chosen) dot(points.chosen, 7, "#fff");
  }

  const simCanvas = document.getElementById("sim");
  const tgtCanvas = document.getElementById("target");
  const fill = document.getElementById("budget-fill");
  const budgetText = document.getElementById("budget-text");
  const readout = document.getElementById("readout");
  const inspect = document.getElementById("inspect");
  let world = makeWorld(7);
  let method = "ours";
  let timer = null;
  let simField = null;

  function resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    for (const c of [simCanvas, tgtCanvas]) {
      const rect = c.getBoundingClientRect();
      c.width = Math.max(280, Math.floor(rect.width * dpr));
      c.height = c.width;
    }
  }

  function field() {
    if (!world.labeled.length) return null;
    const trainX = world.labeled.map((i) => world.pool[i]);
    return gpMean(trainX, world.y, world.grid);
  }

  function render() {
    if (!simField) simField = world.simGrid;
    paint(simCanvas, simField, world, {});
    const pred = world.labeled.length ? field() : null;
    const labeledPts = world.labeled.map((i) => world.pool[i]);
    const shortPts = world.shortlist.map((i) => world.pool[i]);
    paint(tgtCanvas, pred, world, {
      labeled: labeledPts,
      shortlist: shortPts,
      chosen: world.chosen >= 0 ? world.pool[world.chosen] : null,
    });
    const c = counts(world);
    fill.style.width = `${(100 * c.k) / BUDGET}%`;
    budgetText.textContent = `target budget  ${c.k} / ${BUDGET}`;
    document.getElementById("n-fail").textContent = c.fails;
    document.getElementById("n-left").textContent = c.left;
    document.getElementById("n-right").textContent = c.right;
    document.getElementById("n-used").textContent = c.k;
  }

  function step() {
    if (world.labeled.length - N_INIT >= BUDGET) {
      pause();
      readout.textContent = "Budget spent. Reset to run again, or switch the acquisition rule.";
      return;
    }
    const pick = choose(world, method);
    world.shortlist = pick.shortlist;
    world.chosen = pick.chosen;
    world.labeled.push(pick.chosen);
    world.y.push(realMean(world.pool[pick.chosen]));
    const x = world.pool[pick.chosen];
    const where = diamond(x, C2, R2) >= 0 ? "left diamond" : diamond(x, C1, R1) >= 0 ? "right diamond" : "background";
    if (method === "ours") {
      readout.textContent = `Mutual information proposed ${pick.shortlist.length} uncovered designs. The control variate kept the one in the ${where}.`;
    } else if (method === "proxy") {
      readout.textContent = `Proxy-only search evaluated the brightest simulated design, in the ${where}.`;
    } else if (method === "mi") {
      readout.textContent = `Mutual information alone picked the least covered design, in the ${where}.`;
    } else {
      readout.textContent = `Uniform random label, in the ${where}.`;
    }
    render();
  }

  function play() {
    if (timer) return;
    timer = setInterval(step, 700);
  }
  function pause() {
    clearInterval(timer);
    timer = null;
  }
  function reset(seed) {
    pause();
    world = makeWorld(seed);
    simField = null;
    readout.textContent = "Press play. Each step spends one real label.";
    render();
  }

  document.getElementById("play").onclick = play;
  document.getElementById("pause").onclick = pause;
  document.getElementById("step").onclick = () => { pause(); step(); };
  document.getElementById("reset").onclick = () => reset((world.seed + 1) % 997);
  document.querySelectorAll("[data-method]").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll("[data-method]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      method = btn.dataset.method;
      reset(world.seed);
    };
  });

  tgtCanvas.addEventListener("mousemove", (ev) => {
    const rect = tgtCanvas.getBoundingClientRect();
    const x = -3 + ((ev.clientX - rect.left) / rect.width) * 6;
    const y = 3 - ((ev.clientY - rect.top) / rect.height) * 6;
    const p = [x, y];
    let best = 0, bd = Infinity;
    for (let i = 0; i < world.pool.length; i++) {
      const d = dist(p, world.pool[i]);
      if (d < bd) { bd = d; best = i; }
    }
    if (bd > 0.45) { inspect.textContent = "Hover the target field to compare the simulator and the learned score."; return; }
    const q = world.pool[best];
    const pred = world.labeled.length ? gpMean(world.labeled.map((i) => world.pool[i]), world.y, [q])[0] : 0;
    inspect.textContent = `Nearest design (${q[0].toFixed(2)}, ${q[1].toFixed(2)}):  simulator ${simMean(q).toFixed(2)}   learned target ${pred.toFixed(2)}   true target ${realMean(q).toFixed(2)}`;
  });

  window.addEventListener("resize", () => { resize(); render(); });
  resize();
  render();
  play();

  const video = document.getElementById("paper-video");
  const missing = document.getElementById("video-missing");
  video.addEventListener("error", () => { missing.hidden = false; });
})();
