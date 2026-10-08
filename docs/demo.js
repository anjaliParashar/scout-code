/* KITTI demo: learn real missed-car scores from a few labels, using Virtual KITTI as the proxy. */
(function () {
  const BUDGET = 12;
  const N_INIT = 6;
  const GRID = 52;
  const LS = 0.55;
  const NOISE = 0.08;
  const MAGMA = [
    [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
    [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191],
  ];

  const simCanvas = document.getElementById("sim");
  const tgtCanvas = document.getElementById("target");
  const fill = document.getElementById("budget-fill");
  const budgetText = document.getElementById("budget-text");
  const readout = document.getElementById("readout");
  const inspect = document.getElementById("inspect");
  const proxyImg = document.getElementById("proxy-img");
  const realImg = document.getElementById("real-img");
  const proxyScore = document.getElementById("proxy-score");
  const realScore = document.getElementById("real-score");

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function dist(a, b) {
    return Math.hypot(a[0] - b[0], a[1] - b[1]);
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
  function idw(points, values, query) {
    return query.map((q) => {
      let wsum = 0, vsum = 0;
      for (let i = 0; i < points.length; i++) {
        const d = dist(q, points[i]);
        if (d < 1e-3) return values[i];
        const w = 1 / (d * d);
        wsum += w;
        vsum += w * values[i];
      }
      return vsum / wsum;
    });
  }

  let frames = [];
  let bounds = { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  let grid = [];
  let proxyField = [];
  let world = null;
  let method = "ours";
  let timer = null;
  let focus = 0;

  function xy(frame) { return [frame.x, frame.y]; }

  function makeWorld(seed) {
    const rnd = mulberry32(seed);
    const labeled = [];
    while (labeled.length < N_INIT) {
      const i = Math.floor(rnd() * frames.length);
      if (!labeled.includes(i)) labeled.push(i);
    }
    return { seed, labeled, shortlist: [], chosen: -1 };
  }

  function choose() {
    const labeled = new Set(world.labeled);
    const rest = frames.map((_, i) => i).filter((i) => !labeled.has(i));
    const dmin = rest.map((i) => {
      let m = Infinity;
      for (const j of world.labeled) m = Math.min(m, dist(xy(frames[i]), xy(frames[j])));
      return m;
    });
    if (method === "random") {
      return { shortlist: [], chosen: rest[Math.floor(Math.random() * rest.length)] };
    }
    if (method === "mi") {
      const order = rest.map((idx, k) => ({ idx, d: dmin[k] })).sort((a, b) => b.d - a.d);
      return { shortlist: order.slice(0, 12).map((o) => o.idx), chosen: order[0].idx };
    }
    const eligible = [];
    for (let i = 0; i < rest.length; i++) if (dmin[i] > 0.18) eligible.push(i);
    const use = eligible.length ? eligible : rest.map((_, i) => i);
    if (method === "proxy") {
      let best = use[0];
      for (const i of use) if (frames[rest[i]].proxy > frames[rest[best]].proxy) best = i;
      return { shortlist: [], chosen: rest[best] };
    }
    const band = use.filter((i) => dmin[i] < 1.6);
    const base = band.length >= 10 ? band : use;
    const sims = base.map((i) => frames[rest[i]].proxy);
    const novs = base.map((i) => dmin[i]);
    const sMin = Math.min(...sims), sMax = Math.max(...sims);
    const nMin = Math.min(...novs), nMax = Math.max(...novs);
    const ranked = base.map((i, k) => ({
      i,
      score: 0.55 * ((sims[k] - sMin) / (sMax - sMin + 1e-9))
        + 0.45 * ((novs[k] - nMin) / (nMax - nMin + 1e-9)),
    })).sort((a, b) => b.score - a.score).slice(0, 16);
    const short = ranked.map((r) => rest[r.i]);
    const trainX = world.labeled.map((i) => xy(frames[i]));
    const trainY = world.labeled.map((i) => frames[i].target);
    const fhat = gpMean(trainX, trainY, short.map((i) => xy(frames[i])));
    const gs = trainX.map((_, i) => frames[world.labeled[i]].proxy);
    const gMean = gs.reduce((s, v) => s + v, 0) / gs.length;
    const yMean = trainY.reduce((s, v) => s + v, 0) / trainY.length;
    let cov = 0, varg = 0;
    for (let i = 0; i < gs.length; i++) {
      cov += (gs[i] - gMean) * (trainY[i] - yMean);
      varg += (gs[i] - gMean) ** 2;
    }
    const beta = Math.min(1, Math.max(0, cov / (varg + 1e-6)));
    let best = 0;
    for (let i = 1; i < short.length; i++) {
      const si = (1 - 0.35 * beta) * fhat[i] + 0.35 * beta * frames[short[i]].proxy;
      const sb = (1 - 0.35 * beta) * fhat[best] + 0.35 * beta * frames[short[best]].proxy;
      if (si > sb) best = i;
    }
    return { shortlist: short, chosen: short[best] };
  }

  function project(canvas) {
    const pad = 28;
    return {
      px: (x) => pad + ((x - bounds.minX) / (bounds.maxX - bounds.minX)) * (canvas.width - 2 * pad),
      py: (y) => canvas.height - pad - ((y - bounds.minY) / (bounds.maxY - bounds.minY)) * (canvas.height - 2 * pad),
    };
  }

  function paint(canvas, values, points) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    const img = ctx.createImageData(w, h);
    const scale = project(canvas);
    for (let py = 0; py < h; py++) {
      const gy = Math.min(GRID - 1, Math.floor((1 - py / (h - 1)) * (GRID - 1)));
      for (let px = 0; px < w; px++) {
        const gx = Math.min(GRID - 1, Math.floor((px / (w - 1)) * (GRID - 1)));
        const v = values ? values[gy * GRID + gx] : null;
        const rgb = v == null ? [236, 232, 227] : color(v);
        const o = (py * w + px) * 4;
        img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    function dot(frame, radius, fillStyle) {
      ctx.beginPath();
      ctx.arc(scale.px(frame.x), scale.py(frame.y), radius, 0, Math.PI * 2);
      ctx.fillStyle = fillStyle;
      ctx.fill();
      ctx.strokeStyle = "#111";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    for (const i of points.shortlist || []) dot(frames[i], 4, "#f5d76e");
    for (const i of points.labeled || []) dot(frames[i], 5, "white");
    if (points.chosen >= 0) dot(frames[points.chosen], 8, "#fff");
  }

  function learnedField() {
    if (!world.labeled.length) return null;
    const trainX = world.labeled.map((i) => xy(frames[i]));
    const trainY = world.labeled.map((i) => frames[i].target);
    return gpMean(trainX, trainY, grid);
  }

  function show(index) {
    focus = index;
    const frame = frames[index];
    proxyImg.src = frame.proxyImg;
    realImg.src = frame.real;
    proxyScore.textContent = `proxy failure ${frame.proxy.toFixed(2)} · sequence ${frame.seq}, frame ${frame.frame}, ${frame.cars} cars`;
    const known = world.labeled.includes(index);
    realScore.textContent = known
      ? `target failure ${frame.target.toFixed(2)} · real KITTI label`
      : "target failure not labeled yet";
  }

  function render() {
    paint(simCanvas, proxyField, { labeled: world.labeled, chosen: world.chosen });
    paint(tgtCanvas, learnedField(), {
      labeled: world.labeled,
      shortlist: world.shortlist,
      chosen: world.chosen,
    });
    const acquired = world.labeled.slice(N_INIT);
    const k = acquired.length;
    const mean = k ? acquired.reduce((s, i) => s + frames[i].target, 0) / k : 0;
    const severe = acquired.filter((i) => frames[i].target >= 0.5).length;
    fill.style.width = `${(100 * k) / BUDGET}%`;
    budgetText.textContent = `target budget  ${k} / ${BUDGET}`;
    document.getElementById("n-used").textContent = String(k);
    document.getElementById("n-mean").textContent = mean.toFixed(2);
    document.getElementById("n-severe").textContent = String(severe);
    if (world.chosen >= 0) show(world.chosen);
  }

  function step() {
    if (world.labeled.length - N_INIT >= BUDGET) {
      pause();
      readout.textContent = "Budget spent. Reset for another seed, or switch the acquisition rule.";
      return;
    }
    const pick = choose();
    world.shortlist = pick.shortlist;
    world.chosen = pick.chosen;
    world.labeled.push(pick.chosen);
    const frame = frames[pick.chosen];
    const gap = frame.target - frame.proxy;
    readout.textContent = method === "ours"
      ? `Shortlist of ${pick.shortlist.length}, then the control variate kept sequence ${frame.seq} frame ${frame.frame}. Real failure ${frame.target.toFixed(2)}, proxy ${frame.proxy.toFixed(2)} (gap ${gap >= 0 ? "+" : ""}${gap.toFixed(2)}).`
      : `Labeled sequence ${frame.seq} frame ${frame.frame}. Real failure ${frame.target.toFixed(2)}, proxy ${frame.proxy.toFixed(2)}.`;
    render();
  }

  function play() { if (!timer) timer = setInterval(step, 1100); }
  function pause() { clearInterval(timer); timer = null; }
  function reset(seed) {
    pause();
    world = makeWorld(seed);
    world.shortlist = [];
    world.chosen = -1;
    readout.textContent = "Press play. Each step spends one real KITTI label. The pictures are that frame.";
    render();
    show(world.labeled[world.labeled.length - 1]);
  }

  function nearest(canvas, ev) {
    const rect = canvas.getBoundingClientRect();
    const scale = project(canvas);
    const mx = ((ev.clientX - rect.left) / rect.width) * canvas.width;
    const my = ((ev.clientY - rect.top) / rect.height) * canvas.height;
    let best = 0, bd = Infinity;
    frames.forEach((frame, i) => {
      const d = Math.hypot(scale.px(frame.x) - mx, scale.py(frame.y) - my);
      if (d < bd) { bd = d; best = i; }
    });
    return bd < 28 * (canvas.width / rect.width) ? best : -1;
  }

  function bindHover(canvas) {
    canvas.addEventListener("mousemove", (ev) => {
      const i = nearest(canvas, ev);
      if (i < 0) return;
      show(i);
      const frame = frames[i];
      inspect.textContent = `Sequence ${frame.seq}, frame ${frame.frame}: proxy ${frame.proxy.toFixed(2)}, real KITTI ${frame.target.toFixed(2)}.`;
    });
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
  bindHover(simCanvas);
  bindHover(tgtCanvas);
  window.addEventListener("resize", () => { resize(); if (world) render(); });

  function resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    for (const c of [simCanvas, tgtCanvas]) {
      const rect = c.getBoundingClientRect();
      const side = Math.max(240, Math.floor(rect.width * dpr));
      c.width = side;
      c.height = side;
    }
  }

  fetch("assets/kitti_pool.json")
    .then((r) => r.json())
    .then((data) => {
      frames = data.frames;
      const xs = frames.map((f) => f.x), ys = frames.map((f) => f.y);
      const padX = (Math.max(...xs) - Math.min(...xs)) * 0.12;
      const padY = (Math.max(...ys) - Math.min(...ys)) * 0.12;
      bounds = {
        minX: Math.min(...xs) - padX, maxX: Math.max(...xs) + padX,
        minY: Math.min(...ys) - padY, maxY: Math.max(...ys) + padY,
      };
      grid = [];
      for (let iy = 0; iy < GRID; iy++) {
        for (let ix = 0; ix < GRID; ix++) {
          grid.push([
            bounds.minX + (bounds.maxX - bounds.minX) * ix / (GRID - 1),
            bounds.minY + (bounds.maxY - bounds.minY) * iy / (GRID - 1),
          ]);
        }
      }
      proxyField = idw(frames.map(xy), frames.map((f) => f.proxy), grid);
      resize();
      reset(7);
      play();
    })
    .catch(() => {
      readout.textContent = "Could not load the KITTI frame set.";
    });
})();
