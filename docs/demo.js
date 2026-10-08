/* KITTI demo: animate cumulative missed-car failure for each acquisition rule. */
(function () {
  const BUDGET = 12;
  const N_INIT = 6;
  const LS = 0.55;
  const NOISE = 0.08;
  const METHODS = [
    { id: "ours", name: "Ours", color: "#e4572e" },
    { id: "proxy", name: "Proxy only", color: "#4c9be8" },
    { id: "mi", name: "MI only", color: "#e3b341" },
    { id: "random", name: "Random", color: "#b7aea4" },
  ];

  const canvas = document.getElementById("curves");
  const fill = document.getElementById("budget-fill");
  const budgetText = document.getElementById("budget-text");
  const readout = document.getElementById("readout");
  const inspect = document.getElementById("inspect");
  const proxyImg = document.getElementById("proxy-img");
  const realImg = document.getElementById("real-img");
  const proxyScore = document.getElementById("proxy-score");
  const realScore = document.getElementById("real-score");

  let frames = [];
  let traces = null;
  let seed = 7;
  let stepIndex = 0;
  let timer = null;
  let yMin = 0;
  let yMax = 1;

  function mulberry32(s) {
    let a = s >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
  function xy(i) { return [frames[i].x, frames[i].y]; }

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

  function pick(method, labeled, rnd) {
    const taken = new Set(labeled);
    const rest = frames.map((_, i) => i).filter((i) => !taken.has(i));
    const dmin = rest.map((i) => {
      let m = Infinity;
      for (const j of labeled) m = Math.min(m, dist(xy(i), xy(j)));
      return m;
    });
    if (method === "random") return rest[Math.floor(rnd() * rest.length)];
    if (method === "mi") {
      let best = 0;
      for (let i = 1; i < rest.length; i++) if (dmin[i] > dmin[best]) best = i;
      return rest[best];
    }
    const eligible = [];
    for (let i = 0; i < rest.length; i++) if (dmin[i] > 0.18) eligible.push(i);
    const use = eligible.length ? eligible : rest.map((_, i) => i);
    if (method === "proxy") {
      let best = use[0];
      for (const i of use) if (frames[rest[i]].proxy > frames[rest[best]].proxy) best = i;
      return rest[best];
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
    const trainX = labeled.map(xy);
    const trainY = labeled.map((i) => frames[i].target);
    const fhat = gpMean(trainX, trainY, short.map(xy));
    const gs = labeled.map((i) => frames[i].proxy);
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
    return short[best];
  }

  function acquire(method, init, rnd) {
    const labeled = init.slice();
    for (let t = 0; t < BUDGET; t++) labeled.push(pick(method, labeled, rnd));
    return labeled;
  }

  function cum(indices, key, n) {
    let sum = 0;
    const out = [];
    for (let i = 0; i < n; i++) {
      sum += frames[indices[i]][key];
      out.push(sum / (i + 1));
    }
    return out;
  }

  function resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(640, Math.floor(rect.width * dpr));
    canvas.height = Math.floor(canvas.width * 8.5 / 16);
  }

  function draw() {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const shown = N_INIT + stepIndex;
    const total = N_INIT + BUDGET;
    const left = Math.round(w * 0.07);
    const right = Math.round(w * 0.03);
    const top = Math.round(h * 0.16);
    const bottom = Math.round(h * 0.16);
    const gap = Math.round(w * 0.06);
    const panelW = (w - left - right - gap) / 2;
    const panelH = h - top - bottom;
    const panels = [
      { title: "Real KITTI", key: "target", x0: left },
      { title: "Virtual KITTI proxy", key: "proxy", x0: left + panelW + gap },
    ];

    function xAt(panel, i) {
      return panel.x0 + (total <= 1 ? 0 : i / (total - 1)) * panelW;
    }
    function yAt(v) {
      return top + (1 - (v - yMin) / (yMax - yMin || 1)) * panelH;
    }

    ctx.font = `${Math.round(h * 0.045)}px "Source Sans 3", sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = Math.max(2, w / 450);

    for (const panel of panels) {
      ctx.strokeStyle = "rgba(255,255,255,0.18)";
      ctx.lineWidth = 1;
      ctx.strokeRect(panel.x0, top, panelW, panelH);
      ctx.fillStyle = "#f6f1ea";
      ctx.fillText(panel.title, panel.x0 + panelW / 2, top * 0.45);
      ctx.fillStyle = "#c8bfb4";
      ctx.font = `${Math.round(h * 0.038)}px "Source Sans 3", sans-serif`;
      ctx.fillText("labels collected", panel.x0 + panelW / 2, h - bottom * 0.35);
      ctx.save();
      ctx.translate(panel.x0 - left * 0.55, top + panelH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("cumulative mean failure", 0, 0);
      ctx.restore();
      ctx.font = `${Math.round(h * 0.045)}px "Source Sans 3", sans-serif`;

      for (const method of METHODS) {
        const series = cum(traces[method.id], panel.key, shown);
        ctx.beginPath();
        ctx.strokeStyle = method.color;
        ctx.lineWidth = Math.max(2.2, w / 380);
        series.forEach((v, i) => {
          const x = xAt(panel, i);
          const y = yAt(v);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        const last = series[series.length - 1];
        ctx.beginPath();
        ctx.fillStyle = method.color;
        ctx.arc(xAt(panel, series.length - 1), yAt(last), Math.max(3.5, w / 220), 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function showOursFrame() {
    const index = traces.ours[N_INIT + stepIndex - 1];
    const frame = frames[index];
    proxyImg.src = frame.proxyImg;
    realImg.src = frame.real;
    proxyScore.textContent = `proxy failure ${frame.proxy.toFixed(2)} · sequence ${frame.seq}, frame ${frame.frame}`;
    realScore.textContent = `target failure ${frame.target.toFixed(2)} · real KITTI label`;
  }

  function render() {
    draw();
    const prefix = traces.ours.slice(0, N_INIT + stepIndex);
    const mean = prefix.reduce((s, i) => s + frames[i].target, 0) / prefix.length;
    const severe = prefix.slice(N_INIT).filter((i) => frames[i].target >= 0.5).length;
    fill.style.width = `${(100 * stepIndex) / BUDGET}%`;
    budgetText.textContent = `target budget  ${stepIndex} / ${BUDGET}`;
    document.getElementById("n-used").textContent = String(stepIndex);
    document.getElementById("n-mean").textContent = mean.toFixed(2);
    document.getElementById("n-severe").textContent = String(severe);
    showOursFrame();
    const oursNow = cum(traces.ours, "target", N_INIT + stepIndex).at(-1);
    const randNow = cum(traces.random, "target", N_INIT + stepIndex).at(-1);
    inspect.textContent = `Ours cumulative real failure ${oursNow.toFixed(2)}. Random ${randNow.toFixed(2)}. A higher curve has found worse detector failures.`;
  }

  function step() {
    if (stepIndex >= BUDGET) {
      pause();
      readout.textContent = "Budget spent. New seed reruns every method from a shared initial set.";
      return;
    }
    stepIndex += 1;
    const frame = frames[traces.ours[N_INIT + stepIndex - 1]];
    readout.textContent = `Ours labeled sequence ${frame.seq} frame ${frame.frame}: real failure ${frame.target.toFixed(2)}, proxy ${frame.proxy.toFixed(2)}.`;
    render();
  }

  function play() { if (!timer) timer = setInterval(step, 700); }
  function pause() { clearInterval(timer); timer = null; }

  function reset(nextSeed) {
    pause();
    seed = nextSeed;
    const rnd = mulberry32(seed);
    const init = [];
    while (init.length < N_INIT) {
      const i = Math.floor(rnd() * frames.length);
      if (!init.includes(i)) init.push(i);
    }
    traces = {};
    METHODS.forEach((method, k) => {
      traces[method.id] = acquire(method.id, init, mulberry32(seed + 11 * (k + 1)));
    });
    const vals = [];
    for (const method of METHODS) {
      vals.push(...cum(traces[method.id], "target", N_INIT + BUDGET));
      vals.push(...cum(traces[method.id], "proxy", N_INIT + BUDGET));
    }
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const pad = 0.12 * (hi - lo + 1e-3);
    yMin = lo - pad;
    yMax = hi + pad;
    stepIndex = 0;
    readout.textContent = "All four methods share the first labels, then each spends the same real-frame budget.";
    render();
  }

  document.getElementById("play").onclick = play;
  document.getElementById("pause").onclick = pause;
  document.getElementById("step").onclick = () => { pause(); step(); };
  document.getElementById("reset").onclick = () => reset((seed + 1) % 997);
  window.addEventListener("resize", () => { resize(); if (traces) render(); });

  fetch("assets/kitti_pool.json")
    .then((r) => r.json())
    .then((data) => {
      frames = data.frames;
      resize();
      reset(7);
      play();
    })
    .catch(() => {
      readout.textContent = "Could not load the KITTI frame set.";
    });

  const video = document.getElementById("paper-video");
  const missing = document.getElementById("video-missing");
  if (video && missing) video.addEventListener("error", () => { missing.hidden = false; });
})();
