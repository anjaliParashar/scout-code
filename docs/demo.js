/* KITTI demo: severe-failure counts on real labels, and coverage on a shared t-SNE. */
(function () {
  const METHODS = [
    { id: "ours", name: "Ours", color: "#e4572e" },
    { id: "proxy", name: "Proxy only", color: "#4c9be8" },
    { id: "real", name: "Real only", color: "#6fbf73" },
    { id: "mi", name: "MI only", color: "#e3b341" },
    { id: "random", name: "Random", color: "#b7aea4" },
  ];

  const curves = document.getElementById("curves");
  const tsne = document.getElementById("tsne");
  const fill = document.getElementById("budget-fill");
  const budgetText = document.getElementById("budget-text");
  const readout = document.getElementById("readout");
  const inspect = document.getElementById("inspect");
  const proxyImg = document.getElementById("proxy-img");
  const realImg = document.getElementById("real-img");
  const proxyScore = document.getElementById("proxy-score");
  const realScore = document.getElementById("real-score");

  let demo = null;
  let seedSlot = 0;
  let stepIndex = 0;
  let timer = null;

  function current() { return demo.seeds[seedSlot]; }
  function clip() { return current().clips[stepIndex]; }
  function shown() { return clip().at; }

  function severeCount(order, n) {
    let count = 0;
    for (let i = 0; i < n; i++) count += demo.points[order[i]][2];
    return count;
  }

  function series(order, n) {
    const out = [];
    let count = 0;
    for (let i = 0; i < n; i++) {
      count += demo.points[order[i]][2];
      out.push(count);
    }
    return out;
  }

  function resizeCanvas(canvas, ratio) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(320, Math.floor(rect.width * dpr));
    canvas.width = width;
    canvas.height = Math.floor(width * ratio);
  }

  function resize() {
    resizeCanvas(curves, 11 / 16);
    resizeCanvas(tsne, 1);
  }

  function drawCurves() {
    const ctx = curves.getContext("2d");
    const w = curves.width;
    const h = curves.height;
    ctx.clearRect(0, 0, w, h);
    const n = shown();
    const total = demo.budget;
    const left = Math.round(w * 0.12);
    const right = Math.round(w * 0.04);
    const top = Math.round(h * 0.14);
    const bottom = Math.round(h * 0.16);
    const plotW = w - left - right;
    const plotH = h - top - bottom;
    let yMax = 4;
    for (const method of METHODS) {
      yMax = Math.max(yMax, severeCount(current().orders[method.id], total));
    }
    yMax = Math.ceil(yMax * 1.08);

    function xAt(i) { return left + (total <= 1 ? 0 : i / (total - 1)) * plotW; }
    function yAt(v) { return top + (1 - v / yMax) * plotH; }

    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.strokeRect(left, top, plotW, plotH);
    ctx.fillStyle = "#f6f1ea";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `${Math.round(h * 0.045)}px "Source Sans 3", sans-serif`;
    ctx.fillText("Severe failures on real KITTI", left + plotW / 2, top * 0.42);
    ctx.fillStyle = "#c8bfb4";
    ctx.font = `${Math.round(h * 0.036)}px "Source Sans 3", sans-serif`;
    ctx.fillText("real labels", left + plotW / 2, h - bottom * 0.38);
    ctx.save();
    ctx.translate(left * 0.32, top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`count with score ≥ ${demo.threshold}`, 0, 0);
    ctx.restore();

    const initX = xAt(demo.nInit - 1);
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = "rgba(255,255,255,0.28)";
    ctx.beginPath();
    ctx.moveTo(initX, top);
    ctx.lineTo(initX, top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    for (const method of METHODS) {
      const values = series(current().orders[method.id], n);
      ctx.beginPath();
      ctx.strokeStyle = method.color;
      ctx.lineWidth = Math.max(2.4, w / 360);
      values.forEach((v, i) => {
        const x = xAt(i);
        const y = yAt(v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.beginPath();
      ctx.fillStyle = method.color;
      ctx.arc(xAt(n - 1), yAt(values[values.length - 1]), Math.max(3.5, w / 200), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawTsne() {
    const ctx = tsne.getContext("2d");
    const w = tsne.width;
    const h = tsne.height;
    ctx.clearRect(0, 0, w, h);
    const n = shown();
    const cols = 3;
    const rows = 2;
    const gap = Math.round(w * 0.035);
    const head = Math.round(h * 0.055);
    const foot = Math.round(h * 0.1);
    const panelW = (w - gap * (cols + 1)) / cols;
    const panelH = (h - head - foot - gap * (rows + 1)) / rows;
    const pad = Math.round(Math.min(panelW, panelH) * 0.04);

    function xy(panelX, panelY, point) {
      return [
        panelX + pad + point[0] * (panelW - 2 * pad),
        panelY + pad + (1 - point[1]) * (panelH - 2 * pad),
      ];
    }

    function scoreColor(score, alpha) {
      const t = Math.max(0, Math.min(1, score));
      const r = Math.round(70 + (214 - 70) * t);
      const g = Math.round(98 + (54 - 98) * t);
      const b = Math.round(130 + (28 - 130) * t);
      return `rgba(${r},${g},${b},${alpha})`;
    }

    const barX = Math.round(w * 0.18);
    const barW = Math.round(w * 0.64);
    const barH = Math.max(8, Math.round(h * 0.028));
    const barY = h - Math.round(foot * 0.62);
    const gradient = ctx.createLinearGradient(barX, 0, barX + barW, 0);
    for (let step = 0; step <= 8; step++) gradient.addColorStop(step / 8, scoreColor(step / 8, 1));
    ctx.fillStyle = gradient;
    ctx.fillRect(barX, barY, barW, barH);
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.strokeRect(barX, barY, barW, barH);
    const mark = barX + demo.threshold * barW;
    ctx.beginPath();
    ctx.strokeStyle = "#f6f1ea";
    ctx.lineWidth = 2;
    ctx.moveTo(mark, barY - 2);
    ctx.lineTo(mark, barY + barH + 2);
    ctx.stroke();
    ctx.fillStyle = "#e7dfd6";
    ctx.font = `${Math.round(h * 0.028)}px "Source Sans 3", sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("missed-car score", barX + barW / 2, barY - 3);
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText("0", barX, barY + barH + 2);
    ctx.textAlign = "right";
    ctx.fillText("1", barX + barW, barY + barH + 2);
    ctx.textAlign = "center";
    ctx.fillText("0.45", mark, barY + barH + 2);

    METHODS.forEach((method, k) => {
      const row = Math.floor(k / cols);
      const col = k % cols;
      const rowCount = Math.min(cols, METHODS.length - row * cols);
      const rowWidth = rowCount * panelW + (rowCount - 1) * gap;
      const x0 = (w - rowWidth) / 2 + col * (panelW + gap);
      const y0 = head + gap + row * (panelH + gap);
      ctx.strokeStyle = "rgba(255,255,255,0.16)";
      ctx.strokeRect(x0, y0, panelW, panelH);
      ctx.fillStyle = "#f6f1ea";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.font = `${Math.round(h * 0.032)}px "Source Sans 3", sans-serif`;
      ctx.fillText(method.name, x0 + panelW / 2, y0 - head * 0.35);

      const order = current().orders[method.id];
      const covered = new Set();
      for (let i = 0; i < n; i++) {
        const point = demo.points[order[i]];
        if (point[2]) covered.add(point[3]);
      }
      demo.modes.forEach((mode, id) => {
        const hull = mode.hull.map((vertex) => xy(x0, y0, vertex));
        ctx.beginPath();
        hull.forEach((vertex, i) => {
          if (i === 0) ctx.moveTo(vertex[0], vertex[1]);
          else ctx.lineTo(vertex[0], vertex[1]);
        });
        ctx.closePath();
        ctx.fillStyle = scoreColor(mode.severity, covered.has(id) ? 0.38 : 0.16);
        ctx.fill();
        ctx.strokeStyle = covered.has(id) ? method.color : "rgba(255,255,255,0.2)";
        ctx.lineWidth = covered.has(id) ? Math.max(1.6, w / 420) : 1;
        ctx.stroke();
      });

      const radius = Math.max(4.5, w / 145);
      for (let i = 0; i < n; i++) {
        const point = demo.points[order[i]];
        const [x, y] = xy(x0, y0, point);
        ctx.beginPath();
        ctx.fillStyle = scoreColor(point[4], 1);
        ctx.arc(x, y, point[2] ? radius * 1.35 : radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(22,16,12,0.55)";
        ctx.stroke();
      }
    });
  }

  function render() {
    drawCurves();
    drawTsne();
    const frame = clip();
    const n = frame.at;
    const run = current();
    realImg.src = frame.real;
    proxyImg.src = frame.proxyImg;
    realScore.textContent = frame.severe
      ? `severe failure ${frame.target.toFixed(2)} · sequence ${frame.seq}, frame ${frame.frame}`
      : `score ${frame.target.toFixed(2)} · sequence ${frame.seq}, frame ${frame.frame}`;
    proxyScore.textContent = "Same frame in the Virtual KITTI clone. The curve uses the real score only.";
    fill.style.width = `${(100 * n) / demo.budget}%`;
    budgetText.textContent = `real labels  ${n} / ${demo.budget}`;
    document.getElementById("n-used").textContent = String(n);
    const oursNow = severeCount(run.orders.ours, n);
    const proxyNow = severeCount(run.orders.proxy, n);
    const realNow = severeCount(run.orders.real, n);
    document.getElementById("n-severe").textContent = String(oursNow);
    document.getElementById("n-proxy").textContent = String(proxyNow);
    document.getElementById("n-real").textContent = String(realNow);
    inspect.textContent = `Ours has found ${oursNow} severe real failures. Proxy only has ${proxyNow}. Real only has ${realNow}. Dot color is the missed-car score. An outline means that method has a severe label in that island.`;
  }

  function step() {
    if (stepIndex >= current().clips.length - 1) {
      pause();
      readout.textContent = "All 85 real labels are on the plots. New seed switches to another run.";
      return;
    }
    stepIndex += 1;
    const frame = clip();
    const previous = current().clips[stepIndex - 1].at;
    readout.textContent = frame.severe
      ? `Labels ${previous + 1}–${frame.at}: ours labeled a severe frame, score ${frame.target.toFixed(2)}.`
      : `Labels ${previous + 1}–${frame.at}: this clip is score ${frame.target.toFixed(2)}.`;
    render();
  }

  function play() { if (!timer) timer = setInterval(step, 900); }
  function pause() { clearInterval(timer); timer = null; }

  function reset(slot) {
    pause();
    seedSlot = slot % demo.seeds.length;
    stepIndex = 0;
    readout.textContent = `Seed ${current().seed}. All methods share the first ${demo.nInit} real labels, then each spends the rest of the ${demo.budget}.`;
    render();
  }

  document.getElementById("play").onclick = play;
  document.getElementById("pause").onclick = pause;
  document.getElementById("step").onclick = () => { pause(); step(); };
  document.getElementById("reset").onclick = () => reset(seedSlot + 1);
  window.addEventListener("resize", () => { resize(); if (demo) render(); });

  fetch("assets/kitti_demo.json?v=5")
    .then((response) => response.json())
    .then((data) => {
      demo = data;
      resize();
      reset(0);
      play();
    })
    .catch(() => {
      readout.textContent = "Could not load the KITTI acquisition.";
    });

  const video = document.getElementById("paper-video");
  const missing = document.getElementById("video-missing");
  if (video && missing) video.addEventListener("error", () => { missing.hidden = false; });
})();
