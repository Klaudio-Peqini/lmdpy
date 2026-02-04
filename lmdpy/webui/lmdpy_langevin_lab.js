
// lmdpy_langevin_lab.js
//
// In-browser Langevin dynamics playground inspired by lmdpy.
//
// 3D overdamped Langevin dynamics with rich diagnostics:
//
//   State: r = (x, y, z)
//
//   Potentials:
//     - "harmonic":  0.5 k |r|^2
//     - "pdg":       0.5 k |r|^2 - A_cd exp(-|r|^2/(2 xi^2))
//     - "double_well": a x^4 - b x^2 + 0.5 k (y^2 + z^2)
//     - "lj":        Lennard-Jones centered at origin
//     - "morse":     Morse well centered at origin
//
//   Panels (all updated in real time):
//     1) x-component vs time  (position or velocity)
//     2) y-component vs time  (position or velocity)
//     3) z-component vs time  (position or velocity)
//     4) V(x) (cut at y = z = 0)
//     5) 3D trajectory (projected)
//     6) MSD(t) + VACF(t)
//     7) Energies Ek, Ep, Et
//     8) Position distribution P(x) histogram
//
// Layout features:
//   - Auto-scaling canvases to container width
//   - Responsive resize on window resize
//   - Buttons to switch 1-column / 2-column / 3-column layouts
//   - Toggle to display time-series as positions or velocities
//
// Usage in HTML:
//
//   <div id="lmdpy-app"></div>
//   <script src="lmdpy_langevin_lab.js"></script>
//
// The script takes care of plot layout + scaling.

(function () {
  "use strict";

  // --- Helpers -----------------------------------------------------------

  function createElement(tag, attrs = {}, parent = null) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") {
        el.textContent = v;
      } else if (k === "html") {
        el.innerHTML = v;
      } else {
        el.setAttribute(k, v);
      }
    }
    if (parent) parent.appendChild(el);
    return el;
  }

  function clamp(x, min, max) {
    return Math.max(min, Math.min(max, x));
  }

  function gaussian(rng = Math.random) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // --- Langevin Engine: 3D overdamped ------------------------------------

  class Langevin3D {
    constructor(params = {}) {
      this.gamma = params.gamma ?? 1.0;
      this.kT = params.kT ?? 1.0;
      this.dt = params.dt ?? 0.01;
      this.mass = params.mass ?? 1.0;
      this.potentialType = params.potentialType ?? "pdg";

      // Harmonic / PDG params
      this.kHarm = params.kHarm ?? 1.0;
      this.Acd = params.Acd ?? 1.0;
      this.xi = params.xi ?? 1.0;

      // Double-well params (x only)
      this.dwA = params.dwA ?? 1.0;
      this.dwB = params.dwB ?? 1.0;

      // Lennard-Jones params
      this.ljEps = params.ljEps ?? 1.0;
      this.ljSigma = params.ljSigma ?? 1.0;
      this.ljRcut = params.ljRcut ?? 0.0; // 0 => no cutoff

      // Morse params
      this.morseDe = params.morseDe ?? 1.0;
      this.morseA = params.morseA ?? 1.0;
      this.morseR0 = params.morseR0 ?? 1.0;

      this.running = false;
      this.trajectory = [];
      this.maxPoints = 5000;

      this.rng = Math.random;
      this.reset([0.5, 0.0, 0.0]);
    }

    V(r) {
      const x = r[0], y = r[1], z = r[2];
      if (this.potentialType === "harmonic") {
        const r2 = x * x + y * y + z * z;
        return 0.5 * this.kHarm * r2;
      } else if (this.potentialType === "pdg") {
        const r2 = x * x + y * y + z * z;
        const U_h = 0.5 * this.kHarm * r2;
        const U_cd = -this.Acd * Math.exp(-r2 / (2 * this.xi * this.xi));
        return U_h + U_cd;
      } else if (this.potentialType === "double_well") {
        const r2_perp = y * y + z * z;
        const Vx = this.dwA * x * x * x * x - this.dwB * x * x;
        const VyZ = 0.5 * this.kHarm * r2_perp;
        return Vx + VyZ;
      } else if (this.potentialType === "lj") {
        const rnorm = Math.sqrt(x * x + y * y + z * z);
        const eps = this.ljEps;
        const sigma = this.ljSigma;
        const rc = this.ljRcut;
        const tiny = 1e-8;
        const r = rnorm < tiny ? tiny : rnorm;
        const sr = sigma / r;
        const sr6 = Math.pow(sr, 6);
        const sr12 = sr6 * sr6;
        let U = 4.0 * eps * (sr12 - sr6);
        if (rc > 0.0 && rnorm > rc) {
          U = 0.0; // simple cutoff
        }
        return U;
      } else if (this.potentialType === "morse") {
        const rnorm = Math.sqrt(x * x + y * y + z * z);
        const De = this.morseDe;
        const a = this.morseA;
        const r0 = this.morseR0;
        const yexp = Math.exp(-a * (rnorm - r0));
        return De * Math.pow(1.0 - yexp, 2) - De;
      }
      return 0.0;
    }

    gradV(r) {
      const x = r[0], y = r[1], z = r[2];
      if (this.potentialType === "harmonic") {
        return [
          this.kHarm * x,
          this.kHarm * y,
          this.kHarm * z,
        ];
      } else if (this.potentialType === "pdg") {
        const r2 = x * x + y * y + z * z;
        const expTerm = Math.exp(-r2 / (2 * this.xi * this.xi));
        const dU_dr2 =
          0.5 * this.kHarm
          - this.Acd * (-1.0 / (2 * this.xi * this.xi)) * expTerm;
        const factor = dU_dr2 * 2.0;
        return [
          factor * x,
          factor * y,
          factor * z,
        ];
      } else if (this.potentialType === "double_well") {
        const dVdx = 4 * this.dwA * x * x * x - 2 * this.dwB * x;
        const dVdy = this.kHarm * y;
        const dVdz = this.kHarm * z;
        return [dVdx, dVdy, dVdz];
      } else if (this.potentialType === "lj") {
        const eps = this.ljEps;
        const sigma = this.ljSigma;
        const rc = this.ljRcut;
        const r2 = x * x + y * y + z * z;
        const tiny = 1e-8;
        const r = Math.sqrt(r2);
        if (r < tiny) {
          return [0.0, 0.0, 0.0];
        }
        if (rc > 0.0 && r > rc) {
          return [0.0, 0.0, 0.0];
        }
        const sr = sigma / r;
        const sr6 = Math.pow(sr, 6);
        const sr12 = sr6 * sr6;
        const dU_dr = 24.0 * eps * (2.0 * sr12 - sr6) / r;
        const factor = dU_dr / r;
        return [factor * x, factor * y, factor * z];
      } else if (this.potentialType === "morse") {
        const De = this.morseDe;
        const a = this.morseA;
        const r0 = this.morseR0;
        const r2 = x * x + y * y + z * z;
        const tiny = 1e-8;
        const r = Math.sqrt(r2);
        if (r < tiny) {
          return [0.0, 0.0, 0.0];
        }
        const yexp = Math.exp(-a * (r - r0));
        const dU_dr = 2.0 * De * (1.0 - yexp) * (a * yexp);
        const factor = dU_dr / r;
        return [factor * x, factor * y, factor * z];
      }
      return [0.0, 0.0, 0.0];
    }

    step() {
      const dt = this.dt;
      const gamma = this.gamma;
      const kT = this.kT;
      const sigma = Math.sqrt(2.0 * kT / gamma);

      const dW = [
        Math.sqrt(dt) * gaussian(this.rng),
        Math.sqrt(dt) * gaussian(this.rng),
        Math.sqrt(dt) * gaussian(this.rng),
      ];

      const grad = this.gradV(this.r);
      const drift = [
        -grad[0] / gamma,
        -grad[1] / gamma,
        -grad[2] / gamma,
      ];

      const r_prev = [...this.r];

      this.r = [
        this.r[0] + drift[0] * dt + sigma * dW[0],
        this.r[1] + drift[1] * dt + sigma * dW[1],
        this.r[2] + drift[2] * dt + sigma * dW[2],
      ];
      this.t += dt;

      const vx = (this.r[0] - r_prev[0]) / dt;
      const vy = (this.r[1] - r_prev[1]) / dt;
      const vz = (this.r[2] - r_prev[2]) / dt;
      const v2 = vx * vx + vy * vy + vz * vz;
      const Ek = 0.5 * this.mass * v2;
      const Ep = this.V(this.r);
      const Et = Ek + Ep;

      this.trajectory.push({
        t: this.t,
        r: [...this.r],
        v: [vx, vy, vz],
        Ek: Ek,
        Ep: Ep,
        Et: Et,
      });
      if (this.trajectory.length > this.maxPoints) {
        this.trajectory.shift();
      }
    }

    reset(r0 = [0.5, 0.0, 0.0]) {
      this.r = [...r0];
      this.t = 0.0;
      const Ep0 = this.V(this.r);
      this.trajectory = [{
        t: 0.0,
        r: [...this.r],
        v: [0.0, 0.0, 0.0],
        Ek: 0.0,
        Ep: Ep0,
        Et: Ep0,
      }];
    }

    currentX() {
      return this.r[0];
    }
  }

  // --- App / UI ----------------------------------------------------------

  class LangevinApp {
    constructor(containerId = "lmdpy-app") {
      const container = document.getElementById(containerId);
      if (!container) {
        console.error("lmdpy: container #", containerId, "not found.");
        return;
      }
      this.container = container;

      this.engine = new Langevin3D({ potentialType: "pdg" });
      this.layoutMode = 2;
      this.timeSeriesMode = "position";
      this.plotContainer = null;

      this._buildUI();
      this._buildCanvas();

      this.lastFrameTime = null;
      this.engine.reset([0.5, 0.0, 0.0]);
      this.engine.running = false;

      this._loop = this._loop.bind(this);
      this._resizeCanvases = this._resizeCanvases.bind(this);

      window.addEventListener("resize", this._resizeCanvases);
      requestAnimationFrame(this._loop);
    }

    _buildUI() {
      const root = this.container;
      root.innerHTML = "";

      createElement("h2", { text: "lmdpy Langevin Lab (3D JS)" }, root);

      const controls = createElement("div", { class: "lmdpy-controls" }, root);

      const potRow = createElement("div", { class: "lmdpy-row" }, controls);
      createElement("span", { text: "Potential: " }, potRow);
      const potSelect = createElement("select", {}, potRow);
      [
        ["pdg", "P–DG coherent"],
        ["harmonic", "Harmonic"],
        ["double_well", "Double well (x)"],
        ["lj", "Lennard-Jones (center)"],
        ["morse", "Morse (center)"],
      ].forEach(([val, label]) => {
        const opt = createElement("option", { value: val, text: label }, potSelect);
        if (val === "pdg") opt.selected = true;
      });
      potSelect.addEventListener("change", () => {
        this.engine.potentialType = potSelect.value;
        this._updateParamVisibility();
      });

      const makeSliderRow = (label, min, max, step, initial, onChange) => {
        const row = createElement("div", { class: "lmdpy-row" }, controls);
        createElement("span", { text: label + ": " }, row);
        const input = createElement("input", {
          type: "range",
          min: String(min),
          max: String(max),
          step: String(step),
          value: String(initial)
        }, row);
        const box = createElement("input", {
          type: "number",
          value: String(initial),
          step: String(step),
          style: "width:5em; margin-left:0.5em;"
        }, row);

        const sync = (val) => {
          const v = parseFloat(val);
          if (Number.isFinite(v)) {
            input.value = String(v);
            box.value = String(v);
            onChange(v);
          }
        };

        input.addEventListener("input", () => sync(input.value));
        box.addEventListener("change", () => sync(box.value));
        onChange(initial);

        return { row, input, box };
      };

      // Global friction, temperature, dt with wider ranges
      this.gammaCtrl = makeSliderRow("γ (friction)", 0.1, 10.0, 0.1, this.engine.gamma, v => {
        this.engine.gamma = v;
      });

      this.kTCtrl = makeSliderRow("kT", 0.05, 10.0, 0.05, this.engine.kT, v => {
        this.engine.kT = v;
      });

      this.dtCtrl = makeSliderRow("dt", 0.0005, 0.05, 0.0005, this.engine.dt, v => {
        this.engine.dt = v;
      });

      // Harmonic / PDG
      this.kHarmCtrl = makeSliderRow("k_harm", 0.0, 10.0, 0.1, this.engine.kHarm, v => {
        this.engine.kHarm = v;
      });
      this.AcdCtrl = makeSliderRow("A_cd", 0.0, 10.0, 0.1, this.engine.Acd, v => {
        this.engine.Acd = v;
      });
      this.xiCtrl = makeSliderRow("xi", 0.1, 10.0, 0.1, this.engine.xi, v => {
        this.engine.xi = v;
      });

      // Double-well
      this.dwACtrl = makeSliderRow("a (double well)", 0.0, 10.0, 0.1, this.engine.dwA, v => {
        this.engine.dwA = v;
      });
      this.dwBCtrl = makeSliderRow("b (double well)", 0.0, 10.0, 0.1, this.engine.dwB, v => {
        this.engine.dwB = v;
      });

      // Lennard-Jones
      this.ljEpsCtrl = makeSliderRow("ε (LJ)", 0.0, 10.0, 0.1, this.engine.ljEps, v => {
        this.engine.ljEps = v;
      });
      this.ljSigmaCtrl = makeSliderRow("σ (LJ)", 0.1, 5.0, 0.1, this.engine.ljSigma, v => {
        this.engine.ljSigma = v;
      });
      this.ljRcutCtrl = makeSliderRow("r_cut (LJ)", 0.0, 10.0, 0.1, this.engine.ljRcut, v => {
        this.engine.ljRcut = v;
      });

      // Morse
      this.morseDeCtrl = makeSliderRow("D_e (Morse)", 0.0, 10.0, 0.1, this.engine.morseDe, v => {
        this.engine.morseDe = v;
      });
      this.morseACtrl = makeSliderRow("a (Morse)", 0.1, 10.0, 0.1, this.engine.morseA, v => {
        this.engine.morseA = v;
      });
      this.morseR0Ctrl = makeSliderRow("r0 (Morse)", 0.0, 5.0, 0.1, this.engine.morseR0, v => {
        this.engine.morseR0 = v;
      });

      const btnRow = createElement("div", { class: "lmdpy-row" }, controls);
      const btnStart = createElement("button", { text: "Start" }, btnRow);
      const btnStop = createElement("button", { text: "Stop", style: "margin-left:0.5em;" }, btnRow);
      const btnReset = createElement("button", { text: "Reset", style: "margin-left:0.5em;" }, btnRow);

      btnStart.addEventListener("click", () => {
        this.engine.running = true;
      });
      btnStop.addEventListener("click", () => {
        this.engine.running = false;
      });
      btnReset.addEventListener("click", () => {
        this.engine.reset([0.5, 0.0, 0.0]);
      });

      const layoutRow = createElement("div", { class: "lmdpy-row" }, controls);
      createElement("span", { text: "Layout: " }, layoutRow);
      const btn1 = createElement("button", { text: "1 column" }, layoutRow);
      const btn2 = createElement("button", { text: "2 columns", style: "margin-left:0.5em;" }, layoutRow);
      const btn3 = createElement("button", { text: "3 columns", style: "margin-left:0.5em;" }, layoutRow);

      btn1.addEventListener("click", () => this._setLayoutMode(1));
      btn2.addEventListener("click", () => this._setLayoutMode(2));
      btn3.addEventListener("click", () => this._setLayoutMode(3));

      const tsRow = createElement("div", { class: "lmdpy-row" }, controls);
      createElement("span", { text: "Time series: " }, tsRow);
      const btnPos = createElement("button", { text: "Position" }, tsRow);
      const btnVel = createElement("button", { text: "Velocity", style: "margin-left:0.5em;" }, tsRow);

      const updateTSButtons = () => {
        btnPos.style.backgroundColor = this.timeSeriesMode === "position" ? "#555" : "";
        btnVel.style.backgroundColor = this.timeSeriesMode === "velocity" ? "#555" : "";
      };

      btnPos.addEventListener("click", () => {
        this.timeSeriesMode = "position";
        updateTSButtons();
      });
      btnVel.addEventListener("click", () => {
        this.timeSeriesMode = "velocity";
        updateTSButtons();
      });
      updateTSButtons();

      createElement("p", {
        html:
          "3D overdamped Langevin simulation with multiple potentials (P–DG, harmonic, double-well, Lennard-Jones, Morse). " +
          "Use layout (1/2/3 columns) and time-series (position/velocity) toggles. " +
          "All panels update in real time."
      }, root);

      this._updateParamVisibility = () => {
        const type = this.engine.potentialType;

        const showPDG = (type === "pdg");
        const showHarm = (type === "harmonic");
        const showDW = (type === "double_well");
        const showLJ = (type === "lj");
        const showMorse = (type === "morse");

        this.kHarmCtrl.row.style.display =
          (showPDG || showHarm || showDW) ? "flex" : "none";
        this.AcdCtrl.row.style.display = showPDG ? "flex" : "none";
        this.xiCtrl.row.style.display = showPDG ? "flex" : "none";

        this.dwACtrl.row.style.display = showDW ? "flex" : "none";
        this.dwBCtrl.row.style.display = showDW ? "flex" : "none";

        this.ljEpsCtrl.row.style.display = showLJ ? "flex" : "none";
        this.ljSigmaCtrl.row.style.display = showLJ ? "flex" : "none";
        this.ljRcutCtrl.row.style.display = showLJ ? "flex" : "none";

        this.morseDeCtrl.row.style.display = showMorse ? "flex" : "none";
        this.morseACtrl.row.style.display = showMorse ? "flex" : "none";
        this.morseR0Ctrl.row.style.display = showMorse ? "flex" : "none";
      };

      this._updateParamVisibility();
    }

    _buildCanvas() {
      const plotContainer = createElement("div", { class: "lmdpy-plots" }, this.container);
      this.plotContainer = plotContainer;

      this.canvasX = createElement("canvas", {}, plotContainer);
      this.canvasY = createElement("canvas", {}, plotContainer);
      this.canvasZ = createElement("canvas", {}, plotContainer);
      this.canvasPot = createElement("canvas", {}, plotContainer);
      this.canvasTraj3D = createElement("canvas", {}, plotContainer);
      this.canvasMSDVacf = createElement("canvas", {}, plotContainer);
      this.canvasEnergy = createElement("canvas", {}, plotContainer);
      this.canvasPDF = createElement("canvas", {}, plotContainer);

      this._setLayoutMode(this.layoutMode);
      this._resizeCanvases();
    }

    _setLayoutMode(nCols) {
      this.layoutMode = nCols;
      if (!this.plotContainer) return;

      const pc = this.plotContainer;
      if (nCols === 1) {
        pc.style.display = "flex";
        pc.style.flexDirection = "column";
        pc.style.gap = "8px";
        pc.style.gridTemplateColumns = "";
      } else {
        pc.style.display = "grid";
        pc.style.gap = "8px";
        pc.style.gridTemplateColumns = `repeat(${nCols}, minmax(0, 1fr))`;
        pc.style.flexDirection = "";
      }
      this._resizeCanvases();
    }

    _resizeCanvases() {
      if (!this.plotContainer) return;

      const pc = this.plotContainer;
      const w = pc.clientWidth || 600;

      const heights = [
        180,
        180,
        180,
        200,
        260,
        220,
        220,
        220,
      ];

      const canvases = [
        this.canvasX,
        this.canvasY,
        this.canvasZ,
        this.canvasPot,
        this.canvasTraj3D,
        this.canvasMSDVacf,
        this.canvasEnergy,
        this.canvasPDF,
      ];

      canvases.forEach((cv, i) => {
        if (!cv) return;
        cv.style.width = "100%";
        cv.style.height = heights[i] + "px";
        cv.width = w;
        cv.height = heights[i];
      });
    }

    _loop(timestamp) {
      if (this.lastFrameTime == null) {
        this.lastFrameTime = timestamp;
      }
      this.lastFrameTime = timestamp;

      if (this.engine.running) {
        const stepsPerFrame = 5;
        for (let i = 0; i < stepsPerFrame; i++) {
          this.engine.step();
        }
      }

      this._draw();
      requestAnimationFrame(this._loop);
    }

    _draw() {
      this._drawComponentTS(this.canvasX, 0, "x");
      this._drawComponentTS(this.canvasY, 1, "y");
      this._drawComponentTS(this.canvasZ, 2, "z");
      this._drawPotential();
      this._drawTrajectory3D();
      this._drawMSDVacf();
      this._drawEnergies();
      this._drawPositionPDF();
    }

    _drawComponentTS(canvas, compIndex, label) {
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

      const traj = this.engine.trajectory;
      if (traj.length < 2) return;

      const tMin = traj[0].t;
      const tMax = traj[traj.length - 1].t || 1.0;

      const vals = [];
      for (const p of traj) {
        if (this.timeSeriesMode === "position") {
          vals.push(p.r[compIndex]);
        } else {
          vals.push(p.v[compIndex]);
        }
      }

      let vMin = +Infinity, vMax = -Infinity;
      for (const v of vals) {
        if (v < vMin) vMin = v;
        if (v > vMax) vMax = v;
      }
      if (!isFinite(vMin) || !isFinite(vMax)) return;
      if (vMax === vMin) {
        vMax += 1.0;
        vMin -= 1.0;
      }

      function tToPx(t, tMin, tMax, w) {
        return ((t - tMin) / (tMax - tMin)) * (w - 40) + 20;
      }
      function vToPy(v, vMin, vMax, h) {
        return h - (((v - vMin) / (vMax - vMin)) * (h - 40) + 20);
      }

      ctx.strokeStyle = "#4fc3f7";
      ctx.beginPath();
      let first = true;
      for (let i = 0; i < traj.length; i++) {
        const p = traj[i];
        const px = tToPx(p.t, tMin, tMax, w);
        const py = vToPy(vals[i], vMin, vMax, h);
        if (first) {
          ctx.moveTo(px, py);
          first = false;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();

      const last = traj[traj.length - 1];
      const lastVal = vals[vals.length - 1];
      const pxLast = tToPx(last.t, tMin, tMax, w);
      const pyLast = vToPy(lastVal, vMin, vMax, h);
      ctx.fillStyle = "#ffeb3b";
      ctx.beginPath();
      ctx.arc(pxLast, pyLast, 4, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = "#fff";
      ctx.font = "12px sans-serif";
      const modeLabel = this.timeSeriesMode === "position" ? label + "(t)" : "v_" + label + "(t)";
      ctx.fillText(modeLabel, 10, 20);
      ctx.fillText(`${modeLabel} ≈ ${lastVal.toFixed(3)}, t ≈ ${last.t.toFixed(2)}`, 10, 36);
    }

    _drawPotential() {
      const ctx = this.canvasPot.getContext("2d");
      const canvas = this.canvasPot;
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

      const xMin = -6.0;
      const xMax = 6.0;
      const nPts = 400;

      let Vmin = +Infinity, Vmax = -Infinity;
      const xs = [];
      const Vs = [];

      for (let i = 0; i < nPts; i++) {
        const x = xMin + (i / (nPts - 1)) * (xMax - xMin);
        const r = [x, 0.0, 0.0];
        const V = this.engine.V(r);
        xs.push(x);
        Vs.push(V);
        if (V < Vmin) Vmin = V;
        if (V > Vmax) Vmax = V;
      }

      if (!isFinite(Vmin) || !isFinite(Vmax)) return;
      if (Vmax === Vmin) Vmax = Vmin + 1.0;

      function xToPx(x) {
        return ((x - xMin) / (xMax - xMin)) * (w - 40) + 20;
      }
      function VToPy(V) {
        const pad = 0.1 * (Vmax - Vmin);
        const VminPlot = Vmin - pad;
        const VmaxPlot = Vmax + pad;
        return h - (((V - VminPlot) / (VmaxPlot - VminPlot)) * (h - 40) + 20);
      }

      ctx.strokeStyle = "#90caf9";
      ctx.beginPath();
      xs.forEach((x, i) => {
        const V = Vs[i];
        const px = xToPx(x);
        const py = VToPy(V);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();

      const xcur = clamp(this.engine.r[0], xMin, xMax);
      const px = xToPx(xcur);
      const Vcur = this.engine.V([xcur, 0.0, 0.0]);
      const py = VToPy(Vcur);
      ctx.fillStyle = "#ffeb3b";
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = "#fff";
      ctx.font = "12px sans-serif";
      ctx.fillText("V(x) (y=z=0)", 10, 20);
      ctx.fillText(`V(x) ≈ ${Vcur.toFixed(3)}`, 10, 36);
    }

    _drawTrajectory3D() {
      const ctx = this.canvasTraj3D.getContext("2d");
      const canvas = this.canvasTraj3D;
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

      const traj = this.engine.trajectory;
      if (traj.length < 2) return;

      let xmin = +Infinity, xmax = -Infinity;
      let ymin = +Infinity, ymax = -Infinity;
      let zmin = +Infinity, zmax = -Infinity;

      for (const p of traj) {
        const r = p.r;
        if (r[0] < xmin) xmin = r[0];
        if (r[0] > xmax) xmax = r[0];
        if (r[1] < ymin) ymin = r[1];
        if (r[1] > ymax) ymax = r[1];
        if (r[2] < zmin) zmin = r[2];
        if (r[2] > zmax) zmax = r[2];
      }

      if (!isFinite(xmin) || !isFinite(xmax)) return;

      const pad = 0.1;
      const xRange = xmax - xmin || 1.0;
      const yRange = ymax - ymin || 1.0;
      const zRange = zmax - zmin || 1.0;
      const maxRange = Math.max(xRange, yRange, zRange);

      xmin -= pad * maxRange;
      xmax += pad * maxRange;
      ymin -= pad * maxRange;
      ymax += pad * maxRange;
      zmin -= pad * maxRange;
      zmax += pad * maxRange;

      const angleZ = -0.6;
      const angleX = 0.7;
      const cosZ = Math.cos(angleZ), sinZ = Math.sin(angleZ);
      const cosX = Math.cos(angleX), sinX = Math.sin(angleX);

      function transform(r) {
        let x = (r[0] - (xmin + xmax) / 2) / maxRange;
        let y = (r[1] - (ymin + ymax) / 2) / maxRange;
        let z = (r[2] - (zmin + zmax) / 2) / maxRange;

        let xz = cosZ * x - sinZ * y;
        let yz = sinZ * x + cosZ * y;
        let zz = z;

        let xx = xz;
        let yx = cosX * yz - sinX * zz;
        let zx = sinX * yz + cosX * zz;

        return [xx, yx, zx];
      }

      function projXY(r3, w, h) {
        const margin = 20;
        const scale = (Math.min(w, h) - 2 * margin) / 2;
        const X = w / 2 + scale * r3[0];
        const Y = h / 2 - scale * r3[1];
        return [X, Y];
      }

      ctx.strokeStyle = "#4caf50";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let first = true;
      let lastScreen = null;

      for (const p of traj) {
        const r3 = transform(p.r);
        const [X, Y] = projXY(r3, w, h);
        if (first) {
          ctx.moveTo(X, Y);
          first = false;
        } else {
          ctx.lineTo(X, Y);
        }
        lastScreen = [X, Y];
      }
      ctx.stroke();

      if (lastScreen) {
        ctx.fillStyle = "#ffeb3b";
        ctx.beginPath();
        ctx.arc(lastScreen[0], lastScreen[1], 5, 0, 2 * Math.PI);
        ctx.fill();
      }

      ctx.fillStyle = "#fff";
      ctx.font = "12px sans-serif";
      ctx.fillText("3D trajectory (projected)", 10, 20);
      const last = traj[traj.length - 1];
      ctx.fillText(
        `r ≈ (${last.r[0].toFixed(2)}, ${last.r[1].toFixed(2)}, ${last.r[2].toFixed(2)})`,
        10,
        36
      );
    }

    _drawMSDVacf() {
      const ctx = this.canvasMSDVacf.getContext("2d");
      const canvas = this.canvasMSDVacf;
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

      const traj = this.engine.trajectory;
      if (traj.length < 3) return;

      const t0 = traj[0].t;
      const r0 = traj[0].r;
      const v0 = traj[1].v || [0.0, 0.0, 0.0];
      const v0dot = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
      if (v0dot === 0) return;

      const times = [];
      const msd = [];
      const vacf = [];

      for (let i = 0; i < traj.length; i++) {
        const p = traj[i];
        const dx = p.r[0] - r0[0];
        const dy = p.r[1] - r0[1];
        const dz = p.r[2] - r0[2];
        const dr2 = dx * dx + dy * dy + dz * dz;

        const v = p.v;
        const dotvv0 = v[0] * v0[0] + v[1] * v0[1] + v[2] * v0[2];
        const c = dotvv0 / v0dot;

        times.push(p.t - t0);
        msd.push(dr2);
        vacf.push(c);
      }

      const tMin = times[0];
      const tMax = times[times.length - 1] || 1.0;

      let msdMax = 0.0;
      for (const v of msd) if (v > msdMax) msdMax = v;
      if (msdMax <= 0) msdMax = 1.0;

      let vacfMin = +Infinity, vacfMax = -Infinity;
      for (const c of vacf) {
        if (c < vacfMin) vacfMin = c;
        if (c > vacfMax) vacfMax = c;
      }
      if (!isFinite(vacfMin) || !isFinite(vacfMax)) {
        vacfMin = -1.0;
        vacfMax = 1.0;
      }
      if (vacfMax === vacfMin) {
        vacfMax = vacfMin + 1.0;
      }

      function tToPx(t) {
        return ((t - tMin) / (tMax - tMin)) * (w - 40) + 20;
      }
      function msdToPy(m) {
        return h - ((m / msdMax) * (h - 40) + 20);
      }
      function vacfToPy(c) {
        const pad = 0.1 * (vacfMax - vacfMin);
        const vmin = vacfMin - pad;
        const vmax = vacfMax + pad;
        return h - (((c - vmin) / (vmax - vmin)) * (h - 40) + 20);
      }

      ctx.strokeStyle = "#4fc3f7";
      ctx.beginPath();
      let first = true;
      for (let i = 0; i < times.length; i++) {
        const px = tToPx(times[i]);
        const py = msdToPy(msd[i]);
        if (first) {
          ctx.moveTo(px, py);
          first = false;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();

      ctx.strokeStyle = "#ff9800";
      ctx.beginPath();
      first = true;
      for (let i = 0; i < times.length; i++) {
        const px = tToPx(times[i]);
        const py = vacfToPy(vacf[i]);
        if (first) {
          ctx.moveTo(px, py);
          first = false;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();

      ctx.fillStyle = "#fff";
      ctx.font = "12px sans-serif";
      ctx.fillText("MSD(t) & VACF(t)", 10, 20);
      ctx.fillText("blue: MSD, orange: VACF (normalized)", 10, 36);
    }

    _drawEnergies() {
      const ctx = this.canvasEnergy.getContext("2d");
      const canvas = this.canvasEnergy;
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

      const traj = this.engine.trajectory;
      if (traj.length < 2) return;

      const tMin = traj[0].t;
      const tMax = traj[traj.length - 1].t || 1.0;

      const times = [];
      const Ek = [];
      const Ep = [];
      const Et = [];

      let Emin = +Infinity, Emax = -Infinity;

      for (const p of traj) {
        times.push(p.t - tMin);
        Ek.push(p.Ek);
        Ep.push(p.Ep);
        Et.push(p.Et);

        if (p.Ek < Emin) Emin = p.Ek;
        if (p.Ep < Emin) Emin = p.Ep;
        if (p.Et < Emin) Emin = p.Et;
        if (p.Ek > Emax) Emax = p.Ek;
        if (p.Ep > Emax) Emax = p.Ep;
        if (p.Et > Emax) Emax = p.Et;
      }

      if (!isFinite(Emin) || !isFinite(Emax)) return;
      if (Emax === Emin) Emax = Emin + 1.0;

      function tToPx(t) {
        return ((t) / (tMax - tMin || 1.0)) * (w - 40) + 20;
      }
      function EToPy(E) {
        const pad = 0.1 * (Emax - Emin);
        const EminPlot = Emin - pad;
        const EmaxPlot = Emax + pad;
        return h - (((E - EminPlot) / (EmaxPlot - EminPlot)) * (h - 40) + 20);
      }

      function drawSeries(values, color) {
        ctx.strokeStyle = color;
        ctx.beginPath();
        let first = true;
        for (let i = 0; i < times.length; i++) {
          const px = tToPx(times[i]);
          const py = EToPy(values[i]);
          if (first) {
            ctx.moveTo(px, py);
            first = false;
          } else {
            ctx.lineTo(px, py);
          }
        }
        ctx.stroke();
      }

      drawSeries(Ek, "#4caf50");
      drawSeries(Ep, "#2196f3");
      drawSeries(Et, "#f44336");

      ctx.fillStyle = "#fff";
      ctx.font = "12px sans-serif";
      ctx.fillText("Energies vs time", 10, 20);
      ctx.fillText("green: Ek, blue: Ep, red: Et", 10, 36);
    }

    _drawPositionPDF() {
      const ctx = this.canvasPDF.getContext("2d");
      const canvas = this.canvasPDF;
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

      const traj = this.engine.trajectory;
      if (traj.length < 2) return;

      let xmin = +Infinity, xmax = -Infinity;
      for (const p of traj) {
        const x = p.r[0];
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
      }
      if (!isFinite(xmin) || !isFinite(xmax)) return;
      if (xmax === xmin) {
        xmax += 1.0;
        xmin -= 1.0;
      }

      const nBins = 50;
      const bins = new Array(nBins).fill(0);
      const dx = (xmax - xmin) / nBins;

      for (const p of traj) {
        const x = p.r[0];
        let idx = Math.floor((x - xmin) / dx);
        if (idx < 0) idx = 0;
        if (idx >= nBins) idx = nBins - 1;
        bins[idx] += 1;
      }

      const maxCount = bins.reduce((a, b) => Math.max(a, b), 0) || 1;

      const margin = 30;
      const plotW = w - 2 * margin;
      const plotH = h - 2 * margin;

      function binToPx(i) {
        const x0 = margin + (i / nBins) * plotW;
        const x1 = margin + ((i + 1) / nBins) * plotW;
        return [x0, x1];
      }

      ctx.fillStyle = "#4caf50";
      for (let i = 0; i < nBins; i++) {
        const count = bins[i];
        const heightFrac = count / maxCount;
        const barHeight = heightFrac * plotH;
        const [x0, x1] = binToPx(i);
        const bw = x1 - x0 - 1;
        const x = x0 + 0.5;
        const y = h - margin - barHeight;
        ctx.fillRect(x, y, bw, barHeight);
      }

      ctx.strokeStyle = "#888";
      ctx.beginPath();
      ctx.moveTo(margin, h - margin);
      ctx.lineTo(w - margin, h - margin);
      ctx.moveTo(margin, margin);
      ctx.lineTo(margin, h - margin);
      ctx.stroke();

      ctx.fillStyle = "#fff";
      ctx.font = "12px sans-serif";
      ctx.fillText("Position distribution P(x)", 10, 20);
      ctx.fillText("Histogram over recent trajectory", 10, 36);
    }
  }

  window.addEventListener("DOMContentLoaded", () => {
    const el = document.getElementById("lmdpy-app");
    if (el) {
      new LangevinApp("lmdpy-app");
    }
  });
})();
