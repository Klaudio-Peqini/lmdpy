# lmdpy — Langevin Molecular Dynamics in Python

`lmdpy` is a research-grade and educational Python framework for **Langevin dynamics**, **Langevin Monte Carlo**, and **stochastic molecular dynamics**. It is built to be:

- **Physically transparent** (explicit SDEs, forces, noise terms)
- **Extensible** (new potentials, solvents, integrators, diagnostics)
- **HPC-friendly** (ensemble sweeps and HTCondor templates)
- **Teach-and-demo ready** (examples folder + Web UI assets)
- **PDG-oriented** (Preparata–Del Giudice–inspired effective models as a major theme)

This repository is split into two conceptual layers:

1. **`examples/`** — didactic / exploratory scripts (physics-first, explicit, readable)
2. **`lmdpy/`** — the actual reusable **Python library** (`import lmdpy`)

---

## Table of Contents

- [1. Get the code (GitHub)](#1-get-the-code-github)
- [2. Installation](#2-installation)
- [3. Quickstart](#3-quickstart)
- [4. Running the examples](#4-running-the-examples)
- [5. Repository structure](#5-repository-structure)
- [6. Core concepts](#6-core-concepts)
- [7. The examples folder](#7-the-examples-folder)
- [8. The lmdpy library](#8-the-lmdpy-library)
- [9. Diagnostics & figures](#9-diagnostics--figures)
- [10. HPC workflows](#10-hpc-workflows)
- [11. Web UI](#11-web-ui)
- [12. Extending lmdpy](#12-extending-lmdpy)
- [13. Reproducibility](#13-reproducibility)
- [14. Notes & caveats](#14-notes--caveats)
- [15. License & citation](#15-license--citation)
- [16. Contributing (Git workflow)](#16-contributing-git-workflow)

---

## 1. Get the code (GitHub)

### 1.1 Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/lmdpy.git
cd lmdpy
```

If you are using SSH (recommended on HPC/login nodes):

```bash
git clone git@github.com:YOUR_USERNAME/lmdpy.git
cd lmdpy
```

### 1.2 Update your local copy later

```bash
cd lmdpy
git pull
```

---

## 2. Installation

`lmdpy` is currently a source-first library. You can use it immediately via `PYTHONPATH`,
or install it in a virtual environment in “editable” mode once packaging is added.

### 2.1 Recommended: virtual environment + `PYTHONPATH`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip numpy matplotlib
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### 2.2 Alternative: conda environment (common on clusters)

```bash
conda create -n lmdpy python=3.11 -y
conda activate lmdpy
pip install -U numpy matplotlib
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### 2.3 Verify installation

```bash
python -c "import lmdpy; print('lmdpy import OK')"
```

---

## 3. Quickstart

### 3.1 Run a minimal simulation (library usage)

Below is the conceptual pattern: **Potential + Solvent + Integrator**.

```python
import numpy as np
from lmdpy.potentials import HarmonicPotential
from lmdpy.solvent import HomogeneousSolvent
from lmdpy.langevin_core import LangevinIntegrator

pot = HarmonicPotential(k=1.0)
sol = HomogeneousSolvent(gamma=1.0, temperature=1.0)

sim = LangevinIntegrator(
    potential=pot,
    solvent=sol,
    dim=3,
    overdamped=True,
    dt=1e-3,
    integrator="euler",
)

results = sim.simulate(n_steps=5000, x0=np.array([1.0, 0.0, 0.0]))
print(results.keys())
print(results["x"].shape, results["t"].shape)
```

### 3.2 Plot basic diagnostics

```python
import matplotlib.pyplot as plt
from lmdpy.figures import plot_energy, plot_position_distribution

plot_energy(results)
plot_position_distribution(results, component=0)
plt.show()
```

---

## 4. Running the examples

From the repo root:

```bash
python examples/Langevin_general_1D.py
python examples/Langevin_general_3D.py
python examples/pdg_multi_particle.py
```

Examples produce figures interactively (matplotlib). On headless clusters, use:

```bash
export MPLBACKEND=Agg
python examples/pdg_multi_particle.py
```

or run on a login node and save figures to disk (see each script).

---

## 5. Repository structure

```
lmdpy/
├── examples/
│   ├── Langevin_general_1D.py
│   ├── Langevin_general_3D.py
│   ├── Langevin_plottings.py
│   ├── pdg_coherent_field.py
│   ├── pdg_hydrodynamics.py
│   ├── pdg_ligand_receptor.py
│   ├── pdg_multi_particle.py
│   ├── pdg_physical_units.py
│   └── pdg_quantum_effective.py
│
├── lmdpy/
│   ├── __init__.py
│   ├── langevin_core.py
│   ├── potentials.py
│   ├── solvent.py
│   ├── figures.py
│   ├── hpc.py
│   └── webui/
│       └── lmdpy_langevin_lab.js
│
└── README.md
```

---

## 6. Core concepts

The central abstraction of `lmdpy` is:

> **A Langevin system = Potential + Solvent + Integrator**

- **Potential** → physics of interactions (energy & gradients)
- **Solvent** → environment (friction, temperature, flow)
- **Integrator** → numerical realization of the SDE

---

## 7. The examples folder

The scripts in `examples/` are intentionally physics-first and explicit.

1. **`Langevin_general_1D.py`** — single-particle 1D Langevin, overdamped/underdamped, PDG well
2. **`Langevin_general_3D.py`** — dimension-general, diagnostics (MSD/VACF), multiprocessing ensembles
3. **`Langevin_plottings.py`** — plotting utilities (trajectory, energy, MSD, VACF, distributions)
4. **`pdg_coherent_field.py`** — particles coupled to a dynamical coherent field `a(t)`
5. **`pdg_hydrodynamics.py`** — hydrodynamic mobility (Oseen tensor) + correlated noise
6. **`pdg_ligand_receptor.py`** — ligand–receptor binding in water (overdamped)
7. **`pdg_multi_particle.py`** — interacting many-particle Langevin with repulsion + coherent attraction
8. **`pdg_physical_units.py`** — physical units (nm/ps/u) + Stokes friction + thermal noise
9. **`pdg_quantum_effective.py`** — quantum-inspired effective coordinate model

---

## 8. The `lmdpy` library

### 8.1 `lmdpy.langevin_core`

Implements overdamped and underdamped Langevin integration with selectable schemes
(e.g., Euler–Maruyama, SRK/Heun-like). Standard output is a results dictionary, typically including:

- `t`: time array
- `x`: positions
- `v`: velocities (or `None` in overdamped)
- plus metadata/energies depending on configuration

### 8.2 `lmdpy.potentials`

Reusable potential objects (each provides `U(x)` and `gradU(x)`), including:

- Harmonic
- PDG coherent-domain effective well
- Double-well style models
- Single-center Lennard–Jones
- Single-center Morse

### 8.3 `lmdpy.solvent`

Encodes environmental fields:

- Constant friction / temperature
- Shear-flow solvent
- Fully custom space/time-dependent `gamma(x,t)`, `T(x,t)`, `v_flow(x,t)`

### 8.4 `lmdpy.figures`

Diagnostics and plotting utilities designed to work with the results dict.

### 8.5 `lmdpy.hpc`

Cluster-friendly utilities for:

- ensemble runs (local multiprocessing)
- HTCondor submit-file generation
- parameter scans and reproducible seeding

---

## 9. Diagnostics & figures

Common observables supported by examples and/or `lmdpy.figures`:

- Potential / kinetic / total energy time series
- Mean-squared displacement (MSD)
- Velocity autocorrelation function (VACF)
- Position distributions (1D marginals, radii, etc.)

---

## 10. HPC workflows

### 10.1 Run an ensemble locally (multi-core)

A typical pattern (see `lmdpy.hpc`) is:

- define a simulation function
- pass a parameter list
- run with multiprocessing

### 10.2 HTCondor usage

You can generate submit files (or use templates) and run parameter sweeps.
The intended workflow is:

```bash
# Example (if you have a submit file in condor/submit.sub):
condor_submit condor/submit.sub
```

On HTCondor clusters, prefer:
- small, independent jobs
- unique seeds per job
- write results to per-job output folders

---

## 11. Web UI

`lmdpy/webui/` contains assets for an interactive “Langevin lab” in the browser
(`lmdpy_langevin_lab.js`). This is intended for:

- teaching demonstrations
- interactive parameter exploration
- showing `lmdpy` concepts visually

(If you add a Flask backend, document the run command here.)

---

## 12. Extending lmdpy

### 12.1 Add a new potential

Implement a class with:

- `U(x)`
- `gradU(x)`

### 12.2 Add a new solvent

Implement:

- `gamma(x,t)`
- `temperature(x,t)`
- `flow_velocity(x,t)`

### 12.3 Add a new integrator scheme

Add a new stepping function in `langevin_core.py` and expose it by name via the integrator selector.

---

## 13. Reproducibility

For deterministic replay:

- Fix random seed (NumPy Generator)
- Save parameters alongside output
- Prefer per-job seeds on HPC derived from `(job_id, run_id)`

---

## 14. Notes & caveats

- Some examples implement multi-particle or hydrodynamic interactions directly and are not yet fully unified into the library API.
- PDG-inspired potentials are **effective** / phenomenological models meant for exploration and hypothesis testing.
- For large N, naive O(N²) force evaluation will become expensive; consider neighbor lists, numba, or compiled backends.

---

## 15. License & citation

**License:** (to be added; MIT or BSD-3 recommended)

If you use this library in academic work, cite the repository and document:

- The SDE form (overdamped/underdamped)
- Numerical scheme (Euler/SRK)
- Potential choice
- Solvent choice (friction, temperature, flow)

---

## 16. Contributing (Git workflow)

If you plan to contribute or develop inside your fork, a clean minimal workflow is:

```bash
# 1) create a new branch
git checkout -b feature/my-change

# 2) edit files
# ...

# 3) stage and commit
git add .
git commit -m "Describe the change"

# 4) push branch
git push -u origin feature/my-change
```

If you're updating your local branch after remote changes:

```bash
git pull --rebase
```

---

### Maintainers

Klaudio Peqini\
Department of Physics\
Faculty of Natural Sciences\
University of Tirana
