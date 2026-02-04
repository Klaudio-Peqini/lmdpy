
"""
Simple Flask app exposing a small API around lmdpy.langevin_core.LangevinIntegrator.

This is intentionally lightweight, suitable for classroom demos or running on
a laptop. For production or multi-user setups, integrate this into a proper
WSGI / ASGI deployment.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np

from ..potentials import HarmonicPotential, PDGCoherentPotential, DoubleWell1D
from ..solvent import HomogeneousSolvent
from ..langevin_core import LangevinIntegrator

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json(force=True)

    pot_type = data.get("potential", "pdg").lower()
    dim = int(data.get("dim", 3))
    overdamped = bool(data.get("overdamped", True))
    dt = float(data.get("dt", 1e-3))
    n_steps = int(data.get("n_steps", 5000))

    # Potential parameters
    pot_params = data.get("potential_params", {})
    if pot_type == "harmonic":
        k = pot_params.get("k", 1.0)
        potential = HarmonicPotential(k=k)
    elif pot_type == "double_well":
        a = pot_params.get("a", 1.0)
        b = pot_params.get("b", 1.0)
        potential = DoubleWell1D(a=a, b=b)
    else:
        # default PDG
        k = pot_params.get("k", 1.0)
        A = pot_params.get("A_cd", 1.0)
        xi = pot_params.get("xi", 1.0)
        potential = PDGCoherentPotential(k=k, A_cd=A, xi=xi)

    # Solvent
    gamma = float(data.get("gamma", 1.0))
    T = float(data.get("temperature", 1.0))
    solvent = HomogeneousSolvent(gamma=gamma, temperature=T)

    # Initial conditions
    x0 = np.asarray(data.get("x0", [0.5] + [0.0] * (dim - 1)), dtype=float)
    if len(x0) != dim:
        x0 = np.zeros(dim)
    if overdamped:
        v0 = None
    else:
        v0 = np.asarray(data.get("v0", [0.0] * dim), dtype=float)

    integrator = LangevinIntegrator(
        potential=potential,
        solvent=solvent,
        m=float(data.get("m", 1.0)),
        dim=dim,
        overdamped=overdamped,
        dt=dt,
        integrator=data.get("integrator", "euler"),
    )

    results = integrator.simulate(n_steps=n_steps, x0=x0, v0=v0, store_trajectory=True)

    # Convert to lists for JSON
    out = {
        "t": results["t"].tolist(),
        "x": results["x"].tolist(),
        "U": results["U"].tolist(),
        "K": results["K"].tolist(),
        "E": results["E"].tolist(),
    }
    if results["v"] is not None:
        out["v"] = results["v"].tolist()
    else:
        out["v"] = None

    return jsonify(out)


if __name__ == "__main__":
    app.run(debug=True)
