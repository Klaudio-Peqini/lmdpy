
"""
lmdpy.figures

Helper functions to produce "publication-ready" figures from Langevin
simulation results (as produced by lmdpy.langevin_core.LangevinIntegrator).

These functions assume a result dict with keys:
    - "t": shape (N,)
    - "x": shape (N, dim)
    - "v": shape (N, dim) or None
    - "U", "K", "E": shape (N,)

They configure Matplotlib with reasonably clean defaults, but you can further
customize fonts, sizes, etc., in your script.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_components_vs_time(results, mode="position", ax=None, labels=None):
    """
    Plot x_i(t) or v_i(t) for i=0..dim-1 in a single figure.

    Parameters
    ----------
    results : dict
    mode : {"position", "velocity"}
        Which to plot.
    ax : matplotlib Axes, optional
        If None, create a new figure.
    labels : list of str or None
        Labels for components. If None, use "x0", "x1", ...
    """
    t = results["t"]
    x = results["x"]
    v = results.get("v", None)

    if mode == "position":
        data = x
        base_label = "x"
    else:
        if v is None:
            raise ValueError("Velocity data not available in results.")
        data = v
        base_label = "v"

    dim = data.shape[1]
    if labels is None:
        labels = [f"{base_label}{i}" for i in range(dim)]

    if ax is None:
        fig, ax = plt.subplots()

    for i in range(dim):
        ax.plot(t, data[:, i], label=labels[i])

    ax.set_xlabel("t")
    ax.set_ylabel(mode)
    ax.legend()
    ax.set_title(f"{mode.capitalize()} components vs time")
    return ax


def plot_energy(results, ax=None):
    """
    Plot U(t), K(t), E(t).
    """
    t = results["t"]
    U = results["U"]
    K = results["K"]
    E = results["E"]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(t, U, label="U")
    ax.plot(t, K, label="K")
    ax.plot(t, E, label="E")
    ax.set_xlabel("t")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.set_title("Energy vs time")
    return ax


def plot_position_distribution(results, component=0, bins=50, ax=None):
    """
    Histogram of one coordinate x_component.

    Parameters
    ----------
    component : int
        Index of the coordinate (0 for x, 1 for y, ...)
    """
    x = results["x"]
    data = x[:, component]

    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(data, bins=bins, density=True)
    ax.set_xlabel(f"x[{component}]")
    ax.set_ylabel("P(x)")
    ax.set_title("Position distribution")
    return ax


def compute_msd(results):
    """
    Compute mean-squared displacement MSD(t) from the origin x(0).

    Returns
    -------
    t, msd
    """
    t = results["t"]
    x = results["x"]
    x0 = x[0]
    dr = x - x0
    msd = np.sum(dr * dr, axis=1)
    return t, msd


def compute_vacf(results):
    """
    Compute a simple velocity autocorrelation function C_vv(t):

        C(t) = <v(0) · v(t)> / <|v(0)|^2>

    Returns
    -------
    t, C
    """
    v = results.get("v", None)
    if v is None:
        raise ValueError("Velocity data not available in results.")

    t = results["t"]
    v0 = v[0]
    v0_norm2 = np.sum(v0 * v0)
    if v0_norm2 == 0:
        C = np.zeros_like(t)
        return t, C

    C = np.sum(v * v0, axis=1) / v0_norm2
    return t, C


def plot_msd_vacf(results, ax=None):
    """
    Plot MSD(t) and VACF(t) in a single figure with twin y-axes.

    Left y-axis: MSD
    Right y-axis: VACF
    """
    t, msd = compute_msd(results)
    try:
        t_v, C = compute_vacf(results)
    except ValueError:
        t_v = t
        C = np.zeros_like(t)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(t, msd, label="MSD")
    ax.set_xlabel("t")
    ax.set_ylabel("MSD")

    ax2 = ax.twinx()
    ax2.plot(t_v, C, linestyle="--", label="VACF")

    ax2.set_ylabel("VACF")
    ax.set_title("MSD and VACF")

    # Build a combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    return ax, ax2
