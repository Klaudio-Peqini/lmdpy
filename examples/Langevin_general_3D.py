#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:02:37 2025

@author: klaudio
"""

#!/usr/bin/env python3
"""
Langevin dynamics inspired by a P–DG-like Lagrangian.

Underdamped:
    m d2x/dt2 = -∇U(x) - γ dx/dt + sqrt(2 γ k_B T) * η(t)

Overdamped:
    γ dx/dt   = -∇U(x) + sqrt(2 γ k_B T) * η(t)

P–DG-inspired potential (general in dim = 1,2,3,...):

    U(x) = 0.5 * sum_i k_i x_i^2  -  A_cd * exp(-|x|^2 / (2 ξ^2))

- First term: anisotropic harmonic mode (elastic intramolecular degrees of freedom).
- Second term: coherent-domain attraction (low-energy basin),
  mimicking field–matter coherence à la Preparata–Del Giudice.

Integrator options:
    - "euler": Euler–Maruyama
    - "srk":   stochastic Heun / stochastic Runge–Kutta (weak order 2)

Parallelization:
    - run_ensemble_parallel(...) uses multiprocessing up to max_workers (e.g. 32 cores).
"""

import numpy as np
import multiprocessing as mp
import os
import matplotlib.pyplot as plt
import time

start_time = time.time()

class LangevinPDG:
    def __init__(
        self,
        dim=1,
        n_particles=1,
        m=1.0,
        gamma=1.0,
        kT=1.0,
        k_harm=1.0,
        A_cd=0.5,
        xi=1.0,
        dt=1e-3,
        n_steps=10000,
        overdamped=False,
        integrator="euler",
        store_trajectory=True,
        seed=None,
    ):
        """
        Parameters
        ----------
        dim : int
            Spatial dimension (1, 2, 3, ...).
        n_particles : int
            Number of independent particles simulated in parallel (vectorized).
        m : float
            Mass of each particle.
        gamma : float
            Friction coefficient.
        kT : float
            Thermal energy k_B T (you can set k_B = 1 in reduced units).
        k_harm : float or array-like of length dim
            Harmonic stiffness:
                - scalar -> same k in all directions
                - array  -> k_i per coordinate (anisotropic oscillator).
        A_cd : float
            Amplitude of coherent-domain potential well.
        xi : float
            Coherent-domain length scale (controls width of the well).
        dt : float
            Time step.
        n_steps : int
            Number of integration steps.
        overdamped : bool
            If True, use overdamped (Brownian) limit.
        integrator : {"euler", "srk"}
            "euler": Euler–Maruyama; "srk": stochastic Heun / RK2.
        store_trajectory : bool
            If True, store full trajectory (positions and velocities if underdamped).
        seed : int or None
            Random seed for reproducibility.
        """
        self.dim = dim
        self.n_particles = n_particles
        self.m = m
        self.gamma = gamma
        self.kT = kT

        # Generalize harmonic stiffness to dim-D (anisotropic allowed)
        k_harm_arr = np.array(k_harm, dtype=float)
        if k_harm_arr.ndim == 0:
            # scalar -> same in each direction
            self.k_harm_vec = np.full(dim, k_harm_arr)
        else:
            if k_harm_arr.size != dim:
                raise ValueError(
                    f"k_harm must be scalar or length {dim}, "
                    f"but got shape {k_harm_arr.shape}"
                )
            self.k_harm_vec = k_harm_arr

        self.A_cd = A_cd
        self.xi = float(xi)
        self.dt = dt
        self.n_steps = n_steps
        self.overdamped = overdamped
        self.integrator = integrator.lower()
        self.store_trajectory = store_trajectory

        self.rng = np.random.default_rng(seed)

        # Noise coefficients
        if overdamped:
            # Overdamped: dx = (-∇U/γ) dt + sqrt(2 kT / γ) dW
            self.sigma_pos = np.sqrt(2.0 * self.kT / self.gamma)
        else:
            # Underdamped: dv = [(-∇U - γ v)/m] dt + sqrt(2 γ kT / m^2) dW
            self.sigma_vel = np.sqrt(2.0 * self.gamma * self.kT / (self.m**2))

    # ---------- P–DG-inspired potential and gradient (general dim) ----------

    def potential(self, x):
        """
        P–DG-inspired potential, valid in 1D, 2D, 3D, ...

        U(x) = 0.5 * sum_i k_i x_i^2  -  A_cd * exp(-|x|^2 / (2 xi^2))

        x : array, shape (..., dim)
        returns : array, shape (...)
        """
        # Harmonic part, anisotropic in dim
        # (..., dim) * (dim,) -> (..., dim), then sum over last axis
        U_harm = 0.5 * np.sum(self.k_harm_vec * x**2, axis=-1)

        # P–DG coherent-domain term (radial, same in any dim)
        r2 = np.sum(x**2, axis=-1)
        U_cd = -self.A_cd * np.exp(-r2 / (2.0 * self.xi**2))
        return U_harm + U_cd

    def grad_potential(self, x):
        """
        Gradient ∇U(x) = ∇U_harm(x) + ∇U_cd(x).

        U_harm = 0.5 * sum_i k_i x_i^2  ->  ∂U_harm/∂x_i = k_i x_i
        U_cd   = -A_cd exp(-|x|^2 / (2 xi^2))
              -> ∂U_cd/∂x_i = (A_cd/xi^2) exp(-|x|^2 / (2 xi^2)) x_i
        """
        # Harmonic gradient
        grad_harm = self.k_harm_vec * x  # broadcast (dim,) over (..., dim)

        # Coherent-domain gradient
        r2 = np.sum(x**2, axis=-1, keepdims=True)  # shape (..., 1)
        factor = (self.A_cd / (self.xi**2)) * np.exp(-r2 / (2.0 * self.xi**2))
        grad_cd = factor * x  # same shape as x

        return grad_harm + grad_cd

    # ---------- Integration schemes ----------

    def _step_overdamped_euler(self, x):
        """
        Overdamped Euler–Maruyama:
            dx = (-∇U/γ) dt + sqrt(2 kT / γ) dW
        """
        dt = self.dt
        drift = -self.grad_potential(x) / self.gamma  # shape (N, dim)
        dW = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=x.shape)
        x_new = x + drift * dt + self.sigma_pos * dW
        return x_new

    def _step_overdamped_srk(self, x):
        """
        Overdamped stochastic Heun (weak order 2):
            Predictor: x_tilde = x + a(x) dt + b dW
            Corrector: x_{n+1} = x + 0.5 [a(x) + a(x_tilde)] dt + b dW
        with a(x) = -∇U/γ, b = sqrt(2 kT/γ) (constant).
        """
        dt = self.dt
        dW = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=x.shape)
        a_x = -self.grad_potential(x) / self.gamma
        x_tilde = x + a_x * dt + self.sigma_pos * dW
        a_xt = -self.grad_potential(x_tilde) / self.gamma
        x_new = x + 0.5 * (a_x + a_xt) * dt + self.sigma_pos * dW
        return x_new

    def _step_underdamped_euler(self, x, v):
        """
        Underdamped Euler–Maruyama:
            dx = v dt
            dv = [(-∇U(x) - γ v)/m] dt + sigma_vel dW
        """
        dt = self.dt
        dW = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=v.shape)
        gradU = self.grad_potential(x)
        accel = (-gradU - self.gamma * v) / self.m
        v_new = v + accel * dt + self.sigma_vel * dW
        x_new = x + v_new * dt  # semi-implicit
        return x_new, v_new

    def _step_underdamped_srk(self, x, v):
        """
        Underdamped stochastic Heun (weak order 2):

        dx = v dt
        dv = a(x, v) dt + sigma_vel dW, with a = (-∇U - γ v)/m
        """
        dt = self.dt
        dW = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=v.shape)

        gradU = self.grad_potential(x)
        a_xv = (-gradU - self.gamma * v) / self.m

        # Predictor
        x_tilde = x + v * dt
        v_tilde = v + a_xv * dt + self.sigma_vel * dW

        # Corrector
        gradU_t = self.grad_potential(x_tilde)
        a_xt_vt = (-gradU_t - self.gamma * v_tilde) / self.m

        x_new = x + 0.5 * (v + v_tilde) * dt
        v_new = v + 0.5 * (a_xv + a_xt_vt) * dt + self.sigma_vel * dW
        return x_new, v_new

    # ---------- Simulation ----------

    def simulate(self, x0=None, v0=None):
        """
        Run a single trajectory (possibly many particles vectorized).

        Returns
        -------
        results : dict
            {
              "t": times, shape (n_steps+1,),
              "x": positions, shape (n_steps+1, n_particles, dim),
              "v": velocities (underdamped only),
              "params": dict of parameters
            }
        """
        dt = self.dt
        n_steps = self.n_steps
        N = self.n_particles
        dim = self.dim

        # Initial conditions
        if x0 is None:
            x = np.zeros((N, dim), dtype=float)
        else:
            x = np.array(x0, dtype=float).reshape(N, dim)

        if self.overdamped:
            v = None
        else:
            if v0 is None:
                v = np.zeros((N, dim), dtype=float)
            else:
                v = np.array(v0, dtype=float).reshape(N, dim)

        # Allocate trajectory arrays if requested
        if self.store_trajectory:
            x_traj = np.zeros((n_steps + 1, N, dim), dtype=float)
            x_traj[0] = x
            if not self.overdamped:
                v_traj = np.zeros((n_steps + 1, N, dim), dtype=float)
                v_traj[0] = v
            else:
                v_traj = None
        else:
            x_traj = v_traj = None

        t = np.linspace(0.0, n_steps * dt, n_steps + 1)

        for n in range(1, n_steps + 1):
            if self.overdamped:
                if self.integrator == "srk":
                    x = self._step_overdamped_srk(x)
                else:
                    x = self._step_overdamped_euler(x)
            else:
                if self.integrator == "srk":
                    x, v = self._step_underdamped_srk(x, v)
                else:
                    x, v = self._step_underdamped_euler(x, v)

            if self.store_trajectory:
                x_traj[n] = x
                if v_traj is not None:
                    v_traj[n] = v

        results = {
            "t": t,
            "x": x_traj if self.store_trajectory else x,
            "v": v_traj if (self.store_trajectory and not self.overdamped) else v,
            "params": {
                "dim": dim,
                "n_particles": N,
                "m": self.m,
                "gamma": self.gamma,
                "kT": self.kT,
                "k_harm_vec": self.k_harm_vec,
                "A_cd": self.A_cd,
                "xi": self.xi,
                "dt": dt,
                "n_steps": n_steps,
                "overdamped": self.overdamped,
                "integrator": self.integrator,
            },
        }
        return results

    # ---------- Analysis methods ----------

    @staticmethod
    def compute_msd(results, particle_index=0):
        """
        Mean-squared displacement for a given particle index.
        """
        x_traj = results["x"]  # (T, N, dim)
        t = results["t"]
        x0 = x_traj[0, particle_index]
        dx = x_traj[:, particle_index] - x0
        msd = np.mean(dx**2, axis=-1)  # sum over dimensions
        return t, msd

    def compute_energies(self, results, particle_index=0):
        """
        Compute kinetic, potential, and total energy (single particle index).
        """
        x_traj = results["x"]
        t = results["t"]

        x = x_traj[:, particle_index, :]
        U = self.potential(x)

        if self.overdamped or results["v"] is None:
            K = np.zeros_like(U)
        else:
            v_traj = results["v"]
            v = v_traj[:, particle_index, :]
            v2 = np.sum(v**2, axis=-1)
            K = 0.5 * self.m * v2

        E = K + U
        return t, K, U, E

    @staticmethod
    def compute_velocity_autocorrelation(results, particle_index=0):
        """
        Velocity autocorrelation function C_vv(τ) for a single particle.
        """
        v_traj = results["v"]
        if v_traj is None:
            raise ValueError("Velocity data not available (overdamped or not stored).")
        v = v_traj[:, particle_index, :]  # (T, dim)
        T = v.shape[0]
        dim = v.shape[1]

        v = v - np.mean(v, axis=0, keepdims=True)

        Cvv = np.zeros(T)
        for tau in range(T):
            prod = v[: T - tau] * v[tau:]
            Cvv[tau] = np.sum(prod) / (T - tau) / dim

        t = results["t"]
        t_lag = t - t[0]
        return t_lag, Cvv

    @staticmethod
    def position_distribution(results, particle_index=0, bins=50):
        """
        Histogram (distribution) of positions at late times.
        - For dim=1: histogram of x.
        - For dim>1: histogram of |x| (radial).
        """
        x_traj = results["x"]
        x = x_traj[:, particle_index, :]
        data = x.reshape(-1, x.shape[-1])
        if data.shape[1] == 1:
            data1d = data[:, 0]
        else:
            data1d = np.linalg.norm(data, axis=1)
        hist, edges = np.histogram(data1d, bins=bins, density=True)
        return hist, edges

    # ---------- Plotting helpers ----------

    def plot_trajectory(self, results, particle_index=0):
        """
        Plot trajectory for one particle.
        - 1D: x(t)
        - 2D: x-y path
        - 3D+: components vs t
        """
        x_traj = results["x"]
        t = results["t"]
        x = x_traj[:, particle_index, :]
        dim = x.shape[1]

        if dim == 1:
            plt.figure()
            plt.plot(t, x[:, 0])
            plt.xlabel("t")
            plt.ylabel("x")
            plt.title("Trajectory x(t)")
            plt.tight_layout()
            plt.savefig("Figs/1_components_1D.pdf")
            plt.close()
        elif dim == 2:
            plt.figure()
            plt.plot(x[:, 0], x[:, 1], linewidth=0.8)
            plt.scatter(x[0, 0], x[0, 1], marker="o", label="start")
            plt.scatter(x[-1, 0], x[-1, 1], marker="x", label="end")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("2D trajectory")
            plt.axis("equal")
            plt.legend()
            plt.tight_layout()
            plt.savefig("Figs/1_trajectory_2D.pdf")
            plt.close()
        else:
            plt.figure()
            for d in range(dim):
                plt.plot(t, x[:, d], label=f"x_{d}(t)")
            plt.xlabel("t")
            plt.ylabel("coordinate")
            plt.title("Trajectory components vs time")
            plt.legend()
            plt.tight_layout()
            plt.savefig("Figs/1_1_components_3D.pdf")
            plt.close()
            
            plt.figure().add_subplot(projection='3d')
            plt.plot(x[:,0], x[:,1], x[:,2], 'k')
            plt.savefig("Figs/1_2_trajectory_3D.pdf")
            plt.close()
            

    def plot_msd(self, results, particle_index=0):
        t, msd = self.compute_msd(results, particle_index)
        plt.figure()
        plt.plot(t, msd)
        plt.xlabel("t")
        plt.ylabel("MSD")
        plt.title("Mean-squared displacement")
        plt.tight_layout()
        plt.savefig("Figs/2_mean_square_displacement.pdf")
        plt.close()

    def plot_energies(self, results, particle_index=0):
        t, K, U, E = self.compute_energies(results, particle_index)
        plt.figure()
        plt.plot(t, U, label="U(t)")
        if not self.overdamped:
            plt.plot(t, K, label="K(t)")
        plt.plot(t, E, label="E(t)")
        plt.xlabel("t")
        plt.ylabel("Energy")
        plt.title("Energies vs time")
        plt.legend()
        plt.tight_layout()
        plt.savefig("Figs/3_energies.pdf")
        plt.close()

    def plot_vacf(self, results, particle_index=0):
        if self.overdamped or results["v"] is None:
            raise ValueError("VACF requires underdamped simulation with velocities stored.")
        t_lag, Cvv = self.compute_velocity_autocorrelation(results, particle_index)
        plt.figure()
        plt.plot(t_lag, Cvv)
        plt.xlabel("lag time")
        plt.ylabel("C_vv(lag)")
        plt.title("Velocity autocorrelation function")
        plt.tight_layout()
        plt.savefig("Figs/4_velocity_autocorrelation.pdf")
        plt.close()

    def plot_position_distribution(self, results, particle_index=0, bins=50):
        hist, edges = self.position_distribution(results, particle_index, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.figure()
        plt.bar(centers, hist, width=centers[1] - centers[0])
        plt.xlabel("x (1D) or |x| (dim>1)")
        plt.ylabel("Probability density")
        plt.title("Position distribution")
        plt.tight_layout()
        plt.savefig("Figs/5_position_distirbution.pdf")
        plt.close()


# ---------- Parallel ensemble runner ----------

def _run_single_trajectory(args):
    """
    Helper for multiprocessing: run one trajectory with given seed offset.
    """
    idx, base_params = args
    params = base_params.copy()
    if base_params.get("seed") is not None:
        params["seed"] = base_params["seed"] + idx
    else:
        params["seed"] = None

    sim = LangevinPDG(
        dim=params["dim"],
        n_particles=params["n_particles"],
        m=params["m"],
        gamma=params["gamma"],
        kT=params["kT"],
        k_harm=params["k_harm"],
        A_cd=params["A_cd"],
        xi=params["xi"],
        dt=params["dt"],
        n_steps=params["n_steps"],
        overdamped=params["overdamped"],
        integrator=params["integrator"],
        store_trajectory=params["store_trajectory"],
        seed=params["seed"],
    )
    return sim.simulate()


def run_ensemble_parallel(
    n_trajectories=32,
    dim=1,
    n_particles=1,
    m=1.0,
    gamma=1.0,
    kT=1.0,
    k_harm=1.0,
    A_cd=0.5,
    xi=1.0,
    dt=1e-3,
    n_steps=10000,
    overdamped=False,
    integrator="euler",
    store_trajectory=True,
    base_seed=1234,
    max_workers=32,
):
    """
    Run an ensemble of independent trajectories in parallel
    (suitable for up to ~32-core laptop/desktop).

    Returns
    -------
    list_of_results : list of dict
        Each element is the output of LangevinPDG.simulate().
    """
    base_params = dict(
        dim=dim,
        n_particles=n_particles,
        m=m,
        gamma=gamma,
        kT=kT,
        k_harm=k_harm,
        A_cd=A_cd,
        xi=xi,
        dt=dt,
        n_steps=n_steps,
        overdamped=overdamped,
        integrator=integrator,
        store_trajectory=store_trajectory,
        seed=base_seed,
    )

    n_workers = min(max_workers, os.cpu_count() or 1)

    with mp.Pool(processes=n_workers) as pool:
        args_list = [(i, base_params) for i in range(n_trajectories)]
        results = pool.map(_run_single_trajectory, args_list)

    return results


# ---------- Example usage / plotting ----------

if __name__ == "__main__":
    # # === Single underdamped 2D trajectory with SRK integrator, P–DG potential ===
    # sim = LangevinPDG(
    #     dim=3,
    #     n_particles=1,
    #     m=1.0,
    #     gamma=1.0,
    #     kT=0.5,
    #     # anisotropic harmonic: kx = 1.0, ky = 0.5
    #     k_harm=[1.0, 0.5, 1.5],
    #     A_cd=1.0,
    #     xi=1.0,
    #     dt=1e-3,
    #     n_steps=500000,
    #     overdamped=False,
    #     integrator="srk",
    #     store_trajectory=True,
    #     seed=42,
    # )

    # results = sim.simulate()

    # # Diagnostics
    # t_msd, msd = LangevinPDG.compute_msd(results, particle_index=0)
    # t_E, K, U, E = sim.compute_energies(results, particle_index=0)
    # print("Final MSD:", msd[-1])
    # print("Average potential energy (second half):", np.mean(U[int(0.5 * len(U)) :]))
    # print("Average total energy (second half):", np.mean(E[int(0.5 * len(E)) :]))

    # # Plots
    # sim.plot_trajectory(results, particle_index=0)
    # sim.plot_msd(results, particle_index=0)
    # sim.plot_energies(results, particle_index=0)
    # if not sim.overdamped:
    #     sim.plot_vacf(results, particle_index=0)
    # sim.plot_position_distribution(results, particle_index=0, bins=50)

    # === EXAMPLE: parallel ensemble on up to 32 threads (overdamped 3D case) ===
    # Uncomment to run:
    
    ensemble_results = run_ensemble_parallel(
        n_trajectories=16,   # number of independent runs
        dim=3,               # 3D motion inside a PDG-inspired well
        n_particles=1,
        m=1.0,
        gamma=2.0,
        kT=0.5,
        k_harm=[1.0, 1.0, 0.8],  # e.g. slightly softer in z
        A_cd=1.5,
        xi=1.0,
        dt=5e-4,
        n_steps=500000,
        overdamped=True,     # overdamped regime
        integrator="srk",
        store_trajectory=True,  # only final states if you want to save memory
        base_seed=2025,
        max_workers=32,      # will be capped by os.cpu_count()
    )
    print("Ran ensemble of", len(ensemble_results), "trajectories in parallel.")

    # Use a template for plotting/energy calculations
    sim_template = LangevinPDG(
        dim=3,
        n_particles=1,
        m=1.0,
        gamma=2.0,
        kT=0.5,
        k_harm=[1.0, 1.0, 0.8],
        A_cd=1.0,
        xi=1.0,
        dt=1e-3,
        n_steps=500000,
        overdamped=True,
        integrator="srk",
        store_trajectory=True,
        seed=2025
    )
    
    # Diagnostics
    for i in range(16):
        # i = 5 # particle label
        sim_template.plot_trajectory(ensemble_results[i], particle_index=0)
        sim_template.plot_msd(ensemble_results[i], particle_index=0)
        sim_template.plot_energies(ensemble_results[i], particle_index=0)
        if not sim_template.overdamped:
            sim_template.plot_vacf(ensemble_results[i], particle_index=0)
        sim_template.plot_position_distribution(ensemble_results[i], particle_index=0, bins=50)

print(r"--- %s seconds ---" % (time.time() - start_time))
