#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:42:57 2025

@author: klaudio
"""

#!/usr/bin/env python3
"""
Langevin dynamics inspired by a P–DG-like Lagrangian.

We simulate:
    Underdamped:
        m d2x/dt2 = -∇U(x) - γ dx/dt + sqrt(2 γ k_B T) * η(t)

    Overdamped:
        γ dx/dt   = -∇U(x) + sqrt(2 γ k_B T) * η(t)

Potential U(x) is P–DG-inspired:
    U(x) = 0.5 * k_harm * |x|^2  -  A_cd * exp(-|x|^2 / (2 ξ^2))

- First term: internal harmonic mode (molecular elastic degree of freedom).
- Second term: coherent-domain attraction (low-energy basin),
  mimicking field-matter coherence (à la Preparata–Del Giudice).

Integrator options:
    - "euler": Euler–Maruyama
    - "srk":   stochastic Heun / stochastic Runge–Kutta (weak order 2)

Supports:
    - 1D or higher dimension (dim parameter)
    - Parallel ensembles via multiprocessing (see __main__ example)
"""

import numpy as np
import multiprocessing as mp
import os


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
            Number of independent particles simulated in parallel.
        m : float
            Mass of each particle.
        gamma : float
            Friction coefficient.
        kT : float
            Thermal energy k_B T (set k_B=1 in units if desired).
        k_harm : float
            Harmonic stiffness of the internal elastic mode.
        A_cd : float
            Amplitude of coherent-domain potential well.
        xi : float
            Coherent-domain length scale (controls width of the well).
        dt : float
            Time step.
        n_steps : int
            Number of integration steps.
        overdamped : bool
            If True, uses the overdamped (Brownian) limit.
        integrator : {"euler", "srk"}
            Euler–Maruyama or stochastic Runge–Kutta (Heun).
        store_trajectory : bool
            If True, store full trajectory (positions and velocities).
        seed : int or None
            Random seed for reproducibility.
        """
        self.dim = dim
        self.n_particles = n_particles
        self.m = m
        self.gamma = gamma
        self.kT = kT
        self.k_harm = k_harm
        self.A_cd = A_cd
        self.xi = xi
        self.dt = dt
        self.n_steps = n_steps
        self.overdamped = overdamped
        self.integrator = integrator.lower()
        self.store_trajectory = store_trajectory

        self.rng = np.random.default_rng(seed)

        # Noise coefficients (scalar; applied componentwise)
        if overdamped:
            # overdamped: dx = (-∇U/γ) dt + sqrt(2 kT / γ) dW
            self.sigma_pos = np.sqrt(2.0 * self.kT / self.gamma)
            # no velocity state
        else:
            # underdamped: dv = [(-∇U - γ v)/m] dt + sqrt(2 γ kT / m^2) dW
            self.sigma_vel = np.sqrt(2.0 * self.gamma * self.kT / (self.m**2))

    # ---------- P–DG-inspired potential and gradient ----------

    def potential(self, x):
        """
        P–DG-inspired potential:
            U(x) = 0.5 k_harm |x|^2 - A_cd exp(-|x|^2 / (2 xi^2))

        x : array, shape (..., dim)
        returns: array, shape (...)
        """
        r2 = np.sum(x**2, axis=-1)
        U_harm = 0.5 * self.k_harm * r2
        U_cd = -self.A_cd * np.exp(-r2 / (2.0 * self.xi**2))
        return U_harm + U_cd

    def grad_potential(self, x):
        """
        Gradient ∇U(x) for the above radial potential.

        For a radial U(r^2) with r^2 = |x|^2:
            U(r^2) = 0.5 k r^2 - A exp(-r^2 / (2 xi^2))
        dU/dx = (dU/dr^2) * 2x
              = [0.5 k - A * (-1/(2 xi^2)) exp(-r^2/(2 xi^2))] * 2x
        """
        r2 = np.sum(x**2, axis=-1, keepdims=True)
        exp_term = np.exp(-r2 / (2.0 * self.xi**2))
        # dU/dr2
        dU_dr2 = 0.5 * self.k_harm - self.A_cd * (-1.0 / (2.0 * self.xi**2)) * exp_term
        # ∇U = dU/dr2 * 2x
        grad = dU_dr2 * 2.0 * x
        return grad

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
        Underdamped Euler–Maruyama for:
            dx = v dt
            dv = [(-∇U(x) - γ v)/m] dt + sigma_vel dW
        """
        dt = self.dt
        dW = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=v.shape)
        gradU = self.grad_potential(x)
        accel = (-gradU - self.gamma * v) / self.m  # deterministic dv/dt
        v_new = v + accel * dt + self.sigma_vel * dW
        x_new = x + v_new * dt  # or x + v dt (semi-implicit vs explicit)
        return x_new, v_new

    def _step_underdamped_srk(self, x, v):
        """
        Underdamped stochastic Heun (weak order 2):

        State Y = (x, v).
        SDE:
            dx = v dt
            dv = a(x, v) dt + sigma_vel dW, with a = (-∇U - γ v)/m

        Predictor:
            x_tilde = x + v dt
            v_tilde = v + a(x,v) dt + sigma_vel dW

        Corrector:
            a_t = a(x_tilde, v_tilde)
            x_new = x + 0.5 (v + v_tilde) dt
            v_new = v + 0.5 (a(x,v) + a_t) dt + sigma_vel dW
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
        Run a single trajectory (possibly many particles in parallel).

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
                "k_harm": self.k_harm,
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

        Returns
        -------
        t : array, shape (T,)
        msd : array, shape (T,)
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

        For overdamped case: kinetic energy is not meaningful at the SDE level;
        we only compute potential energy.
        """
        x_traj = results["x"]  # (T, N, dim)
        t = results["t"]

        x = x_traj[:, particle_index, :]
        U = self.potential(x)  # (T,)

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

        Returns
        -------
        t_lag : array, shape (T,)
        Cvv : array, shape (T,)
        """
        v_traj = results["v"]
        if v_traj is None:
            raise ValueError("Velocity data not available (overdamped or not stored).")
        v = v_traj[:, particle_index, :]  # (T, dim)
        T = v.shape[0]
        dim = v.shape[1]

        # Subtract mean velocity (should be ~0 but just in case)
        v = v - np.mean(v, axis=0, keepdims=True)

        # Compute autocorrelation via direct (O(T^2)) method (ok for moderate T)
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
        Compute histogram (distribution) of positions at late times.

        Returns
        -------
        hist : array
        edges : array
        """
        x_traj = results["x"]  # (T, N, dim)
        x = x_traj[:, particle_index, :]
        # Flatten over time and dimension if 1D; for dim>1 you may want per-component
        data = x.reshape(-1, x.shape[-1])
        if data.shape[1] == 1:
            # 1D
            data1d = data[:, 0]
        else:
            # magnitude of position
            data1d = np.linalg.norm(data, axis=1)
        hist, edges = np.histogram(data1d, bins=bins, density=True)
        return hist, edges


# ---------- Parallel ensemble runner ----------

def _run_single_trajectory(args):
    """
    Helper for multiprocessing: run one trajectory with given seed.
    """
    idx, base_params = args
    params = base_params.copy()
    params["seed"] = None if base_params.get("seed") is None else base_params["seed"] + idx

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
    (ideal for a 32-thread laptop).

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


# ---------- Example usage ----------

if __name__ == "__main__":
    # Example: single underdamped trajectory, SRK integrator, P–DG-inspired potential
    sim = LangevinPDG(
        dim=1,
        n_particles=1,
        m=1.0,
        gamma=1.0,
        kT=0.5,
        k_harm=1.0,
        A_cd=1.0,
        xi=1.0,
        dt=1e-3,
        n_steps=50000,
        overdamped=False,
        integrator="srk",
        store_trajectory=True,
        seed=42,
    )

    results = sim.simulate()

    # Basic diagnostics: MSD, energies, VACF
    t_msd, msd = LangevinPDG.compute_msd(results, particle_index=0)
    t_E, K, U, E = sim.compute_energies(results, particle_index=0)

    print("Final MSD:", msd[-1])
    print("Average potential energy:", np.mean(U[int(0.5 * len(U)) :]))
    print("Average total energy:", np.mean(E[int(0.5 * len(E)) :]))

    if not sim.overdamped:
        t_lag, Cvv = LangevinPDG.compute_velocity_autocorrelation(results, particle_index=0)
        print("Initial Cvv:", Cvv[0])

    # Example: run an ensemble in parallel (e.g., 32 trajectories on up to 32 threads)
    # ensemble_results = run_ensemble_parallel(
    #     n_trajectories=32,
    #     dim=1,
    #     n_particles=1,
    #     m=1.0,
    #     gamma=1.0,
    #     kT=0.5,
    #     k_harm=1.0,
    #     A_cd=1.0,
    #     xi=1.0,
    #     dt=1e-3,
    #     n_steps=20000,
    #     overdamped=True,
    #     integrator="euler",
    #     store_trajectory=False,
    #     base_seed=2025,
    #     max_workers=32,
    # )
    # print("Ran ensemble of", len(ensemble_results), "trajectories in parallel.")
