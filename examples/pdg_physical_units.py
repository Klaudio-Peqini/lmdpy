
#!/usr/bin/env python3
"""pdg_physical_units.py

Langevin dynamics for a single solute in water, using physical units:

- Position in nm
- Time in ps
- Mass in atomic mass units (u)
- k_B T at 300 K

We simulate:

    m d2x/dt2 = -∇U(x) - γ dx/dt + sqrt(2 γ k_B T) η(t),

with Stokes friction γ = 6π η a and PDG-inspired potential.

Usage:
    python3 pdg_physical_units.py
"""

import numpy as np
import matplotlib.pyplot as plt


class PDGPhysicalUnits:
    def __init__(
        self,
        dim=3,
        mass_u=10.0,
        radius_nm=1.0,
        temperature_K=300.0,
        viscosity_mPa_s=0.89,
        k_harm_kBT=5.0,
        A_cd_kBT=2.0,
        xi_nm=2.0,
        dt_ps=0.001,
        n_steps=200000,
        seed=None,
    ):
        self.dim = dim
        self.mass_u = mass_u
        self.radius_nm = radius_nm
        self.T = temperature_K
        self.eta = viscosity_mPa_s * 1e-3  # mPa*s -> Pa*s

        # physical constants
        self.kB = 1.380649e-23  # J/K
        self.u_kg = 1.66053906660e-27  # kg

        # unit conversions
        self.nm = 1e-9
        self.ps = 1e-12

        self.m_kg = mass_u * self.u_kg
        self.a_m = radius_nm * self.nm
        self.kT_J = self.kB * self.T

        # Stokes friction
        self.gamma = 6 * np.pi * self.eta * self.a_m  # kg/s

        # Potential parameters in SI
        self.k_harm = k_harm_kBT * self.kT_J / (self.nm**2)
        self.A_cd = A_cd_kBT * self.kT_J
        self.xi = xi_nm * self.nm

        self.dt = dt_ps * self.ps
        self.n_steps = n_steps

        self.rng = np.random.default_rng(seed)
        self.sigma_vel = np.sqrt(2.0 * self.gamma * self.kT_J / (self.m_kg**2))

    def potential(self, x):
        """x in meters."""
        r2 = np.sum(x**2, axis=-1)
        U_h = 0.5 * self.k_harm * r2
        U_cd = -self.A_cd * np.exp(-r2 / (2 * self.xi**2))
        return U_h + U_cd

    def grad_potential(self, x):
        r2 = np.sum(x**2, axis=-1, keepdims=True)
        exp_term = np.exp(-r2 / (2 * self.xi**2))
        dU_dr2 = 0.5 * self.k_harm - self.A_cd * (-1.0 / (2 * self.xi**2)) * exp_term
        grad = dU_dr2 * 2.0 * x
        return grad

    def simulate(self, x0_nm=None, v0=None):
        d = self.dim
        if x0_nm is None:
            x = np.zeros((1, d))
        else:
            x = np.array(x0_nm, float).reshape(1, d) * self.nm
        if v0 is None:
            v = np.zeros((1, d))
        else:
            v = np.array(v0, float).reshape(1, d)

        traj = np.zeros((self.n_steps + 1, d))
        traj[0] = x[0]
        dt = self.dt

        for n in range(1, self.n_steps + 1):
            dW = self.rng.normal(0.0, np.sqrt(dt), size=v.shape)
            F = -self.grad_potential(x) - self.gamma * v
            a = F / self.m_kg
            v = v + a * dt + self.sigma_vel * dW
            x = x + v * dt
            traj[n] = x[0]

        return traj

    def plot_r(self, traj):
        t_ps = np.arange(traj.shape[0]) * self.dt / self.ps
        r_nm = np.linalg.norm(traj, axis=1) / self.nm
        plt.figure()
        plt.plot(t_ps, r_nm)
        plt.xlabel("t [ps]")
        plt.ylabel("|x| [nm]")
        plt.title("Radial position of solute in PDG potential")
        plt.tight_layout()
        plt.show()


def main():
    sim = PDGPhysicalUnits(
        dim=3,
        mass_u=20.0,
        radius_nm=1.5,
        k_harm_kBT=5.0,
        A_cd_kBT=3.0,
        xi_nm=2.0,
        dt_ps=0.001,
        n_steps=100000,
        seed=42,
    )
    traj = sim.simulate()
    sim.plot_r(traj)


if __name__ == "__main__":
    main()
