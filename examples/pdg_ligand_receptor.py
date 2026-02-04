
#!/usr/bin/env python3
"""pdg_ligand_receptor.py

Ligand–receptor binding model in water, using P–DG-inspired potentials.

Overdamped Langevin:

    γ_L dX_L/dt = -∇_L U + sqrt(2 γ_L kT) η_L
    γ_R dX_R/dt = -∇_R U + sqrt(2 γ_R kT) η_R

U includes:
    - PDG-inspired single-particle wells for ligand and receptor
    - Ligand–receptor binding potential in separation r
    - Harmonic tether for receptor (e.g. membrane anchoring)

Usage:
    python3 pdg_ligand_receptor.py
"""

import numpy as np
import matplotlib.pyplot as plt


class PDGLigandReceptor:
    def __init__(
        self,
        kT=1.0,
        gamma_L=1.0,
        gamma_R=1.0,
        # PDG single-particle wells
        k_L=0.5,
        A_L=1.0,
        xi_L=1.0,
        k_R=0.5,
        A_R=1.0,
        xi_R=1.0,
        # ligand–receptor binding
        eps_bind=5.0,
        r_bind=1.0,
        w_bind=0.3,
        # receptor tether
        k_tether=0.5,
        R0=None,
        dt=1e-3,
        n_steps=50000,
        overdamped=True,
        seed=None,
    ):
        self.kT = kT
        self.gamma_L = gamma_L
        self.gamma_R = gamma_R
        self.k_L = k_L
        self.A_L = A_L
        self.xi_L = xi_L
        self.k_R = k_R
        self.A_R = A_R
        self.xi_R = xi_R

        self.eps_bind = eps_bind
        self.r_bind = r_bind
        self.w_bind = w_bind

        self.k_tether = k_tether
        self.R0 = np.array(R0, float).reshape(3) if R0 is not None else np.zeros(3)

        self.dt = dt
        self.n_steps = n_steps
        self.overdamped = overdamped

        self.rng = np.random.default_rng(seed)
        if overdamped:
            self.sigma_L = np.sqrt(2.0 * kT / gamma_L)
            self.sigma_R = np.sqrt(2.0 * kT / gamma_R)
        else:
            raise NotImplementedError("This example is overdamped only.")

    def U_pdg_single(self, X, k, A, xi):
        r2 = np.sum(X**2)
        U_h = 0.5 * k * r2
        U_cd = -A * np.exp(-r2 / (2 * xi**2))
        return U_h + U_cd

    def grad_U_pdg_single(self, X, k, A, xi):
        r2 = np.sum(X**2)
        exp_term = np.exp(-r2 / (2 * xi**2))
        dU_dr2 = 0.5 * k - A * (-1.0 / (2 * xi**2)) * exp_term
        return dU_dr2 * 2.0 * X

    def U_bind(self, X_L, X_R):
        d = X_L - X_R
        r = np.linalg.norm(d)
        return -self.eps_bind * np.exp(-(r - self.r_bind)**2 / (2 * self.w_bind**2))

    def grad_U_bind(self, X_L, X_R):
        d = X_L - X_R
        r = np.linalg.norm(d)
        if r < 1e-12:
            return np.zeros(3), np.zeros(3)
        e = d / r
        dUdr = self.eps_bind * np.exp(-(r - self.r_bind)**2 / (2 * self.w_bind**2)) * ((r - self.r_bind) / (self.w_bind**2))
        grad_L = dUdr * e
        grad_R = -grad_L
        return grad_L, grad_R

    def U_tether(self, X_R):
        dR = X_R - self.R0
        return 0.5 * self.k_tether * np.sum(dR**2)

    def grad_U_tether(self, X_R):
        return self.k_tether * (X_R - self.R0)

    def total_potential(self, X_L, X_R):
        U_L = self.U_pdg_single(X_L, self.k_L, self.A_L, self.xi_L)
        U_R = self.U_pdg_single(X_R, self.k_R, self.A_R, self.xi_R)
        U_b = self.U_bind(X_L, X_R)
        U_t = self.U_tether(X_R)
        return U_L + U_R + U_b + U_t

    def forces(self, X_L, X_R):
        grad_L = self.grad_U_pdg_single(X_L, self.k_L, self.A_L, self.xi_L)
        grad_R = self.grad_U_pdg_single(X_R, self.k_R, self.A_R, self.xi_R)
        grad_Lb, grad_Rb = self.grad_U_bind(X_L, X_R)
        grad_Rt = self.grad_U_tether(X_R)

        grad_L_tot = grad_L + grad_Lb
        grad_R_tot = grad_R + grad_Rb + grad_Rt

        F_L = -grad_L_tot
        F_R = -grad_R_tot
        return F_L, F_R

    def simulate(self, X_L0=None, X_R0=None, store_every=10):
        dt = self.dt
        n_frames = self.n_steps // store_every + 1

        if X_L0 is None:
            X_L = self.rng.normal(0, 2.0, size=3)
        else:
            X_L = np.array(X_L0, float).reshape(3)
        if X_R0 is None:
            X_R = self.R0.copy()
        else:
            X_R = np.array(X_R0, float).reshape(3)

        traj_L = np.zeros((n_frames, 3))
        traj_R = np.zeros((n_frames, 3))
        traj_L[0] = X_L
        traj_R[0] = X_R
        frame_idx = 1

        for n in range(1, self.n_steps + 1):
            F_L, F_R = self.forces(X_L, X_R)

            dW_L = self.rng.normal(0.0, np.sqrt(dt), size=3)
            dW_R = self.rng.normal(0.0, np.sqrt(dt), size=3)

            X_L = X_L + (F_L / self.gamma_L) * dt + self.sigma_L * dW_L
            X_R = X_R + (F_R / self.gamma_R) * dt + self.sigma_R * dW_R

            if n % store_every == 0:
                traj_L[frame_idx] = X_L
                traj_R[frame_idx] = X_R
                frame_idx += 1

        return {"L": traj_L, "R": traj_R, "dt": dt, "store_every": store_every}

    def compute_r(self, result):
        L = result["L"]
        R = result["R"]
        d = L - R
        r = np.linalg.norm(d, axis=1)
        t = np.arange(L.shape[0]) * result["store_every"] * result["dt"]
        return t, r

    def plot_r(self, result):
        t, r = self.compute_r(result)
        plt.figure()
        plt.plot(t, r)
        plt.xlabel("t")
        plt.ylabel("r_LR")
        plt.title("Ligand–receptor separation vs time")
        plt.tight_layout()
        plt.show()

    def plot_trajectory_xy(self, result):
        L = result["L"]
        R = result["R"]
        plt.figure()
        plt.plot(L[:, 0], L[:, 1], label="ligand")
        plt.plot(R[:, 0], R[:, 1], label="receptor")
        plt.scatter(L[0, 0], L[0, 1], marker="o", label="L start")
        plt.scatter(R[0, 0], R[0, 1], marker="o", label="R start")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Ligand and receptor trajectories (projection)")
        plt.legend()
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


def main():
    sim = PDGLigandReceptor(
        kT=1.0,
        gamma_L=1.0,
        gamma_R=1.0,
        k_L=0.2,
        A_L=0.5,
        xi_L=2.0,
        k_R=0.2,
        A_R=0.5,
        xi_R=2.0,
        eps_bind=6.0,
        r_bind=1.0,
        w_bind=0.3,
        k_tether=0.5,
        R0=[0.0, 0.0, 0.0],
        dt=1e-3,
        n_steps=100000,
        overdamped=True,
        seed=123,
    )

    result = sim.simulate(store_every=50)
    sim.plot_r(result)
    sim.plot_trajectory_xy(result)


if __name__ == "__main__":
    main()
