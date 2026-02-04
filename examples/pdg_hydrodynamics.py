
#!/usr/bin/env python3
"""pdg_hydrodynamics.py

Multi-particle Langevin dynamics with simple hydrodynamic interactions
(Oseen tensor) plus PDG-inspired single-particle potential.

Overdamped dynamics:

    dX = M F dt + sqrt(2 kT M) dW

M is a 3N x 3N mobility matrix including Oseen interactions.

Usage:
    python3 pdg_hydrodynamics.py
"""

import numpy as np
import matplotlib.pyplot as plt


class PDGHydro:
    def __init__(
        self,
        n_particles=10,
        a=1.0,
        mu0=1.0,
        kT=1.0,
        k_single=1.0,
        A_single=1.0,
        xi_single=1.0,
        dt=1e-3,
        n_steps=10000,
        seed=None,
    ):
        self.N = n_particles
        self.a = a
        self.mu0 = mu0
        self.kT = kT
        self.k_single = k_single
        self.A_single = A_single
        self.xi_single = xi_single
        self.dt = dt
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)

    def U_single(self, x):
        r2 = np.sum(x**2, axis=-1)
        U_h = 0.5 * self.k_single * r2
        U_cd = -self.A_single * np.exp(-r2 / (2 * self.xi_single**2))
        return U_h + U_cd

    def grad_U_single(self, x):
        r2 = np.sum(x**2, axis=-1, keepdims=True)
        exp_term = np.exp(-r2 / (2 * self.xi_single**2))
        dU_dr2 = 0.5 * self.k_single - self.A_single * (-1.0 / (2 * self.xi_single**2)) * exp_term
        grad = dU_dr2 * 2.0 * x
        return grad

    def forces(self, X):
        """X shape: (N,3)"""
        return -self.grad_U_single(X)

    def mobility_matrix(self, X):
        """Build 3N x 3N mobility matrix with Oseen interactions."""
        N = self.N
        M = np.zeros((3*N, 3*N))
        I3 = np.eye(3)
        pref = 3 * self.a / 4.0

        for i in range(N):
            M[3*i:3*i+3, 3*i:3*i+3] = self.mu0 * I3
            for j in range(i+1, N):
                rij = X[i] - X[j]
                r = np.linalg.norm(rij)
                if r < 1e-6:
                    continue
                r_hat = rij / r
                O = pref * (1.0 / r) * (I3 + np.outer(r_hat, r_hat))
                M[3*i:3*i+3, 3*j:3*j+3] = self.mu0 * O
                M[3*j:3*j+3, 3*i:3*i+3] = self.mu0 * O
        return M

    def simulate(self, X0=None, store_every=10):
        N = self.N
        if X0 is None:
            X = self.rng.normal(0, 1.0, size=(N, 3))
        else:
            X = np.array(X0, float).reshape(N, 3)

        n_frames = self.n_steps // store_every + 1
        traj = np.zeros((n_frames, N, 3))
        traj[0] = X
        frame_idx = 1

        dt = self.dt
        for n in range(1, self.n_steps + 1):
            F = self.forces(X).reshape(3*N)
            M = self.mobility_matrix(X)
            drift = M @ F
            try:
                L = np.linalg.cholesky(2 * self.kT * M * dt)
            except np.linalg.LinAlgError:
                M_reg = M + 1e-8 * np.eye(3*N)
                L = np.linalg.cholesky(2 * self.kT * M_reg * dt)
            dW = self.rng.normal(0.0, 1.0, size=3*N)
            dX = drift * dt + L @ dW
            X = X + dX.reshape(N, 3)

            if n % store_every == 0:
                traj[frame_idx] = X
                frame_idx += 1

        return traj

    def plot_snapshot(self, X):
        X = np.asarray(X)
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], s=40)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Hydrodynamic PDG snapshot (x-y)")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


def main():
    sim = PDGHydro(n_particles=10, a=0.2, mu0=1.0, kT=1.0, A_single=0.5)
    traj = sim.simulate(store_every=50)
    sim.plot_snapshot(traj[0])
    sim.plot_snapshot(traj[-1])


if __name__ == "__main__":
    main()
