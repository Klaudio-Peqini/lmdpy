
#!/usr/bin/env python3
"""pdg_multi_particle.py

Multi-particle interacting Langevin model inspired by P–DG coherent domains.

We simulate N particles in d dimensions with underdamped Langevin dynamics:

    m d2x_i/dt2 = -∇_i U({x_j}) - γ dx_i/dt + sqrt(2 γ k_B T) η_i(t),

where the potential has three contributions:

    U = sum_i U_PDGi(x_i) + sum_{i<j} U_rep(r_ij) + sum_{i<j} U_CD_pair(r_ij)

- U_PDGi(x_i): single-particle P–DG coherent well + elastic term
- U_rep: short-range repulsive soft-core (WCA-like)
- U_CD_pair: coherent-domain–inspired long-range attraction (Yukawa-like)

Usage:
    python3 pdg_multi_particle.py
"""

import numpy as np
import matplotlib.pyplot as plt


class PDGMultiParticle:
    def __init__(
        self,
        dim=2,
        n_particles=50,
        m=1.0,
        gamma=1.0,
        kT=1.0,
        # single-particle PDG potential
        k_single=1.0,
        A_single=1.0,
        xi_single=1.0,
        # pairwise interactions
        epsilon_rep=1.0,
        sigma_rep=0.5,
        A_pair=0.5,
        xi_pair=2.0,
        dt=1e-3,
        n_steps=20000,
        integrator="srk",
        seed=None,
    ):
        self.dim = dim
        self.N = n_particles
        self.m = m
        self.gamma = gamma
        self.kT = kT

        self.k_single = k_single
        self.A_single = A_single
        self.xi_single = xi_single

        self.epsilon_rep = epsilon_rep
        self.sigma_rep = sigma_rep
        self.A_pair = A_pair
        self.xi_pair = xi_pair

        self.dt = dt
        self.n_steps = n_steps
        self.integrator = integrator.lower()
        self.rng = np.random.default_rng(seed)
        self.sigma_vel = np.sqrt(2.0 * gamma * kT / (m**2))

    # --- Potentials ---

    def U_single(self, x):
        """Single-particle PDG potential: harmonic + Gaussian CD well.

        x : (N, dim)
        returns : (N,)
        """
        r2 = np.sum(x**2, axis=-1)
        U_h = 0.5 * self.k_single * r2
        U_cd = -self.A_single * np.exp(-r2 / (2 * self.xi_single**2))
        return U_h + U_cd

    def grad_U_single(self, x):
        """Gradient of U_single wrt x (N,dim)."""
        r2 = np.sum(x**2, axis=-1, keepdims=True)
        exp_term = np.exp(-r2 / (2 * self.xi_single**2))
        dU_dr2 = 0.5 * self.k_single - self.A_single * (-1.0 / (2 * self.xi_single**2)) * exp_term
        grad = dU_dr2 * 2.0 * x
        return grad

    def pairwise_forces(self, x):
        """Compute pairwise forces from repulsion + CD-inspired attraction.

        U_rep ~ WCA-like repulsion
        U_pair ~ -A_pair * exp(-r/xi_pair) / (r + r0)
        """
        N, d = x.shape
        F = np.zeros_like(x)
        r0 = 1e-3

        for i in range(N):
            for j in range(i + 1, N):
                rij = x[i] - x[j]
                dist = np.linalg.norm(rij)
                if dist == 0:
                    continue
                e = rij / dist

                # Soft repulsion (WCA-like)
                r_cut = 2**(1/6) * self.sigma_rep
                if dist < r_cut:
                    sr = self.sigma_rep / dist
                    sr6 = sr**6
                    f_rep = 24 * self.epsilon_rep * (2 * sr6**2 - sr6) / dist
                else:
                    f_rep = 0.0

                # Coherent-domain–inspired attraction
                f_pair = self.A_pair * np.exp(-dist / self.xi_pair) * (1.0 / (dist + r0) + 1.0 / self.xi_pair)

                F_ij = (f_pair - f_rep) * e
                F[i] += F_ij
                F[j] -= F_ij

        return F

    def forces(self, x, v):
        F_single = -self.grad_U_single(x)
        F_pair = self.pairwise_forces(x)
        return F_single + F_pair - self.gamma * v

    # --- Integration ---

    def step_srk(self, x, v):
        dt = self.dt
        dW = self.rng.normal(0.0, np.sqrt(dt), size=v.shape)
        F = self.forces(x, v)
        a = F / self.m
        v_t = v + a * dt + self.sigma_vel * dW
        x_t = x + v * dt

        F_t = self.forces(x_t, v_t)
        a_t = F_t / self.m

        v_new = v + 0.5 * (a + a_t) * dt + self.sigma_vel * dW
        x_new = x + 0.5 * (v + v_t) * dt
        return x_new, v_new

    def step_euler(self, x, v):
        dt = self.dt
        dW = self.rng.normal(0.0, np.sqrt(dt), size=v.shape)
        F = self.forces(x, v)
        a = F / self.m
        v_new = v + a * dt + self.sigma_vel * dW
        x_new = x + v_new * dt
        return x_new, v_new

    def simulate(self, x0=None, v0=None, store_every=10):
        N, d = self.N, self.dim
        if x0 is None:
            x = self.rng.normal(0, 1.0, size=(N, d))
        else:
            x = np.array(x0, float).reshape(N, d)
        if v0 is None:
            v = np.zeros((N, d))
        else:
            v = np.array(v0, float).reshape(N, d)

        n_frames = self.n_steps // store_every + 1
        traj = np.zeros((n_frames, N, d))
        traj[0] = x
        frame_idx = 1

        for n in range(1, self.n_steps + 1):
            if self.integrator == "srk":
                x, v = self.step_srk(x, v)
            else:
                x, v = self.step_euler(x, v)
            if n % store_every == 0:
                traj[frame_idx] = x
                frame_idx += 1

        return traj

    def plot_snapshot(self, x):
        if self.dim != 2:
            print("plot_snapshot: only implemented for dim=2")
            return
        x = np.asarray(x)
        plt.figure()
        plt.scatter(x[:, 0], x[:, 1], s=20)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("PDG multi-particle snapshot")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


def main():
    sim = PDGMultiParticle(dim=2, n_particles=80, A_pair=0.8, xi_pair=2.5, A_single=1.0, xi_single=1.0)
    traj = sim.simulate(store_every=100)
    sim.plot_snapshot(traj[0])
    sim.plot_snapshot(traj[-1])


if __name__ == "__main__":
    main()
