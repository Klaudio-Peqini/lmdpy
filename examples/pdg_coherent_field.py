
#!/usr/bin/env python3
"""pdg_coherent_field.py

Particles coupled to a dynamical coherent-domain field.

We model:
    - A scalar field a(t) representing the amplitude of a coherent EM mode.
    - N particles with coordinates x_i and velocities v_i.

Equations:

    m d2x_i/dt2 = -∇_i U_particles(x_i, a) - γ dx_i/dt + sqrt(2 γ kT) η_i(t)
    da/dt       = -(κ a - g * sum_i c(x_i)) + sqrt(2 D_a) ξ(t)

U_particles has:
    - harmonic + PDG-like single-particle term
    - coupling to field a:   - g a c(x_i)

Usage:
    python3 pdg_coherent_field.py
"""

import numpy as np
import matplotlib.pyplot as plt


class PDGCoherentField:
    def __init__(
        self,
        dim=1,
        n_particles=50,
        m=1.0,
        gamma=1.0,
        kT=1.0,
        k_single=1.0,
        A_single=1.0,
        xi_single=1.0,
        # field parameters
        kappa=1.0,
        g=0.5,
        D_a=0.1,
        dt=1e-3,
        n_steps=50000,
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

        self.kappa = kappa
        self.g = g
        self.D_a = D_a

        self.dt = dt
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)
        self.sigma_vel = np.sqrt(2.0 * gamma * kT / (m**2))
        self.sigma_a = np.sqrt(2.0 * D_a)

    def U_single(self, x, a):
        """PDG-like single-particle potential with field-dependent depth."""
        r2 = np.sum(x**2, axis=-1)
        U_h = 0.5 * self.k_single * r2
        A_eff = self.A_single + self.g * a
        U_cd = -A_eff * np.exp(-r2 / (2 * self.xi_single**2))
        return U_h + U_cd

    def grad_U_single(self, x, a):
        r2 = np.sum(x**2, axis=-1, keepdims=True)
        exp_term = np.exp(-r2 / (2 * self.xi_single**2))
        A_eff = self.A_single + self.g * a
        dU_dr2 = 0.5 * self.k_single - A_eff * (-1.0 / (2 * self.xi_single**2)) * exp_term
        grad = dU_dr2 * 2.0 * x
        return grad

    def c_field(self, x):
        """Coupling function c(x) entering the field equation."""
        r2 = np.sum(x**2, axis=-1)
        return np.exp(-r2 / (2 * self.xi_single**2))

    def simulate(self, x0=None, v0=None, a0=0.0, store_every=50):
        N, d = self.N, self.dim
        if x0 is None:
            x = self.rng.normal(0, 1.0, size=(N, d))
        else:
            x = np.array(x0, float).reshape(N, d)
        if v0 is None:
            v = np.zeros((N, d))
        else:
            v = np.array(v0, float).reshape(N, d)
        a = float(a0)

        n_frames = self.n_steps // store_every + 1
        traj_x = np.zeros((n_frames, N, d))
        traj_a = np.zeros(n_frames)
        traj_x[0] = x
        traj_a[0] = a
        frame_idx = 1

        dt = self.dt
        for n in range(1, self.n_steps + 1):
            # particles (underdamped Euler–Maruyama)
            dW_v = self.rng.normal(0.0, np.sqrt(dt), size=v.shape)
            F = -self.grad_U_single(x, a) - self.gamma * v
            a_part = F / self.m
            v = v + a_part * dt + self.sigma_vel * dW_v
            x = x + v * dt

            # scalar field a(t): OU-like with drive from particles
            c_vals = self.c_field(x)
            drive = self.g * np.sum(c_vals)
            dW_a = self.rng.normal(0.0, np.sqrt(dt))
            a = a + (-self.kappa * a + drive) * dt + self.sigma_a * dW_a

            if n % store_every == 0:
                traj_x[frame_idx] = x
                traj_a[frame_idx] = a
                frame_idx += 1

        return {"x": traj_x, "a": traj_a, "dt": dt, "store_every": store_every}

    def plot_field_time(self, result):
        n_frames = result["a"].shape[0]
        t = np.arange(n_frames) * result["store_every"] * result["dt"]
        plt.figure()
        plt.plot(t, result["a"])
        plt.xlabel("t")
        plt.ylabel("a(t)")
        plt.title("Coherent field amplitude")
        plt.tight_layout()
        plt.show()


def main():
    sim = PDGCoherentField(dim=1, n_particles=50, g=0.01, kappa=0.5, D_a=0.05)
    res = sim.simulate(store_every=100)
    sim.plot_field_time(res)


if __name__ == "__main__":
    main()
