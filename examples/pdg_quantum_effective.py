
#!/usr/bin/env python3
"""pdg_quantum_effective.py

Quantum-inspired P–DG effective model.

We simulate:

    m d2q/dt2 = -dV_eff/dq - γ dq/dt + sqrt(2 γ kT) η(t)

with:

    V_eff(q) = α q^2 + β q^4 - A_cd exp(-q^2 / (2 ξ^2))

Usage:
    python3 pdg_quantum_effective.py
"""

import numpy as np
import matplotlib.pyplot as plt


class PDGQuantumEffective:
    def __init__(
        self,
        m=1.0,
        gamma=1.0,
        kT=1.0,
        alpha=-1.0,
        beta=1.0,
        A_cd=1.0,
        xi=1.0,
        dt=1e-3,
        n_steps=200000,
        seed=None,
    ):
        self.m = m
        self.gamma = gamma
        self.kT = kT
        self.alpha = alpha
        self.beta = beta
        self.A_cd = A_cd
        self.xi = xi
        self.dt = dt
        self.n_steps = n_steps

        self.rng = np.random.default_rng(seed)
        self.sigma_vel = np.sqrt(2.0 * gamma * kT / (m**2))

    def V(self, q):
        return self.alpha * q**2 + self.beta * q**4 - self.A_cd * np.exp(-q**2 / (2 * self.xi**2))

    def dVdq(self, q):
        exp_term = np.exp(-q**2 / (2 * self.xi**2))
        return 2 * self.alpha * q + 4 * self.beta * q**3 + self.A_cd * (q / (self.xi**2)) * exp_term

    def simulate(self, q0=0.0, v0=0.0):
        dt = self.dt
        q = float(q0)
        v = float(v0)

        qs = np.zeros(self.n_steps + 1)
        vs = np.zeros(self.n_steps + 1)
        qs[0] = q
        vs[0] = v

        for n in range(1, self.n_steps + 1):
            dW = self.rng.normal(0.0, np.sqrt(dt))
            F = -self.dVdq(q) - self.gamma * v
            a = F / self.m
            v = v + a * dt + self.sigma_vel * dW
            q = q + v * dt
            qs[n] = q
            vs[n] = v

        return qs, vs

    def plot_q(self, qs):
        t = np.arange(len(qs)) * self.dt
        plt.figure()
        plt.plot(t, qs)
        plt.xlabel("t")
        plt.ylabel("q")
        plt.title("PDG quantum-effective coordinate q(t)")
        plt.tight_layout()
        plt.show()


def main():
    sim = PDGQuantumEffective(alpha=-1.0, beta=1.0, A_cd=0.5, xi=1.0)
    qs, vs = sim.simulate(q0=1.0, v0=0.0)
    sim.plot_q(qs)


if __name__ == "__main__":
    main()
