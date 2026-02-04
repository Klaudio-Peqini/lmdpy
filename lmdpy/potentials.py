
"""
lmdpy.potentials

Generic potential-energy models U(x) and their gradients ∇U(x).

All potentials follow the interface:

    class SomePotential:
        def U(self, x):
            '''
            Parameters
            ----------
            x : np.ndarray, shape (..., dim)
                Position(s) in Cartesian coordinates.

            Returns
            -------
            U : np.ndarray, shape (...)
                Potential energy at each position.

        def gradU(self, x):

            Returns ∇U with same shape as x.
            '''

The convention is that x can be:
- a single point, shape (dim,)
- an array of points, shape (N, dim)
- or higher-dimensional (..., dim).
"""

import numpy as np


class HarmonicPotential:
    """
    U(x) = 0.5 sum_i k_i x_i^2

    Parameters
    ----------
    k : float or array_like of shape (dim,)
        Stiffness coefficients. If scalar, same k in all directions.
    """

    def __init__(self, k=1.0):
        self.k = np.atleast_1d(k).astype(float)

    def U(self, x):
        x = np.asarray(x, dtype=float)
        k = self._broadcast_k(x)
        return 0.5 * np.sum(k * x * x, axis=-1)

    def gradU(self, x):
        x = np.asarray(x, dtype=float)
        k = self._broadcast_k(x)
        return k * x

    def _broadcast_k(self, x):
        # k shape (dim,) -> broadcast to x shape (..., dim)
        if self.k.size == 1:
            return np.broadcast_to(self.k[0], x.shape)
        if self.k.shape[-1] != x.shape[-1]:
            raise ValueError("HarmonicPotential: k dimension does not match x.")
        # reshape to (1,...,1, dim) then broadcast
        k = self.k.reshape((1,) * (x.ndim - 1) + (-1,))
        return np.broadcast_to(k, x.shape)


class PDGCoherentPotential:
    """
    PDG-inspired coherent-domain potential:

        U(x) = 0.5 * k * |x|^2 - A_cd * exp(-|x|^2 / (2 xi^2))

    Parameters
    ----------
    k : float
        Harmonic stiffness.
    A_cd : float
        Coherent-domain depth (amplitude).
    xi : float
        Coherence length.
    """

    def __init__(self, k=1.0, A_cd=1.0, xi=1.0):
        self.k = float(k)
        self.A_cd = float(A_cd)
        self.xi = float(xi)

    def U(self, x):
        x = np.asarray(x, dtype=float)
        r2 = np.sum(x * x, axis=-1)
        U_h = 0.5 * self.k * r2
        U_cd = -self.A_cd * np.exp(-r2 / (2.0 * self.xi**2))
        return U_h + U_cd

    def gradU(self, x):
        x = np.asarray(x, dtype=float)
        r2 = np.sum(x * x, axis=-1, keepdims=True)
        grad_h = self.k * x
        factor = (self.A_cd / (self.xi**2)) * np.exp(-r2 / (2.0 * self.xi**2))
        grad_cd = factor * x
        return grad_h + grad_cd


class DoubleWell1D:
    """
    One-dimensional double-well along x:

        U(x) = a x^4 - b x^2

    If x has dim>1, only the first coordinate x[...,0] is affected and the
    others are left unchanged (U only depends on x0).

    Parameters
    ----------
    a, b : float
        Parameters of the double well.
    """

    def __init__(self, a=1.0, b=1.0):
        self.a = float(a)
        self.b = float(b)

    def U(self, x):
        x = np.asarray(x, dtype=float)
        x0 = x[..., 0]
        return self.a * x0**4 - self.b * x0**2

    def gradU(self, x):
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x)
        x0 = x[..., 0]
        g[..., 0] = 4.0 * self.a * x0**3 - 2.0 * self.b * x0
        return g


class LennardJonesSingleCenter:
    """
    Single-center Lennard-Jones potential around the origin:

        U(r) = 4 ε [ (σ/r)^12 - (σ/r)^6 ]

    This is useful as a simple testbed for steric repulsion / attraction around
    a central point (e.g. receptor, obstacle). For true multi-particle LJ,
    build pairwise sums externally.

    Parameters
    ----------
    epsilon : float
        Depth of the potential well.
    sigma : float
        Characteristic length (particle size).
    r_cut : float or None
        Optional cutoff radius; beyond r_cut, U(r) = 0 (shifted to be continuous).
    """

    def __init__(self, epsilon=1.0, sigma=1.0, r_cut=None):
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.r_cut = r_cut

    def U(self, x):
        x = np.asarray(x, dtype=float)
        r = np.linalg.norm(x, axis=-1)
        sr6 = (self.sigma / r)**6
        sr12 = sr6 * sr6
        U = 4.0 * self.epsilon * (sr12 - sr6)

        if self.r_cut is not None:
            rc = float(self.r_cut)
            src6 = (self.sigma / rc)**6
            src12 = src6 * src6
            U_cut = 4.0 * self.epsilon * (src12 - src6)
            U = np.where(r < rc, U - U_cut, 0.0)
        return U

    def gradU(self, x):
        x = np.asarray(x, dtype=float)
        r = np.linalg.norm(x, axis=-1, keepdims=True)
        # Avoid division by zero
        r_safe = np.where(r == 0.0, 1e-14, r)
        sr6 = (self.sigma / r_safe)**6
        sr12 = sr6 * sr6

        dU_dr = 24.0 * self.epsilon * (2.0 * sr12 - sr6) / r_safe
        grad = dU_dr * (x / r_safe)

        if self.r_cut is not None:
            rc = float(self.r_cut)
            mask = (r > rc)
            grad = np.where(mask, 0.0, grad)
        return grad


class MorseSingleCenter:
    """
    Single-center Morse potential around the origin:

        U(r) = D_e (1 - exp[-a (r - r0)])^2 - D_e

    Parameters
    ----------
    D_e : float
        Dissociation energy.
    a : float
        Range parameter.
    r0 : float
        Equilibrium distance.
    """

    def __init__(self, D_e=1.0, a=1.0, r0=1.0):
        self.D_e = float(D_e)
        self.a = float(a)
        self.r0 = float(r0)

    def U(self, x):
        x = np.asarray(x, dtype=float)
        r = np.linalg.norm(x, axis=-1)
        y = np.exp(-self.a * (r - self.r0))
        return self.D_e * (1.0 - y)**2 - self.D_e

    def gradU(self, x):
        x = np.asarray(x, dtype=float)
        r = np.linalg.norm(x, axis=-1, keepdims=True)
        r_safe = np.where(r == 0.0, 1e-14, r)
        y = np.exp(-self.a * (r_safe - self.r0))
        dU_dr = 2.0 * self.D_e * (1.0 - y) * (self.a * y)
        # gradU = dU/dr * (x / r)
        grad = dU_dr * (x / r_safe)
        return grad
