
"""
lmdpy.solvent

Solvent / medium fields for Langevin dynamics:

The idea is to factor out all "environment" effects into an object that
provides:

    - gamma(x, t): friction coefficient (scalar or per-dimension)
    - temperature(x, t): local temperature (sets noise amplitude k_B T)
    - flow_velocity(x, t): solvent flow / drift velocity field

These quantities are then consumed by the Langevin integrator.
"""

import numpy as np


class HomogeneousSolvent:
    """
    Simple homogeneous solvent:

        gamma(x, t) = const
        T(x, t)     = const
        v_flow(x,t) = 0

    Useful baseline for both overdamped and underdamped dynamics.
    """

    def __init__(self, gamma=1.0, temperature=1.0):
        self.gamma_val = float(gamma)
        self.temperature_val = float(temperature)

    def gamma(self, x, t):
        x = np.asarray(x, dtype=float)
        shape = x.shape[:-1]
        return np.broadcast_to(self.gamma_val, shape)

    def temperature(self, x, t):
        x = np.asarray(x, dtype=float)
        shape = x.shape[:-1]
        return np.broadcast_to(self.temperature_val, shape)

    def flow_velocity(self, x, t):
        x = np.asarray(x, dtype=float)
        return np.zeros_like(x)


class LinearShearFlowSolvent:
    """
    Simple shear flow in the x-direction with velocity v_x = shear_rate * y.

    Parameters
    ----------
    gamma : float
        Constant friction coefficient.
    temperature : float
        Constant temperature.
    shear_rate : float
        Shear rate (velocity gradient dv_x/dy).
    """

    def __init__(self, gamma=1.0, temperature=1.0, shear_rate=1.0):
        self.gamma_val = float(gamma)
        self.temperature_val = float(temperature)
        self.shear_rate = float(shear_rate)

    def gamma(self, x, t):
        x = np.asarray(x, dtype=float)
        shape = x.shape[:-1]
        return np.broadcast_to(self.gamma_val, shape)

    def temperature(self, x, t):
        x = np.asarray(x, dtype=float)
        shape = x.shape[:-1]
        return np.broadcast_to(self.temperature_val, shape)

    def flow_velocity(self, x, t):
        x = np.asarray(x, dtype=float)
        v = np.zeros_like(x)
        if x.shape[-1] >= 2:
            y = x[..., 1]
            v[..., 0] = self.shear_rate * y
        return v


class CustomSolvent:
    """
    Fully user-defined solvent via callables:

    Parameters
    ----------
    gamma_func : callable (x, t) -> array_like
        Friction field. If None, use constant gamma0.
    temperature_func : callable (x, t) -> array_like
        Temperature field. If None, use constant T0.
    flow_velocity_func : callable (x, t) -> array_like
        Flow velocity field. If None, assume zero.

    gamma0, T0 are used if corresponding functions are None.
    """

    def __init__(
        self,
        gamma_func=None,
        temperature_func=None,
        flow_velocity_func=None,
        gamma0=1.0,
        T0=1.0,
    ):
        self.gamma_func = gamma_func
        self.temperature_func = temperature_func
        self.flow_velocity_func = flow_velocity_func
        self.gamma0 = float(gamma0)
        self.T0 = float(T0)

    def gamma(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.gamma_func is None:
            shape = x.shape[:-1]
            return np.broadcast_to(self.gamma0, shape)
        return np.asarray(self.gamma_func(x, t), dtype=float)

    def temperature(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.temperature_func is None:
            shape = x.shape[:-1]
            return np.broadcast_to(self.T0, shape)
        return np.asarray(self.temperature_func(x, t), dtype=float)

    def flow_velocity(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.flow_velocity_func is None:
            return np.zeros_like(x)
        return np.asarray(self.flow_velocity_func(x, t), dtype=float)
