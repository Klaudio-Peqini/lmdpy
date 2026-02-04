
"""
lmdpy.langevin_core

Generic Langevin integrator that combines:

    - a Potential object (with U(x), gradU(x))
    - a Solvent object (γ(x,t), T(x,t), v_flow(x,t))

and integrates either overdamped or underdamped Langevin dynamics.

SDEs (k_B = 1 units):

Overdamped:
    dx = [ -∇U(x) / γ(x,t) + v_flow(x,t) ] dt + sqrt(2 T(x,t) / γ(x,t)) dW

Underdamped:
    dx = v dt
    dv = [ -∇U(x)/m - γ(x,t)/m (v - v_flow(x,t)) ] dt
         + sqrt(2 γ(x,t) T(x,t) / m^2) dW

The integrator supports:
    - Euler–Maruyama ("euler")
    - stochastic Heun / SRK ("srk") for overdamped case
"""

import numpy as np


class LangevinIntegrator:
    """
    Generic Langevin integrator for a single particle in dim dimensions.

    Parameters
    ----------
    potential : object with U(x) and gradU(x)
    solvent : lmdpy.solvent.* instance (or similar API)
        If None, use homogeneous solvent with given gamma and temperature.
    m : float
        Mass (for underdamped).
    dim : int
        Dimension (1, 2, 3, ...).
    overdamped : bool
        If True, integrate overdamped dynamics; else underdamped.
    dt : float
        Time step.
    integrator : {"euler", "srk"}
        Integration scheme. SRK only affects overdamped case here.
    """

    def __init__(
        self,
        potential,
        solvent=None,
        m=1.0,
        dim=3,
        overdamped=True,
        dt=1e-3,
        integrator="euler",
        rng=None,
    ):
        from .solvent import HomogeneousSolvent

        self.potential = potential
        self.dim = int(dim)
        self.m = float(m)
        self.overdamped = bool(overdamped)
        self.dt = float(dt)
        self.integrator = integrator.lower()
        self.rng = np.random.default_rng() if rng is None else rng

        if solvent is None:
            solvent = HomogeneousSolvent()
        self.solvent = solvent

    # ------------------------------------------------------------------
    # Core stepping
    # ------------------------------------------------------------------
    def step(self, t, x, v=None):
        """
        Perform one Langevin step.

        Parameters
        ----------
        t : float
            Current time.
        x : array_like, shape (dim,)
            Current position.
        v : array_like, shape (dim,), optional
            Current velocity (for underdamped).

        Returns
        -------
        t_new, x_new, v_new
        """
        x = np.asarray(x, dtype=float).reshape(self.dim)
        if not self.overdamped:
            if v is None:
                raise ValueError("Underdamped mode requires velocity v.")
            v = np.asarray(v, dtype=float).reshape(self.dim)

        if self.overdamped:
            if self.integrator == "srk":
                x_new = self._step_overdamped_srk(t, x)
            else:
                x_new = self._step_overdamped_euler(t, x)
            return t + self.dt, x_new, None
        else:
            if self.integrator == "srk":
                # same scheme, but with Heun-like dv; keep it simple here
                x_new, v_new = self._step_underdamped_euler(t, x, v)
            else:
                x_new, v_new = self._step_underdamped_euler(t, x, v)
            return t + self.dt, x_new, v_new

    # --- Overdamped ---------------------------------------------------
    def _step_overdamped_euler(self, t, x):
        dt = self.dt
        x = x.reshape(self.dim)
        gamma = self.solvent.gamma(x, t)  # scalar or array shape ()
        T = self.solvent.temperature(x, t)
        v_flow = self.solvent.flow_velocity(x, t).reshape(self.dim)

        # Broadcast gamma, T to scalar
        gamma = float(np.asarray(gamma).mean())
        T = float(np.asarray(T).mean())

        dW = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=self.dim)
        gradU = self.potential.gradU(x)
        drift = -gradU / gamma + v_flow
        sigma = np.sqrt(2.0 * T / gamma)
        x_new = x + drift * dt + sigma * dW
        return x_new

    def _step_overdamped_srk(self, t, x):
        dt = self.dt
        x = x.reshape(self.dim)
        dW = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=self.dim)

        gamma = self.solvent.gamma(x, t)
        T = self.solvent.temperature(x, t)
        v_flow = self.solvent.flow_velocity(x, t).reshape(self.dim)
        gamma = float(np.asarray(gamma).mean())
        T = float(np.asarray(T).mean())
        sigma = np.sqrt(2.0 * T / gamma)

        gradU = self.potential.gradU(x)
        drift_x = -gradU / gamma + v_flow
        x_tilde = x + drift_x * dt + sigma * dW

        gamma_t = self.solvent.gamma(x_tilde, t + dt)
        T_t = self.solvent.temperature(x_tilde, t + dt)
        v_flow_t = self.solvent.flow_velocity(x_tilde, t + dt).reshape(self.dim)
        gamma_t = float(np.asarray(gamma_t).mean())
        T_t = float(np.asarray(T_t).mean())
        sigma_t = np.sqrt(2.0 * T_t / gamma_t)

        gradU_t = self.potential.gradU(x_tilde)
        drift_xt = -gradU_t / gamma_t + v_flow_t

        drift_avg = 0.5 * (drift_x + drift_xt)
        # For simplicity we reuse the original noise amplitude sigma
        x_new = x + drift_avg * dt + sigma * dW
        return x_new

    # --- Underdamped --------------------------------------------------
    def _step_underdamped_euler(self, t, x, v):
        dt = self.dt
        x = x.reshape(self.dim)
        v = v.reshape(self.dim)

        gamma = self.solvent.gamma(x, t)
        T = self.solvent.temperature(x, t)
        v_flow = self.solvent.flow_velocity(x, t).reshape(self.dim)
        gamma = float(np.asarray(gamma).mean())
        T = float(np.asarray(T).mean())

        gradU = self.potential.gradU(x)

        # dv = [ -∇U/m - γ/m (v - v_flow) ] dt + sqrt(2 γ T / m^2) dW
        dW = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=self.dim)
        sigma_v = np.sqrt(2.0 * gamma * T / (self.m**2))

        a = (-gradU / self.m) - (gamma / self.m) * (v - v_flow)
        v_new = v + a * dt + sigma_v * dW
        x_new = x + v_new * dt
        return x_new, v_new

    # ------------------------------------------------------------------
    # High-level simulation
    # ------------------------------------------------------------------
    def simulate(
        self,
        n_steps,
        x0=None,
        v0=None,
        store_trajectory=True,
    ):
        """
        Run a single-particle simulation.

        Parameters
        ----------
        n_steps : int
            Number of time steps.
        x0 : array_like, shape (dim,), optional
            Initial position. Defaults to zeros.
        v0 : array_like, shape (dim,), optional
            Initial velocity (for underdamped). Defaults to zeros.
        store_trajectory : bool
            If True, store all t, x, v. Otherwise, only final state.

        Returns
        -------
        results : dict
            Keys:
                - "t": shape (N,)
                - "x": shape (N, dim)
                - "v": shape (N, dim) or None
                - "U": potential energy time series
                - "K": kinetic energy time series
                - "E": total energy time series
        """
        x = np.zeros(self.dim) if x0 is None else np.asarray(x0, dtype=float).reshape(self.dim)
        if self.overdamped:
            v = None
        else:
            v = np.zeros(self.dim) if v0 is None else np.asarray(v0, dtype=float).reshape(self.dim)

        t = 0.0
        times = []
        xs = []
        vs = []
        U_list = []
        K_list = []

        for _ in range(int(n_steps)):
            if store_trajectory:
                times.append(t)
                xs.append(x.copy())
                if not self.overdamped:
                    vs.append(v.copy())
                U = self.potential.U(x)
                if self.overdamped:
                    K = 0.0
                else:
                    K = 0.5 * self.m * np.sum(v * v)
                U_list.append(U)
                K_list.append(K)

            t, x, v = self.step(t, x, v)

        times.append(t)
        xs.append(x.copy())
        if not self.overdamped:
            vs.append(v.copy())
        U = self.potential.U(x)
        if self.overdamped:
            K = 0.0
        else:
            K = 0.5 * self.m * np.sum(v * v)
        U_list.append(U)
        K_list.append(K)

        times = np.asarray(times)
        xs = np.asarray(xs)
        U_arr = np.asarray(U_list)
        K_arr = np.asarray(K_list)
        E_arr = U_arr + K_arr

        if self.overdamped:
            v_arr = None
        else:
            v_arr = np.asarray(vs)

        return {
            "t": times,
            "x": xs,
            "v": v_arr,
            "U": U_arr,
            "K": K_arr,
            "E": E_arr,
        }
