import math
import numpy as np

# Try gymnasium first, fallback to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    import gym
    from gym import spaces

from typing import Tuple, Optional, Dict, Any

from env.dynamics.parafoil_model import parafoil_model
from reward.shaping import compute_reward


def rk4_step(func, y, t, dt, *args, **kwargs):
    k1 = func(y, t, *args, **kwargs)
    k2 = func(y + 0.5 * dt * k1, t + 0.5 * dt, *args, **kwargs)
    k3 = func(y + 0.5 * dt * k2, t + 0.5 * dt, *args, **kwargs)
    k4 = func(y + dt * k3, t + dt, *args, **kwargs)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class ParaParams:
    """Minimal parameter set to run parafoil_model.

    The values are placeholders and should be adjusted to your platform.
    """

    def __init__(self):
        # Environment
        self.gn = 9.81
        self.Rho = 1.225

        # Geometry (approximate)
        self.b = 2.0  # span [m]
        self.c = 0.7  # chord [m]
        self.t = 0.15  # thickness scale [m]
        self.As = self.b * self.c  # reference area [m^2]
        self.Ap = 0.10  # payload frontal area [m^2]

        # Masses
        self.mc = 6.0  # canopy mass [kg]
        self.mp = 4.0  # payload mass [kg]

        # Installation/geometry extras
        self.miu = 0.0
        self.r = 1.0
        self.sloc = 0.1
        self.ca = math.radians(60.0)  # use radians

        # Aerodynamic coefficients (coarse defaults)
        self.CD0 = 0.04
        self.CDa2 = 0.30
        self.CDds = 0.10
        self.CYbeta = 0.05
        self.CL0 = 0.2
        self.CLa = 3.0
        self.CLds = 0.30
        self.Clbeta = -0.05
        self.Clp = -0.40
        self.Clr = 0.20
        self.Clda = 0.10
        self.Cm0 = 0.0
        self.Cma = -0.50
        self.Cmq = -2.0
        self.Cnbeta = 0.20
        self.Cnp = -0.10
        self.Cnr = -0.20
        self.Cnda = 0.10
        self.CDp = 1.0

        # Coupling stiffness/damping (placeholders)
        self.k_r = 0.5
        self.k_f = 5.0
        self.c_f = 0.8
        self.k_psi = 0.5

        # Wind and attachment points
        self.vw = np.array([[0.0], [0.0], [0.0]])  # wind in inertial frame
        self.rcOc = np.array([[0.0], [0.0], [0.20]])
        self.rcOp = np.array([[0.0], [0.0], [-0.50]])

        # Controls (to be set each step)
        self.left = 0.0
        self.right = 0.0
        self.thrust = np.array([[0.0], [0.0], [0.0]])


class ParafoilEnv(gym.Env):
    """RL environment wrapping parafoil_model with RK4 integration.

    Action: 3-dim Box [-1,1] -> [throttle, left, right]
    Observation: tracking errors and selected states
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt_env: float = 0.04,
        n_substeps: int = 4,
        ref_type: str = "line",
        ref_speed: float = 8.0,
        ref_alt: float = 50.0,
        max_episode_steps: int = 2000,
        action_limits: Optional[Dict[str, float]] = None,
        actuator_tau: float = 0.15,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dt_env = dt_env
        self.n_substeps = n_substeps
        self.dt_sub = dt_env / n_substeps
        self.ref_type = ref_type
        self.ref_speed = ref_speed
        self.ref_alt = ref_alt
        self.max_episode_steps = max_episode_steps
        self.actuator_tau = actuator_tau
        self._rng = np.random.default_rng(seed)

        # Parameters instance used by the dynamics
        self.para = ParaParams()

        # Action mapping scales (placeholders, configurable)
        lims = action_limits or {}
        self.thrust_max = lims.get("thrust_max", 10.0)  # N
        self.defl_max = lims.get("deflection_max", 0.5)  # non-dimensional or rad-equivalent
        self.defl_rate_max = lims.get("deflection_rate_max", 2.0)  # per second
        self.thrust_rate_max = lims.get("thrust_rate_max", 50.0)  # N/s

        # Gym spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # obs: [e_p(3), e_v(3), e_psi(1), w(3), thetar, psir, speed]
        high_obs = np.array([
            1e3, 1e3, 1e3,  # position error
            1e2, 1e2, 1e2,  # velocity error
            math.pi,         # yaw error
            50.0, 50.0, 50.0,  # angular rates
            math.pi, math.pi,  # thetar, psir
            1e2,              # speed
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)

        # Internal state
        self.state = np.zeros(20, dtype=float)
        self.t = 0.0
        self.steps = 0
        self._last_cmd = np.zeros(3, dtype=float)  # [T,left,right] physical

    # --- Reference ---
    def _ref(self, t: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (p_ref[3], v_ref[3], yaw_ref) in inertial frame."""
        if self.ref_type == "circle":
            R = 100.0
            w = self.ref_speed / R
            x = R * math.cos(w * t)
            y = R * math.sin(w * t)
            vx = -R * w * math.sin(w * t)
            vy = R * w * math.cos(w * t)
            yaw = math.atan2(vy, vx)
            return np.array([x, y, self.ref_alt]), np.array([vx, vy, 0.0]), yaw
        # default: straight line along x
        x = self.ref_speed * t
        y = 0.0
        z = self.ref_alt
        vx = self.ref_speed
        vy = 0.0
        vz = 0.0
        yaw = 0.0
        return np.array([x, y, z]), np.array([vx, vy, vz]), yaw

    # --- Reset ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.para = ParaParams()

        # Randomize initial state near reference
        pos0 = np.array([
            0.0 + self._rng.normal(0, 2.0),
            0.0 + self._rng.normal(0, 2.0),
            self.ref_alt + self._rng.normal(0, 1.0),
        ])
        # small initial angles and rates
        phi0 = self._rng.normal(0, math.radians(5))
        theta0 = self._rng.normal(0, math.radians(5))
        psi0 = self._rng.normal(0, math.radians(5))

        thetar0 = 0.0
        psir0 = 0.0

        v0 = np.array([self.ref_speed + self._rng.normal(0, 0.5), 0.0, 0.0])
        w0 = np.zeros(3)
        vp0 = v0.copy()
        ws0 = np.zeros(3)

        y = np.zeros(20)
        y[0:3] = pos0
        y[3:6] = np.array([phi0, theta0, psi0])
        y[6] = thetar0
        y[7] = psir0
        y[8:11] = v0
        y[11:14] = w0
        y[14:17] = vp0
        y[17:20] = ws0

        self.state = y
        self.t = 0.0
        self.steps = 0
        self._last_cmd = np.array([0.0, 0.0, 0.0])  # [T,left,right]

        obs = self._get_obs()
        info = {}
        return obs, info

    # --- Step ---
    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        # Map normalized action to physical commands (cmd_T, cmd_left, cmd_right)
        cmd_T = 0.5 * (action[0] + 1.0) * self.thrust_max  # [0, T_max]
        cmd_left = action[1] * self.defl_max
        cmd_right = action[2] * self.defl_max

        # Apply rate limits and first-order actuator (discretized)
        for _ in range(self.n_substeps):
            # rate limiting towards command
            dT_max = self.thrust_rate_max * self.dt_sub
            dD_max = self.defl_rate_max * self.dt_sub
            dT = np.clip(cmd_T - self._last_cmd[0], -dT_max, dT_max)
            dL = np.clip(cmd_left - self._last_cmd[1], -dD_max, dD_max)
            dR = np.clip(cmd_right - self._last_cmd[2], -dD_max, dD_max)
            # first-order lag
            alpha = self.dt_sub / max(self.actuator_tau, 1e-3)
            self._last_cmd[0] += alpha * dT
            self._last_cmd[1] += alpha * dL
            self._last_cmd[2] += alpha * dR

            # Write to para
            self.para.left = float(np.clip(self._last_cmd[1], -self.defl_max, self.defl_max))
            self.para.right = float(np.clip(self._last_cmd[2], -self.defl_max, self.defl_max))
            self.para.thrust = np.array([[self._last_cmd[0]], [0.0], [0.0]])

            # Integrate dynamics
            self.state = rk4_step(parafoil_model, self.state, self.t, self.dt_sub, self.para)
            self.t += self.dt_sub

        self.steps += 1

        obs = self._get_obs()
        reward = self._get_reward(obs, action)
        terminated, truncated, done_info = self._termination()
        info = {"t": self.t, **done_info}
        return obs, reward, terminated, truncated, info

    # --- Observation ---
    def _get_obs(self) -> np.ndarray:
        y = self.state
        p = y[0:3]
        v = y[8:11]
        phi, theta, psi = y[3], y[4], y[5]
        thetar, psir = y[6], y[7]
        w = y[11:14]

        p_ref, v_ref, yaw_ref = self._ref(self.t)
        e_p = p - p_ref
        e_v = v - v_ref
        e_psi = self._wrap_angle(psi - yaw_ref)
        speed = float(np.linalg.norm(v))

        obs = np.array([
            e_p[0], e_p[1], e_p[2],
            e_v[0], e_v[1], e_v[2],
            e_psi,
            w[0], w[1], w[2],
            thetar, psir,
            speed,
        ], dtype=np.float32)
        return obs

    # --- Reward ---
    def _get_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        return float(compute_reward(obs, action))

    # --- Termination ---
    def _termination(self) -> Tuple[bool, bool, Dict[str, Any]]:
        y = self.state
        p = y[0:3]
        phi, theta = y[3], y[4]
        speed = float(np.linalg.norm(y[8:11]))

        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        if p[2] < 0.0:
            terminated = True
            info["event"] = "ground_impact"
        if abs(phi) > math.radians(60) or abs(theta) > math.radians(60):
            terminated = True
            info["event"] = info.get("event", "attitude_limit")
        if not np.isfinite(self.state).all():
            terminated = True
            info["event"] = info.get("event", "numerical")
        if speed < 0.5:
            terminated = True
            info["event"] = info.get("event", "stall")
        if self.steps >= self.max_episode_steps:
            truncated = True

        return terminated, truncated, info

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi


def make_env(**kwargs) -> ParafoilEnv:
    return ParafoilEnv(**kwargs)
