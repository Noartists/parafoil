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

        # Geometry (small scale model from MATLAB)
        self.b = 2.3  # span along arc [m]
        self.Ac = 1.6  # canopy area [m^2]
        self.c = self.Ac / self.b  # chord [m]
        self.t = 0.15 * self.c  # thickness scale [m]
        self.As = 1.5  # projected area [m^2]
        self.Ap = 0.03  # payload frontal area [m^2]

        # Masses (small scale model from MATLAB)
        self.mc = 0.3  # canopy mass [kg]
        self.mp = 12.0  # payload mass [kg]

        # Installation/geometry extras  
        self.ca = 90 * math.pi / 180  # arc angle [rad] (from MATLAB)
        self.r = self.b / self.ca  # arc radius [m]
        self.sloc = 2 * self.r * math.sin(self.ca / 2) / self.ca  # center to CG distance [m]
        self.miu = -7 * math.pi / 180  # installation angle [rad]

        # Aerodynamic coefficients (real scale)
        self.CD0 = 0.13
        self.CDa2 = 1.1
        self.CDds = 0.3 * 2
        self.CYbeta = -0.23 * 2
        self.CL0 = 0.50
        self.CLa = 3.5
        self.CLds = 0.21 * 2
        self.Clbeta = -0.037 * 2
        self.Clp = -0.08 * 2
        self.Clr = 0
        self.Clda = 0.001 * 2
        self.Cm0 = 0.1 * 2
        self.Cma = -0.02 * 2
        self.Cmq = -1 * 2
        self.Cnbeta = 0
        self.Cnp = 0
        self.Cnr = -0.07 * 2
        self.Cnda = -0.01 * 2
        self.CDp = 0.4

        # Coupling stiffness/damping (real scale)
        self.k_r = 1.0
        self.k_f = 0.1
        self.c_f = 120.0
        self.k_psi = 0.07

        # Wind and attachment points (real scale)
        self.vw = np.array([[0.0], [0.0], [0.0]])  # wind in inertial frame
        self.rcOc = np.array([[0.0], [0.0], [2.2]])  # connection to canopy CG
        self.rcOp = np.array([[0.0], [0.0], [-0.1]])  # connection to payload CG

        # Controls (to be set each step)
        self.left = 0.0
        self.right = 0.0
        self.thrust = np.array([[15.0], [0.0], [0.0]])  # default cruise thrust


class ParafoilEnv(gym.Env):
    """RL environment wrapping parafoil_model with RK4 integration.

    Action: 3-dim Box [-1,1] -> [throttle, left, right]
    Observation: tracking errors and selected states
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt_env: float = 0.04,
        n_substeps: int = 10,
        ref_type: str = "line",
        ref_speed: float = 12.0,   # m/s, achievable cruise speed with current thrust
        ref_alt: float = 230.0,   # m, realistic altitude from your data
        max_episode_steps: int = 2000,
        action_limits: Optional[Dict[str, float]] = None,
        actuator_tau: float = 0.15,
        # Wind configuration (hidden from observation by default)
        wind_mode: str = "none",  # none | const | ou | sin
        wind_mean_max: float = 5.0,  # max horizontal mean wind [m/s]
        gust_sigma: float = 1.0,     # OU noise std [m/s]
        gust_tau: float = 5.0,       # OU time constant [s]
        wind_include_in_obs: bool = False,  # if True, append wind to obs for debugging
        seed: Optional[int] = None,
        # Optional default init config (can be overridden per-reset via options={"init": ...})
        init_config: Optional[Dict[str, Any]] = None,
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

        # Wind config
        self.wind_mode = wind_mode
        self.wind_mean_max = wind_mean_max
        self.gust_sigma = gust_sigma
        self.gust_tau = gust_tau
        self.wind_include_in_obs = wind_include_in_obs
        self._wind_mean = np.zeros(3)
        self._wind_state = np.zeros(3)

        # Parameters instance used by the dynamics
        self.para = ParaParams()

        # Default initial condition template (overridden by reset options)
        self._default_init: Optional[Dict[str, Any]] = init_config

        # Action mapping scales (placeholders, configurable)
        lims = action_limits or {}
        self.thrust_max = lims.get("thrust_max", 10.0)  # N
        self.defl_max = lims.get("deflection_max", 0.5)  # non-dimensional or rad-equivalent
        self.defl_rate_max = lims.get("deflection_rate_max", 2.0)  # per second
        self.thrust_rate_max = lims.get("thrust_rate_max", 50.0)  # N/s

        # Gym spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # obs: [e_p(3), e_v(3), e_psi(1), w(3), thetar, psir, speed] (+ optional wind) + psi (last)
        high_obs = np.array([
            1e3, 1e3, 1e3,  # position error
            1e2, 1e2, 1e2,  # velocity error
            math.pi,         # yaw error
            50.0, 50.0, 50.0,  # angular rates
            math.pi, math.pi,  # thetar, psir
            1e2,              # speed
        ], dtype=np.float32)
        if self.wind_include_in_obs:
            high_obs = np.concatenate([high_obs, np.array([50.0, 50.0, 50.0], dtype=np.float32)])
        # Append absolute yaw psi at the end for trajectory tracking
        high_obs = np.concatenate([high_obs, np.array([math.pi], dtype=np.float32)])
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
            R = 120.0  # larger radius for gentler turning
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
        self._init_wind()

        # Build initial state from options/init_config or fallback to randomized defaults
        init = None
        if options is not None and isinstance(options, dict):
            init = options.get("init", None)
        if init is None:
            init = self._default_init

        if init is None:
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
        else:
            # Deterministic init from provided dict
            def arr3(x, default):
                a = np.array(x, dtype=float).reshape(-1) if x is not None else np.array(default, dtype=float)
                if a.size != 3:
                    raise ValueError("init fields must be 3-dim vectors")
                return a

            pos = arr3(init.get("pos", None), [0.0, 0.0, self.ref_alt])
            att = arr3(init.get("att", None), [0.0, 0.0, 0.0])  # phi, theta, psi
            v = arr3(init.get("v", None), [self.ref_speed, 0.0, 0.0])
            w = arr3(init.get("w", None), [0.0, 0.0, 0.0])
            vp = arr3(init.get("vp", None), v)
            ws = arr3(init.get("ws", None), [0.0, 0.0, 0.0])
            thetar = float(init.get("thetar", 0.0))
            psir = float(init.get("psir", 0.0))

            y = np.zeros(20)
            y[0:3] = pos
            y[3:6] = att
            y[6] = thetar
            y[7] = psir
            y[8:11] = v
            y[11:14] = w
            y[14:17] = vp
            y[17:20] = ws

            self.state = y
            self.t = 0.0
            self.steps = 0

            # Initial actuator commands (physical units)
            act = init.get("act", {}) if isinstance(init.get("act", {}), dict) else {}
            T0 = float(np.clip(act.get("throttle", 0.0), 0.0, self.thrust_max))
            L0 = float(np.clip(act.get("left", 0.0), -self.defl_max, self.defl_max))
            R0 = float(np.clip(act.get("right", 0.0), -self.defl_max, self.defl_max))
            self._last_cmd = np.array([T0, L0, R0], dtype=float)

        obs = self._get_obs()
        info = {}
        return obs, info

    # --- Init helpers ---
    def set_default_init(self, init: Dict[str, Any]) -> None:
        """Set a default initial condition used by future reset() calls unless overridden.

        init keys:
          - pos: [x,y,z]
          - att: [phi,theta,psi]
          - v: [vx,vy,vz]
          - w: [wx,wy,wz]
          - vp: [vpx,vpy,vpz]
          - ws: [wsx,wsy,wsz]
          - thetar: float
          - psir: float
          - act: {throttle: N, left: defl, right: defl}
        """
        self._default_init = init

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
            # Update wind
            self._update_wind(self.dt_sub)
            self.para.left = float(np.clip(self._last_cmd[1], -self.defl_max, self.defl_max))
            self.para.right = float(np.clip(self._last_cmd[2], -self.defl_max, self.defl_max))
            self.para.thrust = np.array([[self._last_cmd[0]], [0.0], [0.0]])
            # Set wind in inertial frame as column vector
            self.para.vw = np.array(self._wind_state.reshape(3, 1))

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

        obs_list = [
            e_p[0], e_p[1], e_p[2],
            e_v[0], e_v[1], e_v[2],
            e_psi,
            w[0], w[1], w[2],
            thetar, psir,
            speed,
        ]
        if self.wind_include_in_obs:
            obs_list += [self._wind_state[0], self._wind_state[1], self._wind_state[2]]
        # Always append absolute yaw psi as the last element
        obs_list += [psi]
        obs = np.array(obs_list, dtype=np.float32)
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
            print(f"🚁 TERMINATED: Ground impact at altitude {p[2]:.2f}m (step {self.steps})")
        if abs(phi) > math.radians(80) or abs(theta) > math.radians(80):
            terminated = True
            info["event"] = info.get("event", "attitude_limit")
            print(f"🚁 TERMINATED: Attitude limit exceeded - phi={math.degrees(phi):.1f}°, theta={math.degrees(theta):.1f}° (step {self.steps})")
        if not np.isfinite(self.state).all():
            terminated = True
            info["event"] = info.get("event", "numerical")
            print(f"🚁 TERMINATED: Numerical instability detected (step {self.steps})")
            print(f"   State: {self.state}")
        if speed < 2.0:  # more reasonable stall speed for parafoil
            terminated = True
            info["event"] = info.get("event", "stall")
            print(f"🚁 TERMINATED: Stall condition - speed={speed:.2f}m/s < 2.0m/s (step {self.steps})")
        if self.steps >= self.max_episode_steps:
            truncated = True
            print(f"✅ TRUNCATED: Maximum steps reached ({self.max_episode_steps} steps, {self.max_episode_steps*0.04:.1f}s simulation time)")

        return terminated, truncated, info

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    # --- Wind helpers ---
    def _init_wind(self):
        if self.wind_mode == "none":
            self._wind_mean = np.zeros(3)
        elif self.wind_mode == "const":
            spd = self._rng.uniform(0.0, self.wind_mean_max)
            ang = self._rng.uniform(-math.pi, math.pi)
            self._wind_mean = np.array([spd * math.cos(ang), spd * math.sin(ang), 0.0])
        elif self.wind_mode == "sin":
            spd = self._rng.uniform(0.0, self.wind_mean_max)
            ang = self._rng.uniform(-math.pi, math.pi)
            self._wind_mean = np.array([spd * math.cos(ang), spd * math.sin(ang), 0.0])
        else:  # ou
            spd = self._rng.uniform(0.0, self.wind_mean_max)
            ang = self._rng.uniform(-math.pi, math.pi)
            self._wind_mean = np.array([spd * math.cos(ang), spd * math.sin(ang), 0.0])
        self._wind_state = self._wind_mean.copy()

    def _update_wind(self, dt: float):
        if self.wind_mode == "none":
            self._wind_state = np.zeros(3)
        elif self.wind_mode == "const":
            self._wind_state = self._wind_mean
        elif self.wind_mode == "sin":
            # Slow sinusoidal variation around mean in horizontal plane
            t = self.t
            wfreq = 0.05  # Hz approx (rad/s ~ 0.3)
            amp = 0.5 * self.wind_mean_max
            dx = amp * math.sin(2 * math.pi * wfreq * t)
            dy = amp * math.cos(2 * math.pi * wfreq * t)
            self._wind_state = self._wind_mean + np.array([dx, dy, 0.0])
        else:  # ou gust
            theta = 1.0 / max(self.gust_tau, 1e-3)
            sigma = self.gust_sigma
            mu = self._wind_mean
            dW = theta * (mu - self._wind_state) * dt + sigma * math.sqrt(max(dt, 1e-6)) * self._rng.normal(0.0, 1.0, size=3)
            self._wind_state = self._wind_state + dW


def make_env(**kwargs) -> ParafoilEnv:
    return ParafoilEnv(**kwargs)
