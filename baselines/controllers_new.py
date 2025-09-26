import numpy as np


class ADRCYawSpeed:
    """ADRC inner-loop with trajectory-error guidance.

    外环：用 e_p/e_v 生成 yaw_cmd；
    内环：ADRC 近似 ESO 估计 yaw 误差动态并做 PD 型控制；另有速度 PI。
    """

    def __init__(self, speed_ref: float = 8.5, Ts: float = 0.04,
                 kp_pos: float = 0.03, kd_pos: float = 0.05,
                 # Symmetric brake parameters
                 kp_v: float = 0.4, ki_v: float = 0.05,
                 d_s_max: float = 0.4, yaw_reserve: float = 0.15,
                 speed_deadband: float = 0.2, defl_max: float = 1.0):
        self.Ts = Ts
        self.beta1 = 8.0
        self.beta2 = 80.0
        self.kp = 1.5
        self.kd = 0.4
        self.speed_ref = speed_ref
        self.kp_spd = 0.2
        self.ki_spd = 0.05
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos

        # Symmetric brake parameters
        self.kp_v = kp_v
        self.ki_v = ki_v
        self.d_s_max = min(d_s_max, defl_max - yaw_reserve)
        self.yaw_reserve = yaw_reserve
        self.speed_deadband = speed_deadband
        self.defl_max = defl_max

        # State variables
        self.z1 = 0.0
        self.z2 = 0.0
        self._ei_spd = 0.0
        self._ei_v = 0.0  # integral for symmetric brake
        self._d_s = 0.0   # current symmetric brake command

    def reset(self):
        self.z1 = 0.0
        self.z2 = 0.0
        self._ei_spd = 0.0
        self._ei_v = 0.0
        self._d_s = 0.0
        self._debug_count = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        # Extract needed signals
        e_p = obs[0:3].astype(float)       # position error
        e_v = obs[3:6].astype(float)       # velocity error
        e_psi = float(obs[6])              # yaw error from obs
        wz = float(obs[9])                 # yaw rate (w[2], canopy angular velocity z)
        speed = float(obs[12])             # speed magnitude
        psi = float(obs[-1])               # absolute yaw angle (always last element)

        # Strategy 1: Direct yaw error control (trajectory tangent following)
        # Use the pre-computed yaw error from observation (psi - yaw_ref)
        e_yaw = e_psi  # Use the yaw error directly from obs[6]

        # Position correction for off-track situations
        vcmd_xy = - self.kp_pos * e_p[0:2] - self.kd_pos * e_v[0:2]
        # Add cross-track error correction to yaw command
        if np.linalg.norm(e_p[0:2]) > 5.0:  # if more than 5m off track
            cross_track_yaw = float(np.arctan2(vcmd_xy[1], vcmd_xy[0]))
            cross_track_error = (psi - cross_track_yaw + np.pi) % (2 * np.pi) - np.pi
            e_yaw -= 0.3 * cross_track_error  # gentle correction towards track

        # Add simple altitude control via thrust modulation
        thrust_bias = 0  # altitude control disabled for now

        # ADRC-like ESO on yaw error signal
        y = e_yaw
        e = self.z1 - y
        self.z1 += self.Ts * (self.z2 - self.beta1 * e)
        self.z2 += self.Ts * (- self.beta2 * e)

        # Control law with rate damping (fixed sign for correct direction)
        u_yaw = self.kp * self.z1 + self.kd * self.z2 - 0.2 * wz
        u_yaw_raw = u_yaw
        u_yaw = float(np.clip(u_yaw, -1.0, 1.0))

        # Speed control with symmetric braking (new approach)
        e_spd_raw = self.speed_ref - speed
        # Apply deadband to prevent oscillation
        e_spd = e_spd_raw if abs(e_spd_raw) > self.speed_deadband else 0.0

        # Symmetric brake PI controller
        d_s_cmd = self.kp_v * e_spd + self.ki_v * self._ei_v
        d_s_cmd = np.clip(d_s_cmd, 0.0, self.d_s_max)

        # Anti-windup: pause integration if saturated and still pushing same direction
        if not (d_s_cmd >= self.d_s_max and e_spd > 0):
            self._ei_v += e_spd

        # Update symmetric brake state (could add rate limiting here)
        self._d_s = d_s_cmd

        # Thrust control (simplified - mainly for altitude/basic speed)
        e_spd_thr = self.speed_ref - speed
        self._ei_spd += e_spd_thr
        u_spd = self.kp_spd * e_spd_thr + self.ki_spd * self._ei_spd + thrust_bias
        # Optional: thrust compensation for symmetric brake
        k_comp = 0.1  # compensation factor
        u_spd += k_comp * self._d_s  # compensate for brake-induced drag
        a_thr = float(np.clip(u_spd, 0.2, 1.0))

        # Control allocation: baseline d_s + differential yaw
        u_yaw_pos = max(0.0, u_yaw)
        u_yaw_neg = max(0.0, -u_yaw)

        a_left = np.clip(self._d_s + u_yaw_neg, 0.0, self.defl_max)
        a_right = np.clip(self._d_s + u_yaw_pos, 0.0, self.defl_max)

        act = np.array([2.0 * a_thr - 1.0, a_left, a_right], dtype=np.float32)
        return act


class PIDController:
    """Trajectory-tracking PID via velocity-vector guidance with symmetric brake speed control.

    外环（制导）：用 e_p/e_v 生成期望地面速度向量并得到 yaw_cmd；
    内环：航向 PD 跟踪 yaw_cmd，对称刹车 + 推力调整速度。
    """

    def __init__(
        self,
        kp_pos: float = 0.03,
        kd_pos: float = 0.05,
        kp_yaw: float = 1.0,
        kd_yaw: float = 0.3,
        speed_ref: float = 8.5,
        kp_spd: float = 0.2,
        ki_spd: float = 0.05,
        # Symmetric brake parameters
        kp_v: float = 0.4,
        ki_v: float = 0.05,
        d_s_max: float = 0.4,
        yaw_reserve: float = 0.15,
        speed_deadband: float = 0.2,
        defl_max: float = 1.0,
    ):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.speed_ref = speed_ref
        self.kp_spd = kp_spd
        self.ki_spd = ki_spd

        # Symmetric brake parameters
        self.kp_v = kp_v
        self.ki_v = ki_v
        self.d_s_max = min(d_s_max, defl_max - yaw_reserve)
        self.yaw_reserve = yaw_reserve
        self.speed_deadband = speed_deadband
        self.defl_max = defl_max

        # State variables
        self._ei_spd = 0.0
        self._ei_v = 0.0  # integral for symmetric brake
        self._d_s = 0.0   # current symmetric brake command

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def reset(self):
        self._ei_spd = 0.0
        self._ei_v = 0.0
        self._d_s = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        # Extract errors and states
        e_p = obs[0:3].astype(float)       # position error
        e_v = obs[3:6].astype(float)       # velocity error
        e_psi = float(obs[6])              # yaw error from obs
        wz = float(obs[9])                 # yaw rate (w[2], canopy angular velocity z)
        speed = float(obs[12])             # speed magnitude
        psi = float(obs[-1])               # absolute yaw angle (always last element)

        # Strategy 1: Direct yaw error control (trajectory tangent following)
        # Use the pre-computed yaw error from observation (psi - yaw_ref)
        # This directly follows the trajectory tangent direction
        e_yaw = e_psi  # Use the yaw error directly from obs[6]
        # Fixed sign: positive e_yaw (left of reference) requires positive u_yaw (right turn)
        u_yaw = self.kp_yaw * e_yaw - self.kd_yaw * wz

        # Position correction for off-track situations
        vcmd_xy = - self.kp_pos * e_p[0:2] - self.kd_pos * e_v[0:2]
        # Add cross-track error correction to yaw command
        if np.linalg.norm(e_p[0:2]) > 5.0:  # if more than 5m off track
            cross_track_yaw = float(np.arctan2(vcmd_xy[1], vcmd_xy[0]))
            cross_track_error = self._wrap(psi - cross_track_yaw)
            u_yaw += 0.3 * cross_track_error  # gentle correction towards track

        # Add simple altitude control via thrust modulation
        thrust_bias = 0  # altitude control disabled for now
        u_yaw = float(np.clip(u_yaw, -1.0, 1.0))

        # Speed control with symmetric braking (new approach)
        e_spd_raw = self.speed_ref - speed
        # Apply deadband to prevent oscillation
        e_spd = e_spd_raw if abs(e_spd_raw) > self.speed_deadband else 0.0

        # Symmetric brake PI controller
        d_s_cmd = self.kp_v * e_spd + self.ki_v * self._ei_v
        d_s_cmd = np.clip(d_s_cmd, 0.0, self.d_s_max)

        # Anti-windup: pause integration if saturated and still pushing same direction
        if not (d_s_cmd >= self.d_s_max and e_spd > 0):
            self._ei_v += e_spd

        # Update symmetric brake state (could add rate limiting here)
        self._d_s = d_s_cmd

        # Thrust control (simplified - mainly for altitude/basic speed)
        e_spd_thr = self.speed_ref - speed
        self._ei_spd += e_spd_thr
        u_spd = self.kp_spd * e_spd_thr + self.ki_spd * self._ei_spd + thrust_bias
        # Optional: thrust compensation for symmetric brake
        k_comp = 0.1  # compensation factor
        u_spd += k_comp * self._d_s  # compensate for brake-induced drag
        a_thr = float(np.clip(u_spd, 0.2, 1.0))

        # Control allocation: baseline d_s + differential yaw
        u_yaw_pos = max(0.0, u_yaw)
        u_yaw_neg = max(0.0, -u_yaw)

        a_left = np.clip(self._d_s + u_yaw_neg, 0.0, self.defl_max)
        a_right = np.clip(self._d_s + u_yaw_pos, 0.0, self.defl_max)

        act = np.array([2.0 * a_thr - 1.0, a_left, a_right], dtype=np.float32)
        return act