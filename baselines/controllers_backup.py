import numpy as np


class ADRCYawSpeed:
    """ADRC inner-loop with trajectory-error guidance.

    外环：用 e_p/e_v 生成 yaw_cmd；
    内环：ADRC 近似 ESO 估计 yaw 误差动态并做 PD 型控制；另有速度 PI。
    """

    def __init__(self, speed_ref: float = 8.5, Ts: float = 0.04,
                 kp_pos: float = 0.03, kd_pos: float = 0.05):
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
        self.z1 = 0.0
        self.z2 = 0.0
        self._ei_spd = 0.0

    def reset(self):
        self.z1 = 0.0
        self.z2 = 0.0
        self._ei_spd = 0.0
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
        alt_error = e_p[2]  # positive if above reference
        thrust_bias = -0.4 * alt_error  # reduce thrust if too high

        # ADRC-like ESO on yaw error signal
        y = e_yaw
        e = self.z1 - y
        self.z1 += self.Ts * (self.z2 - self.beta1 * e)
        self.z2 += self.Ts * (- self.beta2 * e)

        # Control law with rate damping (fixed sign for correct direction)
        u_yaw = self.kp * self.z1 + self.kd * self.z2 - 0.2 * wz
        u_yaw_raw = u_yaw
        u_yaw = float(np.clip(u_yaw, -1.0, 1.0))
        
        # Speed control: combine thrust and symmetric braking
        e_spd = self.speed_ref - speed
        self._ei_spd += e_spd
        speed_term = self.kp_spd * e_spd + self.ki_spd * self._ei_spd
        u_spd = speed_term + thrust_bias

        # Split speed control between thrust and symmetric brake
        if u_spd >= 0:  # speed up: use thrust, minimal braking
            a_thr = float(np.clip(u_spd, 0.2, 1.0))  # Minimum 20% thrust to avoid stall
            symmetric_brake = 0.0
        else:  # slow down: reduce thrust and apply symmetric brake
            a_thr = 0.2  # minimum thrust to maintain control
            symmetric_brake = float(np.clip(-u_spd * 0.5, 0.0, 0.6))  # max 60% symmetric brake

        # Control allocation: differential + symmetric braking
        # Base symmetric brake for speed control
        base_brake = symmetric_brake

        # Add differential component for yaw control
        if u_yaw > 0:  # turn right: pull right brake more
            a_left = base_brake
            a_right = base_brake + np.clip(u_yaw, 0.0, 1.0 - base_brake)
        else:  # turn left: pull left brake more
            a_left = base_brake + np.clip(-u_yaw, 0.0, 1.0 - base_brake)
            a_right = base_brake
        act = np.array([2.0 * a_thr - 1.0, a_left, a_right], dtype=np.float32)
        return act


class PIDController:
    """Trajectory-tracking PID via velocity-vector guidance.

    外环（制导）：用 e_p/e_v 生成期望地面速度向量并得到 yaw_cmd；
    内环：航向 PD 跟踪 yaw_cmd，速度 PI 调整油门。
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
    ):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.speed_ref = speed_ref
        self.kp_spd = kp_spd
        self.ki_spd = ki_spd
        self._ei_spd = 0.0

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def reset(self):
        self._ei_spd = 0.0

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
        alt_error = e_p[2]  # positive if above reference
        thrust_bias =  0 #-0.4 * alt_error  # reduce thrust if too high
        u_yaw = float(np.clip(u_yaw, -1.0, 1.0))

        # Speed control: combine thrust and symmetric braking
        # Option A: constant speed_ref (simple, robust)
        e_spd = self.speed_ref - speed
        self._ei_spd += e_spd
        u_spd = self.kp_spd * e_spd + self.ki_spd * self._ei_spd + thrust_bias

        # Split speed control between thrust and symmetric brake
        if u_spd >= 0:  # speed up: use thrust, minimal braking
            a_thr = float(np.clip(u_spd, 0.2, 1.0))  # Minimum 20% thrust to avoid stall
            symmetric_brake = 0.0
        else:  # slow down: reduce thrust and apply symmetric brake
            a_thr = 0.2  # minimum thrust to maintain control
            symmetric_brake = float(np.clip(-u_spd * 0.5, 0.0, 0.6))  # max 60% symmetric brake

        # Control allocation: differential + symmetric braking
        # Base symmetric brake for speed control
        base_brake = symmetric_brake

        # Add differential component for yaw control
        if u_yaw > 0:  # turn right: pull right brake more
            a_left = base_brake
            a_right = base_brake + np.clip(u_yaw, 0.0, 1.0 - base_brake)
        else:  # turn left: pull left brake more
            a_left = base_brake + np.clip(-u_yaw, 0.0, 1.0 - base_brake)
            a_right = base_brake
        act = np.array([2.0 * a_thr - 1.0, a_left, a_right], dtype=np.float32)
        return act
