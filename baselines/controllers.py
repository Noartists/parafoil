import numpy as np


class PIDController:
    def __init__(self, speed_ref: float = 8.0):
        self.kp_yaw = 1.0
        self.kd_yaw = 0.2
        self.kp_spd = 0.2
        self.ki_spd = 0.05
        self.speed_ref = speed_ref
        self._ei_spd = 0.0

    def reset(self):
        self._ei_spd = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        e_psi = float(obs[6])
        wz = float(obs[9])
        speed = float(obs[12])

        a_diff = - (self.kp_yaw * e_psi + self.kd_yaw * wz)
        a_diff = float(np.clip(a_diff, -1.0, 1.0))

        e_spd = self.speed_ref - speed
        self._ei_spd += e_spd
        u_spd = self.kp_spd * e_spd + self.ki_spd * self._ei_spd
        a_thr = float(np.clip(u_spd, 0.0, 1.0))

        a_left = np.clip(+a_diff, -1.0, 1.0)
        a_right = np.clip(-a_diff, -1.0, 1.0)
        act = np.array([2.0 * a_thr - 1.0, a_left, a_right], dtype=np.float32)
        return act


class ADRCYawSpeed:
    def __init__(self, speed_ref: float = 8.0, Ts: float = 0.04):
        self.Ts = Ts
        self.beta1 = 30.0
        self.beta2 = 300.0
        self.kp = 1.2
        self.kd = 0.3
        self.speed_ref = speed_ref
        self.kp_spd = 0.25
        self.ki_spd = 0.05
        self.z1 = 0.0
        self.z2 = 0.0
        self._ei_spd = 0.0

    def reset(self):
        self.z1 = 0.0
        self.z2 = 0.0
        self._ei_spd = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        e_psi = float(obs[6])
        wz = float(obs[9])
        speed = float(obs[12])

        y = e_psi
        e = self.z1 - y
        self.z1 += self.Ts * (self.z2 - self.beta1 * e)
        self.z2 += self.Ts * (- self.beta2 * e)

        u_yaw = - (self.kp * self.z1 + self.kd * self.z2 + 0.2 * wz)
        u_yaw = float(np.clip(u_yaw, -1.0, 1.0))

        e_spd = self.speed_ref - speed
        self._ei_spd += e_spd
        u_spd = self.kp_spd * e_spd + self.ki_spd * self._ei_spd
        a_thr = float(np.clip(u_spd, 0.0, 1.0))

        a_left = np.clip(+u_yaw, -1.0, 1.0)
        a_right = np.clip(-u_yaw, -1.0, 1.0)
        act = np.array([2.0 * a_thr - 1.0, a_left, a_right], dtype=np.float32)
        return act

