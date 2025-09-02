import numpy as np


def compute_reward(obs: np.ndarray, action: np.ndarray) -> float:
    e_p = float(np.linalg.norm(obs[0:3]))
    e_v = float(np.linalg.norm(obs[3:6]))
    e_psi = abs(float(obs[6]))
    w_norm = float(np.linalg.norm(obs[7:10]))
    thetar = abs(float(obs[10]))
    psir = abs(float(obs[11]))
    a_reg = float(np.linalg.norm(action))
    da_reg = float(np.linalg.norm(action - np.clip(action, -0.9, 0.9)))

    r = 0.0
    r += -1.0 * e_p
    r += -0.5 * e_v
    r += -0.3 * e_psi
    r += -0.05 * w_norm
    r += -0.02 * (thetar + psir)
    r += -0.01 * a_reg
    r += -0.01 * da_reg
    r += 0.02
    return float(r)

