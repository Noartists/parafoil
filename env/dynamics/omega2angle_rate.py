import math
import numpy as np

def omega2angle_rate(psi, theta, zeta):
    x = np.mat([[
        1,
        math.sin(zeta) * math.tan(theta),
        math.cos(zeta) * math.tan(theta)
    ], [0, math.cos(zeta), -math.sin(zeta)],
                [
                    0,
                    math.sin(zeta) / math.cos(theta),
                    math.cos(zeta) / math.cos(theta)
                ]])
    return x

