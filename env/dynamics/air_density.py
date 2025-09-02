import math

def air_density(h):
    rho0 = 1.225
    T0 = 288.15
    if h <= 11000:
        T = T0 - 0.0065 * h
        return rho0 * math.pow((T / T0), 4.25588)
    elif ((h > 11000) and (h <= 20000)):
        T = 216.65
        return 0.36392 * math.exp((-h + 11000) / 6341.62)
    else:
        T = 216.65 + 0.001 * (h - 20000)
        return 0.088035 * math.pow((T / 216.65), -35.1632)

