import numpy as np

def w2wx(omega):
    w = np.mat([[0, -omega[2][0], omega[1][0]], [omega[2][0], 0, -omega[0][0]],
                [-omega[1][0], omega[0][0], 0]])
    return w

