import numpy as np
import xraylib as xray

def snell(alpha: float, n1, n2) -> float:
    alpha_p = np.arcsin(n1/n2 * np.sin(alpha))
    return alpha_p

def reflectivity():
    pass