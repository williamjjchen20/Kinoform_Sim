import numpy as np
import scipy.constants as const
import xraylib as xrl

import matplotlib.pyplot as plt

JOULE_TO_EV = 1/const.e
M_TO_ANG = 1e10

def snell(alpha, n1, n2):
    alpha_p = np.arcsin(n1/n2 * np.sin(alpha))
    return alpha_p

def reflectivity(Q0, Q1, D):
    r01 = (Q0-Q1)/(Q0+Q1)
    p = np.exp(1j*Q1*D/2)
    r_slab = (r01*(1-p**2))/(1-r01**2*p**2) 
    return r_slab   
    
if __name__ == "__main__":
    plt.figure()
    ## Introduction to Modern X-ray Physics (Ch. 3)
    wavelength = 1.54051e-10
    E = const.h * const.c/wavelength  # keV
    k = 2*const.pi/(wavelength*M_TO_ANG)
    D = 10*2*const.pi
    print(k)
    n1 = 1.
    n2 = xrl.Refractive_Index("W", E/1000*JOULE_TO_EV, 19.3)
    
    a = np.linspace(0, const.pi/2, 10000) # incident angles

    Qc = 0.081 # A^-1
    Q = 2*k*np.sin(a)
    Qp = np.sqrt(Q**2 - Qc**2 + 0j)

    r_slab = reflectivity(Q, Qp, D)

    plt.plot(Q, np.abs(r_slab)**2)
    plt.yscale("log")
    plt.gca().set(xlim=(0, 1), ylim=(1e-10, 1))
    plt.show()
    # q = Q/Qc
    # qp = Qp/Qc
    # print(q, qp)
    # # quit()
    # r_slab = reflectivity(q, qp, D)

    # # r_slab = (Qc/(2*Q))**2*(1-np.exp(1j*Q*D))
    # plt.plot(Q, np.abs(r_slab)**2)
    # plt.yscale("log")
    # plt.gca().set(xlim=(1e-10, 1-1e-10), ylim=(1e-10, 1))
    # plt.show()
    
    
    
    