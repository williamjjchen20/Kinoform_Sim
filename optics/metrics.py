import numpy as np
import xraylib as xrl
import scipy.integrate as integrate

from .classes import Waveform

def FWHM(wave: Waveform):
    '''
    Assumes radial invariance
    args
    - wave: waveform object
    
    return
    - d: full width half maximum [m]
    '''
    I = wave.intensity()
    
    I_max = np.max(I)
    I_halfmax = I_max/2
    
    val_idx = np.argmin(np.abs(I-I_halfmax))
    idx = np.unravel_index(val_idx, I.shape)
    i1 = np.where(np.ravel(I) == I[idx])[0][0]
    i1 = np.unravel_index(i1, I.shape)
    i2 = np.unravel_index(np.argmax(I), I.shape)
    
    if wave.dim == 1:
        X = wave.grid
        r1 = X[i1]
        r2 = X[i2]
        r = np.abs(r1-r2)
    else:
        X, Y = wave.grid
        r1 = np.array([X[i1], Y[i1]])
        r2 = np.array([X[i2], Y[i2]])
        r = np.linalg.norm(r1-r2)
    d = 2*r
    return d

def intensity_stats(wave: Waveform):
    '''
    args
    - wave: waveform object
    
    return
    - I_max: maximum intensity
    - I_avg: average intensity
    '''
    I = wave.intensity()
    I_max = np.max(I)
    
    mask = I != 0
    I_avg = np.mean(I[mask])
    
    return I_max, I_avg

def total_power(wave: Waveform):
    I = wave.intensity()
    if wave.dim == 1:
        X = wave.grid
        P = integrate.simpson(I, x=X)
    else:
        X, Y = wave.grid
        tmp = integrate.simpson(I, X[0,:], axis=-1)
        P = integrate.simpson(tmp, Y[:,0])
        
    return P

def focusing_efficiency(wave1: Waveform, wave2: Waveform):
    '''
    args
    - wave1: incident plane waveform
    - wave2: focal plane waveform
    '''
    
    P_in = total_power(wave1)
    P_focal = total_power(wave2)
    
    return P_focal/P_in
    
def strehl_ratio(wave1: Waveform, wave2: Waveform):
    '''
    args 
    - wave1: waveform through ideal lens
    - wave2: waveform through aberrated lens
    
    '''
    
    I1 = wave1.intensity()
    I2 = wave2.intensity()
    return I2/I1

