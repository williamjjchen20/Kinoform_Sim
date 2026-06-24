import numpy as np
import xraylib as xrl
import scipy.integrate as integrate

from ..classes import Waveform

def FWHM(wave: Waveform):
    """
    Compute FWHM assuming radial symmetry.
    Extracts a 1D radial profile through the peak and interpolates.
    """
    I = wave.intensity()
    I_max = np.max(I)
    I_half = I_max / 2

    if wave.dim == 1:
        X = wave.grid
        peak_idx = np.argmax(I)
        x_peak = X[peak_idx]

        # Take the right side of the peak (or left, doesn't matter for symmetric)
        right = I[peak_idx:]
        x_right = X[peak_idx:]

        # Interpolate to find half-max crossing
        # Find first crossing below half-max
        cross_idx = np.where(right <= I_half)[0]
        if len(cross_idx) == 0:
            return np.nan
        j = cross_idx[0]

        # Linear interpolation between j-1 and j
        r_half = np.interp(I_half, 
                           [right[j], right[j-1]],  # reversed: decreasing I
                           [x_right[j], x_right[j-1]])
        return 2 * abs(r_half - x_peak)

    else:
        X, Y = wave.grid
        peak_idx = np.unravel_index(np.argmax(I), I.shape)
        x_peak = X[peak_idx]
        y_peak = Y[peak_idx]

        # Extract a 1D slice through the peak (e.g., horizontal)
        row = I[peak_idx[0], :]
        x_row = X[peak_idx[0], :]

        # Right side of peak
        j_peak = peak_idx[1]
        right = row[j_peak:]
        x_right = x_row[j_peak:]

        cross_idx = np.where(right <= I_half)[0]
        if len(cross_idx) == 0:
            return np.nan
        j = cross_idx[0]

        r_half = np.interp(I_half,
                           [right[j], right[j-1]],
                           [x_right[j], x_right[j-1]])
        return 2 * abs(r_half - x_right[0])

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
        tmp = integrate.simpson(I, x=X[0,:], axis=-1)
        P = integrate.simpson(tmp, x=Y[:,0])
        
    return P

def focal_power(wave: Waveform, radius):
    """
    Integrate only within the first-order Airy disk.
    
    args:
    - wave: waveform
    - radius: radius [m] in which to calculate power over
    
    """
    
    I = wave.intensity()
    if wave.dim == 1:
        X = wave.grid
        
        xc = np.argmax(I)
        R = np.abs(X-xc)
        
        I_masked = np.where(R <= radius, I, 0.0)
        
        P_focal = integrate.simpson(I_masked, x=X)
    else:
        X, Y = wave.grid
    
        peak = np.unravel_index(np.argmax(I), I.shape)
        xc, yc = X[peak], Y[peak]
        
        R2 = (X - xc)**2 + (Y - yc)**2
        I_masked = np.where(R2 <= radius**2, I, 0.0)
 
        tmp = integrate.simpson(I_masked, x=X[0,:], axis=-1)
        P_focal = integrate.simpson(tmp, x=Y[:,0])
        
    return P_focal

def focal_efficiency(P_in, wave_out: Waveform, radius=None):
    '''
    args
    - wave_in: incident plane waveform
    - wave_out: focal plane waveform
    - radius: radius [m] in which to calculate power over
    '''
    if radius is None:
        P_focal = total_power(wave_out)
    else:
        P_focal = focal_power(wave_out, radius)
    
    return P_focal/P_in

    
def strehl_ratio(wave_in: Waveform, wave_out: Waveform):
    '''
    args 
    - wave_in: waveform through ideal lens
    - wave_out: waveform through aberrated lens
    
    '''
    
    I1 = np.max(wave_in.intensity())
    I2 = np.max(wave_out.intensity())
    return I2/I1