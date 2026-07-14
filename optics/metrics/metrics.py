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
        X = wave.grid*wave.Rx
        peak_idx = np.argmax(I)
        x_peak = X[peak_idx]

        # Take the right side of the peak (or left, doesn't matter for symmetric)
        right = I[peak_idx:]
        x_right = X[peak_idx:]
    else:
        X, Y = wave.grid
        X, Y = X*wave.Rx, Y*wave.Ry
        peak_idx = np.unravel_index(np.argmax(I), I.shape)
        x_peak = X[peak_idx]

        # Extract a 1D horizontal slice through the peak
        row = I[peak_idx[0], :]
        x_row = X[peak_idx[0], :]

        j_peak = peak_idx[1]
        right = row[j_peak:]
        x_right = x_row[j_peak:]

    # Find first crossing at/below half-max
    cross_idx = np.where(right <= I_half)[0]
    if len(cross_idx) == 0 or cross_idx[0] == 0:
        return np.nan
    j = cross_idx[0]

    # Linear interpolation between samples j-1 (above) and j (below) half-max.
    # `right` is decreasing here, so we can't use np.interp directly.
    y0, y1 = right[j-1], right[j]
    x0, x1 = x_right[j-1], x_right[j]
    r_half = x0 + (I_half - y0) * (x1 - x0) / (y1 - y0)
    return 2 * abs(r_half - x_peak)

def max_intensity(wave: Waveform):
    '''
    args
    - wave: waveform object
    
    return
    - I_max: maximum intensity
    '''
    I = wave.intensity()
    I_max = np.max(I)
    
    return I_max

def mean_intensity(wave: Waveform):
    '''
    args
    - wave: waveform object
    
    return
    - I_max: maximum intensity
    '''
    
    I = wave.intensity()
    I_mean = np.mean(I)
    
    return I_mean
    

def total_power(wave: Waveform):
    I = wave.intensity()
    if wave.dim == 1:
        X = wave.grid*wave.Rx
        P = integrate.simpson(I, x=X)
    else:
        X, Y = wave.grid
        X, Y = X*wave.Rx, Y*wave.Ry
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
        X = wave.grid*wave.Rx
        
        peak = np.argmax(I)
        xc = X[peak]
        R = np.abs(X-xc)
        
        I_masked = np.where(R <= radius, I, 0.0)
        
        P_focal = integrate.simpson(I_masked, x=X)
    else:
        X, Y = wave.grid
        X, Y = X*wave.Rx, Y*wave.Ry
    
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