import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, frequency, order):
    """
    Create a Butterworth bandpass filter.
    
    Parameters:
    -----------
    lowcut : float
        Lower cutoff frequency in Hz
    highcut : float
        Upper cutoff frequency in Hz
    frequency : float
        Sampling rate in Hz
    order : int
        Filter order 
        
    Returns:
    --------
    b, a : ndarray
        Numerator (b) and denominator (a) polynomials of the filter
    """
    nyquist_freq = 0.5 * frequency
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_butter_bandpass_filter(data, frequency, lowcut=0.5, highcut=40, order=4):
    """
    Apply a Butterworth bandpass filter to the data.
    Uses filtfilt for zero-phase filtering to preserve the timing of waves and peaks.
    
    Parameters:
    -----------
    data : array_like
        The input signal to be filtered
    frequency : float
        Sampling rate in Hz
    lowcut : float
        Lower cutoff frequency in Hz
    highcut : float
        Upper cutoff frequency in Hz
    order : int
        Filter order (higher = more aggressive filtering)
        
    Returns:
    --------
    y : ndarray
        The filtered output with the same shape as data
    """
    b, a = butter_bandpass(lowcut, highcut, frequency, order)
    if data.ndim == 2:
        n_samples, n_leads = data.shape
        filtered = np.zeros_like(data) # create empty array to hold filtered data
        for lead_idx in range(n_leads): # filter each lead separately
            filtered[:, lead_idx] = filtfilt(b, a, data[:, lead_idx])
        return filtered
    elif data.ndim == 1:
        return filtfilt(b, a, data)
    else:
        raise ValueError("Data must be 1D or 2D array.")
    