"""
This scripts implements PyTorch resampling based on Fourier method as in scipy.signal.resample.
The code is adapted from https://github.com/soumickmj/CodeSnippets
"""

import sys
import torch
import torch.fft



def resample(x, num, axis=-1):
    """
    Resample `x` to `num` samples using Fourier method along the given axis.
    
    NB: This function assumes x is real and in time domain.

    The resampled signal starts at the same value as `x` but is sampled
    with a spacing of ``len(x) / num * (spacing of x)``.  Because a
    Fourier method is used, the signal is assumed to be periodic.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    num : int or array_like
        The number of samples in the resampled signal. 
        If array_like is supplied, then the resample function will be 
        called recursively for each element of num.
    t : array_like, optional
        If `t` is given, it is assumed to be the equally spaced sample
        positions associated with the signal data in `x`.
    axis : (int, optional) or (array_like)
        The axis of `x` that is resampled.  Default is 0.
        If num is array_like, then axis has to be supplied and has to be array_like.
        Each element of axis should have one-on-on mapping wtih num.
        If num is int but axis is array_like, then num will be repeated and will be
        made a list with same number of elements as axis. Then will proceed both as array_like.

    Returns
    -------
    resampled_x or (resampled_x, resampled_t)
        Either the resampled array, or, if `t` was given, a tuple
        containing the resampled array and the corresponding resampled
        positions.
    """

    #number of data points
    Nx = x.shape[axis]

    #compute data real fft
    X = torch.fft.rfft(x, dim=axis)        
    
    # Copy each half of the original spectrum to the output spectrum, either
    # truncating high frequences (downsampling) or zero-padding them
    # (upsampling)

    # Placeholder array for output spectrum
    newshape = list(x.shape)
    newshape[axis] = num // 2 + 1


    # Copy positive frequency components (and Nyquist, if present)
    N = min(num, Nx)
    nyq = N // 2 + 1  # Slice index that includes Nyquist if present
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y = X[sl]
    
    
    # Split/join Nyquist component(s) if present
    # So far we have set Y[+N/2]=X[+N/2]
    if N % 2 == 0:
        if num < Nx:  # downsampling
            sl[axis] = slice(N//2, N//2 + 1)
            Y[sl] *= 2.
            
        elif Nx < num:  # upsampling
            # select the component at frequency +N/2 and halve it
            sl[axis] = slice(N//2, N//2 + 1)
            Y[sl] *= 0.5
            
    # Inverse transform
    y = torch.fft.irfft(Y, num, dim=axis) * (num/Nx)
    
    return y
    
