"""Various wrappers and/or implementations of window functions."""
import torch
from torch.special import expit

__all__ = ['hann', 'tukey', 'planck']

def get_window(window, **kwargs):
    """Get a window function by name."""
    return globals()[window](**kwargs)

########################
# Helper functions 
########################
def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1

def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return M + 1, True
    else:
        return M, False
    
def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    if needed:
        return w[:-1]
    else:
        return w


########################
# window functions
########################
class HANN():
    """wrapper for torch.hann_window."""
    def __call__(self, **kwargs):
        return torch.hann_window(**kwargs)
hann = HANN()


def tukey(M=2048, alpha = 0.5, sym = True, device='cpu'):
    """Implementation of Tukey window function
    (adapted from the Scipy library).
    
    Args:
    -----
        M: int
            Number of samples in the window. If zero or less, an empty
            tensor is returned.
        alpha: float
            Shape parameter of the Tukey window, representing the fraction
            of the window inside the cosine tapered region. If zero, the
            Tukey window is equivalent to a rectangular window. If one, the
            Tukey window is equivalent to a Hann window.
        sym: bool
            When True generates a symmetric window for filter design, while
            False generates a periodic window for spectral analysis. 
            (Default: True)
            
    Returns:
    --------
        A tensor containing the Tukey window, with the same size as M.
        
    Note:
    -----
        See scipy.signal.windows.tukey for more details.
        
    """
    
    if _len_guards(M):
        return torch.ones(M, device=device)

    if alpha <= 0:
        return torch.ones(M, device=device)
    elif alpha >= 1.0:
        return torch.hann_window(M, periodic=False, device=device)

    M, needs_trunc = _extend(M, sym)

    n = torch.arange(0, M, device = device)
    width = int((alpha*(M-1)/2.0))
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]

    w1 = 0.5 * (1 + torch.cos(torch.pi * (-1 + 2.0*n1/alpha/(M-1))))
    w2 = torch.ones(n2.shape, device = device)
    w3 = 0.5 * (1 + torch.cos(torch.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))

    window = torch.cat([w1, w2, w3])
    
    return _truncate(window, needs_trunc)

def planck(N, nleft=0, nright=0, device='cpu'):
    """
    Returns a Planck-taper window.
    
    Args:
    -----
        N (int)      : Number of samples in the window.
        nleft (int)  : Number of samples to taper on the left side  (should be < N/2).
        nright (int) : Number of samples to taper on the right side (should be < N/2).
        device (str) : Device to store the window tensor. (Default: 'cpu')
    
    Returns:
    --------
        out (tensor) : A tensor containing the Planck window, with the same size as N.
    """
    
    out = torch.ones(N, device=device)
    if nleft:
        out[0] *= 0
        zleft = torch.tensor([nleft * (1./k + 1./(k-nleft))
                            for k in range(1, nleft)]).to(device)
        out[1:nleft] *= expit(-zleft)
    if nright:
        out[N-1] *= 0
        zright = torch.tensor([-nright * (1./(k-nright) + 1./k)
                                for k in range(1, nright)]).to(device)
        out[N-nright:N-1] *= expit(-zright)
        
    return out
