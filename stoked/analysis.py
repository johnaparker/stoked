"""
Functions for post-processing analysis of trajectory data
"""

import numpy as np

def autocorrelation_fft(x):
    """
    compute the autocorrelation of x using the FFT
    """
    N = len(x)
    F = np.fft.fft(x, n=2*N)
    PSD = F*F.conjugate()
    res = np.fft.ifft(PSD)
    res = res[:N].real
    n = N*np.ones(N) - np.arange(N)

    return res/n

def msd_single(trajectory, fft=True):
    """
    Compute the mean-squared displacement (MSD) of a single trajectory

    Arguments:
        trajectory    shape [T,D] for a single trajectory of T steps in D dimensions
        fft           use the FFT to compute the MSD (default: True)
    """
    if fft:
        N, dim = trajectory.shape

        D = np.sum(np.square(trajectory), axis=1)
        D = np.append(D, 0)
        S2 = sum([autocorrelation_fft(trajectory[:,i]) for i in range(dim)])

        Q = 2*np.sum(D)
        S1 = np.zeros(N, dtype=float)
        for m in range(N):
            Q = Q - D[m-1] - D[N-m]
            S1[m] = Q/(N-m)

        return S1 - 2*S2
    else:
        Nsteps = len(trajectory)
        lags = np.arange(Nsteps)
        msd_vals = np.zeros(Nsteps, dtype=float)    

        for i, lag in enumerate(lags):
            dr = trajectory[:-lag if lag else None] - trajectory[lag:]
            dr2 = np.square(dr).sum(axis=1)
            msd_vals[i] = dr2.mean()

        return msd_vals

def msd(trajectory, fft=True):
    """
    Compute the mean-squared displacement (MSD) of a trajectory

    Arguments:
        trajectory    shape [T,D] or [T,N,D] for a trajectory of T steps, N particles in D dimensions
        fft           use the FFT to compute the MSD (default: True)
    """
    if trajectory.ndim == 2:
        return msd_single(trajectory, fft)

    elif trajectory.ndim == 3:
        T, N, D = trajectory.shape
        ret = np.zeros([T,N], dtype=float)
        for i in range(N):
            ret[:,i] = msd_single(trajectory[:,i], fft)
        return ret

    else:
        raise ValueError('trajectory does not have the appropriate shape (ndim needs to be 2 or 3)')
