"""
Functions for post-processing analysis of trajectory data
"""

import numpy as np
from scipy.integrate import cumtrapz

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

def angle_of_cluster(trajectory, wrap=False, initial=0, N_cutoff=1):
    """Determine the angle in time of a cluster of particles that is approximately a rigid structure
    
    Arguments:
        trajectory[T,N,D]        trajectory of N particles of length T (dimension D > 1)
        wrap       (bool) wrap the angles between -π and +π (default: False)
        initial    initial angle of the cluster (default: 0)
        N_cutoff   number of particles closest to the center of mass to ignore (default: 1)
    """
    Tsteps = trajectory.shape[0]
    
    ### obtain the trajectory in 2 dimensions centered around the center of mass
    com = np.average(trajectory[...,:2], axis=1)
    traj_2d = trajectory[...,:2] - com[:,np.newaxis]

    ### calculate the distance to center of mass for each particle at each time-step
    dist_to_com = np.linalg.norm(traj_2d, axis=-1)

    ### obtain 'trackers': particle indices that are furthert from the center of mass at time t
    trackers = np.argsort(dist_to_com, axis=-1)[:,N_cutoff:]

    ### calculate differential angle change of trackers at time t
    pos = np.array([traj_2d[i,trackers[i]] for i in range(Tsteps-1)])
    pos_next = np.array([traj_2d[i+1,trackers[i]] for i in range(Tsteps-1)])

    # phi = np.arctan2(pos[...,1], pos[...,0])
    # phi_next = np.arctan2(pos_next[...,1], pos_next[...,0])
    # dphi = phi_next - phi
    # dphi = np.sign(dphi) * (np.abs(dphi) % np.pi)

    dr = pos_next - pos
    dphi = np.cross(pos, dr, axis=-1)/np.linalg.norm(pos, axis=-1)**2

    ### the angle of the cluster is then an integrated average of dphi
    dphi_avg = np.average(dphi, axis=1)    # average of trackers
    phi_cluster = cumtrapz(dphi_avg, initial=initial)    # angle integral of differential angles
    phi_cluster = np.insert(phi_cluster, 0, 0, axis=0)

    if not wrap:
        phi_cluster = np.unwrap(phi_cluster)

    # from IPython import embed; embed()

    return phi_cluster
