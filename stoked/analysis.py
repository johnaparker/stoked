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

    return phi_cluster

def transform_to_lattice(trajectory, lattice):
    """
    Returns a new trajectory that is translated and rotated to be best aligned with a given fixed lattice
    
    Arguments:
        trajectory[T,N,2]      trajecrory data with T steps, N particles in 2 dimensions
        lattice[N,2]           lattice to align to
    """
    Nsteps, Nparticles, _ = trajectory.shape

    translation = np.average(trajectory[...,:2], axis=1)
    P = trajectory - translation[:,np.newaxis]
    Q = lattice[:,:2]

    H = np.einsum('TNi,Nj->Tij', P, Q)
    U, S, V = np.linalg.svd(H)
    R = np.einsum('Tij,Tjk->Tik', V, U)

    d = np.linalg.det(R)

    new_trajectory = np.einsum('Tij,TNj->TNi', R, P)

def transform_to_lattice_2d(trajectory, lattice):
    """
    Returns a new trajectory that is translated and rotated to be best aligned with a given fixed lattice in 2 dimensions
    
    Arguments:
        trajectory[T,N,2]      trajecrory data with T steps, N particles in 2 dimensions
        lattice[N,2]           lattice to align to
    """
    Nsteps, Nparticles, _ = trajectory.shape

    z1 = trajectory[...,0] + 1j*trajectory[...,1]
    z2 = lattice[...,0] + 1j*lattice[...,1]

    az1 = np.average(z1, axis=1)
    az2 = np.average(z2, axis=0)

    z = az2 - az1
    translation = np.array((z.real, z.imag)).T

    z1 -= az1[:,np.newaxis]
    z2 -= az2
    a = np.linalg.norm(z1, axis=1)**2

    err = np.linalg.norm(z2)**2
    zz1 = np.einsum('TN,N->T', z1, z2)
    zz_1 = np.einsum('TN,N->T', z1, np.conj(z2))

    b1 = np.abs(zz1)
    b_1 = np.abs(zz_1)
    err1 = -b1**2/a
    err_1 = -b_1**2/a

    idx = err1 <= err_1

    mirr = np.empty(Nsteps, dtype=float)
    sc = np.empty(Nsteps, dtype=complex)
    theta = np.empty(Nsteps, dtype=float)
    error = err*np.ones(Nsteps, dtype=float)
    rot = err*np.ones(Nsteps, dtype=complex)

    mirr[idx] = 1
    error[idx] += err1[idx]
    sc[idx] = b1[idx]/a[idx]
    theta[idx] = -np.angle(zz1[idx])
    rot[idx] = zz1[idx]

    mirr[~idx] = -1
    error[~idx] += err_1[~idx]
    sc[~idx] = b_1[~idx]/a[~idx]
    theta[~idx] = -np.angle(zz_1[~idx])
    rot[~idx] = zz_1[~idx]

    z1 = np.einsum('TN,T->TN', z1, np.conj(rot)/np.abs(rot))
    new_trajectory = np.empty_like(trajectory)
    new_trajectory[...,0] = z1.real
    new_trajectory[...,1] = z1.imag
    error = np.sqrt(error)

    return new_trajectory

def linear_stability_analysus(bd):
    """
    Perform a linear stability analysis for a given brownian dynamics simulation.
    Returns (eigenvalues, eigenvectors)
    """
    # bd.run(2000)

    F = bd._total_force(bd.time, bd.position, bd.orientation)
    T = np.cross(bd.position, F, axis=1)[:,2]

    dx = 1e-11
    force_matrix = np.zeros([bd.Nparticles, 2, bd.Nparticles, 2], dtype=float)
    for i in range(bd.Nparticles):
        for j in [0,1]:
            bd.position[i,j] += dx
            bd._update_interactions(bd.time, bd.position, bd.orientation)
            F = bd._total_force(bd.time, bd.position, bd.orientation)
            bd.position[i,j] -= dx

            for k in range(len(F)):
                r = np.linalg.norm(bd.position[k])
                tangent = np.array([-bd.position[k][1], bd.position[k][0], 0])
                if r > dx:
                    F[k] -= T[k]/r**2*tangent

            force_matrix[...,i,j] = F[:,:2]/dx

    w, v = np.linalg.eig(force_matrix.reshape([2*force_matrix.shape[0], -1]))
    v = v.reshape([bd.Nparticles, 2, 2*bd.Nparticles])
    v = np.moveaxis(v, -1, 0)

    return w, v
