from functools import partial

import numpy as np
from scipy.constants import k as kb

from stoked import brownian_dynamics, drag_sphere


def harmonic_force(t, rvec, orientation, k=1):
    """Harmonic restoring force."""
    return -k * rvec


def test_free_diffusion():
    """Test that mean squared displacement grows linearly with time for free diffusion."""
    np.random.seed(42)

    position = [0, 0, 0]
    drag = drag_sphere(1 / (6 * np.pi), 1)
    temperature = 0.05
    dt = 10
    Nsteps = 10000

    history = np.zeros([Nsteps, 3], dtype=float)

    sim = brownian_dynamics(position=position, drag=drag, temperature=temperature, dt=dt, force=None)

    for i in range(Nsteps):
        history[i] = sim.position.squeeze()
        sim.step()

    # Calculate mean squared displacement
    msd = np.sum(history**2, axis=1)

    # Theoretical MSD: <r²> = 6*D*t where D = kB*T / (6*pi*eta*a) = kB*T / drag
    # For drag_sphere(1/(6*pi), 1): drag_T = 6*pi*eta*a = 6*pi*1*(1/(6*pi)) = 1
    D = kb * temperature / drag.drag_T
    time = np.arange(Nsteps) * dt
    expected_msd = 6 * D * time

    # Check linear growth by comparing slope in later half (equilibrated)
    # Use last 50% of data for linear fit
    start_idx = Nsteps // 2
    msd_fit = msd[start_idx:]
    time_fit = time[start_idx:]

    # Fit slope
    slope = np.polyfit(time_fit, msd_fit, 1)[0]
    expected_slope = 6 * D

    # Slope should match within 10% (statistical noise)
    assert np.isclose(slope, expected_slope, rtol=0.1), (
        f"MSD slope mismatch: expected={expected_slope:.6e}, actual={slope:.6e}"
    )


def test_harmonic_potential_energy_distribution():
    """Test that equilibrium energy distribution in harmonic potential matches theory."""
    np.random.seed(42)

    position = [0, 0, 0]
    drag = drag_sphere(1 / (6 * np.pi), 1)
    temperature = 0.05
    dt = 10
    k = 0.01  # Spring constant
    Nsteps = 10000

    history = np.zeros([Nsteps, 3], dtype=float)

    sim = brownian_dynamics(
        position=position, drag=drag, temperature=temperature, dt=dt, force=partial(harmonic_force, k=k)
    )

    for i in range(Nsteps):
        history[i] = sim.position.squeeze()
        sim.step()

    # Calculate radial distribution and infer energy
    rad = np.linalg.norm(history, axis=1)
    hist, edges = np.histogram(rad, bins=100)

    rad = edges[1:]
    counts = hist / (4 * np.pi * rad**2)

    # Filter out bins with zero counts to avoid log(0)
    valid = counts > 0
    rad_valid = rad[valid]
    counts_valid = counts[valid]

    # Calculate energy from Boltzmann distribution
    E = -kb * temperature * np.log(counts_valid)
    E -= E[6]  # Shift to match reference point as in original test

    # Theoretical harmonic potential energy
    E_theory = 0.5 * k * rad_valid**2
    E_theory -= E_theory[6]  # Apply same shift

    # Use valid range where statistics are good (exclude tails with poor statistics)
    # Use middle 60% of valid points
    start = len(E) // 5
    end = 4 * len(E) // 5

    # Calculate relative error in the well-sampled region
    relative_error = np.abs((E[start:end] - E_theory[start:end]) / (E_theory[start:end] + 1e-10))
    mean_relative_error = np.mean(relative_error)

    # Mean relative error should be less than 20% in well-sampled region
    assert mean_relative_error < 0.2, f"Energy distribution error too large: {mean_relative_error:.3f}"

    # Also check correlation coefficient
    correlation = np.corrcoef(E[start:end], E_theory[start:end])[0, 1]
    assert correlation > 0.95, f"Energy correlation too low: {correlation:.3f}"
