from functools import partial

import numpy as np

import stoked
from stoked import brownian_dynamics, drag_sphere


def uniform_force(t, rvec, orient, Fx):
    """Apply a uniform force in the x-direction."""
    F = np.zeros_like(rvec)
    F[:, 0] = Fx
    return F


def test_uniform_force_drift_velocity():
    """Test that drift velocity under uniform force matches theoretical prediction at finite temperature."""
    np.random.seed(42)

    position = [[0, 0, 0]]
    drag = drag_sphere(100e-9, 1)
    temperature = 300
    dt = 1e-6
    Fx = 10e-12

    Nsteps = 50000

    velocity = np.zeros([Nsteps, 3], dtype=float)
    sim = brownian_dynamics(
        position=position,
        drag=drag,
        temperature=temperature,
        dt=dt,
        force=partial(uniform_force, Fx=Fx),
        inertia=stoked.inertia_sphere(100e-9, 10490),
    )

    for i in range(Nsteps):
        velocity[i] = sim.velocity.squeeze()
        sim.step()

    v_drift_theory = Fx / drag.drag_T
    v_drift_sim = np.average(velocity[:, 0])

    # At finite temperature with sufficient sampling, drift velocity should match theory within 5%
    assert np.isclose(v_drift_theory, v_drift_sim, rtol=0.05), (
        f"Drift velocity mismatch: theory={v_drift_theory:.6e}, sim={v_drift_sim:.6e}"
    )

    # Verify other directions have zero drift (within noise)
    assert np.abs(np.average(velocity[:, 1])) < 0.1 * v_drift_theory
    assert np.abs(np.average(velocity[:, 2])) < 0.1 * v_drift_theory


def test_uniform_force_zero_temperature():
    """Test that motion under uniform force is deterministic at zero temperature."""
    np.random.seed(42)

    position = [[0, 0, 0]]
    drag = drag_sphere(100e-9, 1)
    temperature = 0.0
    dt = 1e-6
    Fx = 10e-12

    Nsteps = 1000

    sim = brownian_dynamics(
        position=position,
        drag=drag,
        temperature=temperature,
        dt=dt,
        force=partial(uniform_force, Fx=Fx),
        inertia=stoked.inertia_sphere(100e-9, 10490),
    )

    for _ in range(Nsteps):
        sim.step()

    # At T=0, motion should be purely deterministic
    # Final position should match: x = v * t = (F/drag) * dt * Nsteps
    expected_displacement = (Fx / drag.drag_T) * dt * Nsteps
    actual_displacement = sim.position[0, 0]

    assert np.isclose(expected_displacement, actual_displacement, rtol=1e-6), (
        f"Displacement mismatch: expected={expected_displacement:.6e}, actual={actual_displacement:.6e}"
    )

    # Y and Z should remain zero
    assert np.abs(sim.position[0, 1]) < 1e-15
    assert np.abs(sim.position[0, 2]) < 1e-15
