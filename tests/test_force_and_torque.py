import numpy as np
from scipy.integrate import cumulative_trapezoid

from stoked import brownian_dynamics, drag_ellipsoid


def force(t, pos, orient):
    """Constant force in x-direction."""
    return np.array([[2e-12, 0, 0]])


def torque(t, pos, orient):
    """Constant torque about z-axis."""
    return np.array([[0, 0, 5e-3]])


def test_force_and_torque_deterministic():
    """Test that force and torque produce expected motion at near-zero temperature."""
    np.random.seed(42)

    position = [0, 0, 0]
    drag = drag_ellipsoid([1, 1.5, 1], 1)
    temperature = 0.00001
    dt = 10
    Nsteps = 1000

    history = np.zeros([Nsteps, 3], dtype=float)
    omega_z = np.zeros([Nsteps], dtype=float)

    sim = brownian_dynamics(position=position, drag=drag, temperature=temperature, dt=dt, force=force, torque=torque)

    for i in range(Nsteps):
        history[i] = sim.position.squeeze()
        omega_z[i] = sim.angular_velocity[0, 2]
        sim.step()

    # Calculate expected displacement in x-direction
    Fx = 2e-12
    expected_displacement_x = (Fx / drag.drag_T[0, 0]) * dt * Nsteps
    actual_displacement_x = sim.position[0, 0]

    # At very low temperature, displacement should match theory closely
    assert np.isclose(expected_displacement_x, actual_displacement_x, rtol=0.01), (
        f"X displacement mismatch: expected={expected_displacement_x:.6e}, actual={actual_displacement_x:.6e}"
    )

    # Y and Z should remain near zero (small thermal noise)
    assert np.abs(sim.position[0, 1]) < 0.1 * np.abs(actual_displacement_x)
    assert np.abs(sim.position[0, 2]) < 0.1 * np.abs(actual_displacement_x)

    # Calculate expected cumulative rotation about z-axis
    tau_z = 5e-3
    time = np.arange(Nsteps) * dt
    cumulative_rotation = cumulative_trapezoid(omega_z, time, initial=0)
    expected_rotation = (tau_z / drag.drag_R[0, 2]) * dt * Nsteps
    actual_rotation = cumulative_rotation[-1]

    # Rotation should also match theory at low temperature
    assert np.isclose(expected_rotation, actual_rotation, rtol=0.01), (
        f"Rotation mismatch: expected={expected_rotation:.6e}, actual={actual_rotation:.6e}"
    )
