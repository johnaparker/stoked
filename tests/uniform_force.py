from stoked import brownian_dynamics
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm

def uniform_force(t, rvec, Fx):
    F = np.zeros_like(rvec)
    F[:,0] = Fx
    return F

def test_uniform_motion():
    position = [[0,0,0]]
    damping = 1
    temperature = .1
    dt = 10
    Fx = 10e-13

    Nsteps = 50000

    history = np.zeros([Nsteps,3], dtype=float)
    velocity = np.zeros([Nsteps,3], dtype=float)
    sim = brownian_dynamics(position, damping, temperature, dt, force=partial(uniform_force, Fx=Fx))

    for i in tqdm(range(Nsteps)):
        history[i] = sim.position.squeeze()
        velocity[i] = sim.velocity.squeeze()
        sim.step()

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.plot(history[:,0], history[:,1], lw=.5)

    fig, ax = plt.subplots()
    ax.hist(velocity[:,0], bins=100)

    v_drift_theory = Fx/(damping)
    v_drift_sim = np.average(velocity[:,0])
    print(v_drift_theory, v_drift_sim)

    plt.show()


