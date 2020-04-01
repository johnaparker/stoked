from stoked import brownian_dynamics, drag_sphere
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def uniform_force(t, rvec, orient, Fx):
    F = np.zeros_like(rvec)
    F[:,0] = Fx
    return F

def test_uniform_motion():
    position = [[0,0,0]]
    drag = drag_sphere(100e-9, 1)
    temperature = 300
    dt = 1e-6
    Fx = 10e-12

    Nsteps = 50000

    history = np.zeros([Nsteps,3], dtype=float)
    velocity = np.zeros([Nsteps,3], dtype=float)
    sim = brownian_dynamics(position=position, 
                            drag=drag,
                            temperature=temperature,
                            dt=dt,
                            force=partial(uniform_force, Fx=Fx))

    for i in tqdm(range(Nsteps)):
        history[i] = sim.position.squeeze()
        velocity[i] = sim.velocity.squeeze()
        sim.step()

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.plot(history[:,0], history[:,1], lw=.5)

    fig, ax = plt.subplots()
    ax.hist(velocity[:,0], bins=100)

    v_drift_theory = Fx/(drag.drag_T)
    v_drift_sim = np.average(velocity[:,0])
    print(v_drift_theory, v_drift_sim)

    plt.show()

test_uniform_motion()
