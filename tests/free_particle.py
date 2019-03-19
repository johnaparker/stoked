from pedesis import brownian_dynamics, sphere_drag
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

position = [0,0,0]
drag = sphere_drag(.2, 1)
temperature = .5
dt = 10

Nsteps = 10000
fig, ax = plt.subplots()
ax.set_aspect('equal')

history = np.zeros([Nsteps,3], dtype=float)

sim = brownian_dynamics(position=position, drag=drag, temperature=temperature, dt=dt)

for i in tqdm(range(Nsteps)):
    history[i] = sim.position.squeeze()
    sim.step()

ax.plot(history[:,0], history[:,1], lw=.5)
ax.legend()

plt.show()
