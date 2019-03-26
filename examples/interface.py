import numpy as np
import matplotlib.pyplot as plt
import pedesis
import miepy
from scipy.integrate import cumtrapz
from scipy import constants
from miepy.constants import Z0

nm = 1e-9
radius = 75*nm

A1 = pedesis.van_der_waals_sphere_interface(radius, .5e-19)
A2 = pedesis.double_layer_sphere_interface(radius, -16e-3, -77e-3, debye=22.7*nm)
A3 = pedesis.gravity_sphere(radius, 10490, direction=[0,0,1])

z = np.linspace(radius+5*nm, 400*nm, 100)
F1 = np.zeros_like(z)
F2 = np.zeros_like(z)
F3 = np.zeros_like(z)
F4 = np.zeros_like(z)

cluster = miepy.sphere_cluster(position=[0,0,0],
                               radius=radius,
                               material=miepy.materials.Ag(),
                               source=miepy.sources.gaussian_beam(width=1500*nm, power=.006, polarization=[1,0], theta=np.pi),
                               wavelength=800*nm,
                               lmax=2,
                               medium=miepy.materials.water())

E = cluster.E_source(0, 0, 0)
H = cluster.H_source(0, 0, 0)
S = 0.5/Z0*np.cross(E, np.conjugate(H), axis=0)[2]*1.33
print(S*1000/1e18)

for i,zval in enumerate(z):
    pos = np.array([[0,0,zval]])
    A1._update(None, pos, None, 300)
    A2._update(None, pos, None, 300)
    F1[i] = A1.force()[0,2]
    F2[i] = A2.force()[0,2]
    F3[i] = A3(None, pos, None)[0,2]

    cluster.update_position(pos)
    F4[i] = cluster.force_on_particle(0)[2]

fig, ax = plt.subplots()
ax.plot(z/nm, F1, label='waals')
ax.plot(z/nm, F2, label='estatic')
ax.plot(z/nm, F3, label='gravity')
ax.plot(z/nm, F4, label='light')

ax.plot(z/nm, F1 + F2 + F3 + F4, label='sum', color='k')
ax.legend()

fig, ax = plt.subplots()

kT = 300*constants.k
W1 = cumtrapz(-F1, z, initial=0)/kT
W2 = cumtrapz(-F2, z, initial=0)/kT
W3 = cumtrapz(-F3, z, initial=0)/kT
W4 = cumtrapz(-F4, z, initial=0)/kT

W1 -= W1[-1]
W2 -= W2[-1]
W3 -= W3[-1]
# W4 -= W4[-1]

ax.plot(z/nm - radius/nm, W1, label='waals')
ax.plot(z/nm - radius/nm, W2, label='estatic')
ax.plot(z/nm - radius/nm, W3, label='gravity')
ax.plot(z/nm - radius/nm, W4, label='light')

ax.plot(z/nm - radius/nm, W1 + W2 + W3 + W4, label='sum', color='k')

ax.set(xlabel='surface separation (nm)', ylabel='potential energy (kT)')
ax.legend()

plt.show()
