StokeD
==============
StokeD solves the Stokesian dynamics (SD) equations for N interacting particles. SD is a generalization of Brownian dynamics (BD) that includes hydrodynamic coupling interactions.


<p align="center">
  <img src="./docs/estatics.png" width="500" alt="">
  <br>
  <em>Electrically charged particles repulsed from each other confined in a harmonic potential</em>
</p>

<p align="center">
  <img src="./docs/rt_coupling.png" width="500" alt="">
  <br>
  <em>Seven particles spinning due to an applied torque; the rotation of the fluid results in translational motion of the outer particles</em>
</p>


Features
--------------
+ Hydrodynamic coupling interactions can be turned on or off
+ Flexible interface for user-defined particle interactions
+ Several interactions already available: point electrostatics and electrodynamics, screened Coulomb, van der Walls, hard-sphere collisions, gravity
+ Animation module for visualizing trajectories

Installation
--------------
StokeD can be installed with pip
```shell
pip install stoked
```

Usage
--------------

See the [examples](src/stoked/examples) folder for how to use StokeD.

Run any of the available [examples](src/stoked/examples) without explicit installation using `uv`:

| Command | Description |
|---------|-------------|
| `uvx stoked free_particle` | Free Brownian diffusion in 3D |
| `uvx stoked estatics` | Electrostatic interactions with harmonic confinement |
| `uvx stoked ellipsoid` | Ellipsoidal particle with external force and torque |
| `uvx stoked tt_coupling` | Hydrodynamic translation-translation coupling |
| `uvx stoked rt_coupling` | Hydrodynamic rotation-translation coupling |
| `uvx stoked bigaussian_potential` | Particle in double-well potential |

For full documentation, see docs folder.

License
--------------
StokeD is licensed under the terms of the MIT license.
