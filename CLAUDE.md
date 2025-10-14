# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StokeD is a Python library for simulating Stokesian dynamics (SD) for N interacting particles. SD is a generalization of Brownian dynamics (BD) that includes hydrodynamic coupling interactions. The library is designed for computational physics research, particularly for simulating particle motion in viscous fluids.

## Development Commands

### Linting and Formatting
```bash
# Check code with ruff (lint)
ruff check

# Auto-fix linting issues
ruff check --fix

# Format code with ruff
ruff format

# Pre-commit hooks (runs both lint and format)
pre-commit run --all-files
```

### Installation
```bash
# Install with dev dependencies (using uv)
uv sync 
```

### Running Examples
```bash
# Examples are in the examples/ directory
python examples/estatics.py
python examples/hydrodynamic_tt_coupling.py
```

## Code Architecture

### Core Components

**solver.py** - Contains the main simulation engines:
- `stokesian_dynamics`: Main class for SD simulations with hydrodynamic coupling
- `brownian_dynamics`: Wrapper function that creates a `stokesian_dynamics` instance with `hydrodynamic_coupling=False`
- `trajectory`: Data structure holding position and orientation arrays over time
- `interactions`: Abstract base class for particle interaction implementations

**integrators/** - Time-stepping algorithms:
- `integrator.py`: Base abstract class defining the integration interface
- `basic_integrator.py`: Standard Euler-Maruyama integration
- `predictor_corrector_integrator.py`: Higher-order integration scheme
- Four step methods per integrator: `bd_step`, `ld_step`, `hbd_step`, `hld_step` (Brownian dynamics, Langevin dynamics, hydrodynamic-Brownian, hydrodynamic-Langevin)

**hydrodynamics.py** - Hydrodynamic coupling calculations:
- `grand_mobility_matrix()`: Constructs the 6N×6N grand mobility matrix including TT (translation-translation), RR (rotation-rotation), RT, and TR coupling terms
- `particle_wall_self_mobility()`: Computes particle-wall interactions for interface simulations
- `interface`: Class representing no-slip boundary interfaces

**forces.py** - Interaction force framework:
- `pairwise_force`: Base class for arbitrary pairwise forces F(r_ij)
- `pairwise_central_force`: Optimized class for central forces F(r)
- Supports addition (`+`) to combine multiple force objects

### Physics Modules

Each module defines specific physical interactions:

- **drag.py**: Drag coefficient classes (`drag_sphere`, `drag_ellipsoid`) - required for all simulations
- **inertia.py**: Inertia coefficients (`inertia_sphere`, `inertia_ellipsoid`) - optional for Langevin dynamics
- **electrostatics.py**: Coulomb and screened electrostatic interactions
- **electrodynamics.py**: Point dipole electrodynamics
- **gravity.py**: Gravitational forces
- **van_der_waals.py**: van der Waals interactions
- **collisions.py**: Hard-sphere collision handling
- **constraints.py**: Position and rotation constraints

### Visualization

**vis/** directory:
- `animate_2d.py`: 2D trajectory animations
- `animate_3d.py`: 3D trajectory animations
- `snapshots.py`: Static trajectory snapshots
- Provides helper functions for creating particle patches (circles, ellipses, spheres, ellipsoids)

### Typical Simulation Setup Pattern

1. Define initial positions and orientations (quaternions)
2. Create drag coefficients object
3. Define external forces/torques as functions: `force(time, position, orientation) -> force_array`
4. Define pairwise interactions using provided physics modules or custom `interactions` subclasses
5. Create solver instance (`stokesian_dynamics` or `brownian_dynamics`)
6. Run simulation with `.run(Nsteps)` or `.run_until(condition)`
7. Analyze or visualize the returned `trajectory` object

## Important Implementation Details

### Quaternion Orientations
- Particle orientations are represented using the `numpy-quaternion` package
- Import is suppressed with warnings to avoid numpy compatibility warnings (see `__init__.py:8-10`)
- Quaternions are normalized after each time step (see `integrator.py:post_step`)

### Hydrodynamic Coupling
- When `hydrodynamic_coupling=True`, the grand mobility matrix is computed periodically (default every 10 steps via `grand_mobility_interval`)
- Interface simulations only work in 3D and are incompatible with anisotropic particles
- Isotropic particles without hydrodynamic coupling take an optimized code path

### Interaction Updates
- All interactions are updated at the start of each step via `_update_interactions()`
- The `interactions` base class requires implementing three methods: `force()`, `torque()`, `update()`
- Multiple `pairwise_central_force` objects are automatically combined into a single object for efficiency (see `solver.py:263-275`)

### Force and Torque Functions
- External forces/torques can be single functions or lists of functions
- Functions receive `(time, position, orientation)` and return arrays of shape `[N, D]` where N is number of particles and D is dimensionality
- Zero-force/torque defaults are applied if not specified

## Code Style Notes

- Uses Ruff with Google-style docstrings
- Line length: 120 characters
- Lowercase variable naming allowed for physics quantities (N806, N802, N816, N803 ignored)
- Tests and examples have relaxed docstring requirements
- Type hints are not extensively used (legacy codebase)
