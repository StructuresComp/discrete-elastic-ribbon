# dismech-homotopy-general

A discrete differential geometry deformable structure simulator with **homotopic geometry and material updates** during simulation. Based on [Dismech](https://github.com/StructuresComp/dismech-rods).

## Setup

To install this Python library within a new [virtual environment](https://docs.python.org/3/library/venv.html) execute the following bash commands. If you wish to use your own package manager (conda), only execute the commands after the comment.

```bash
python -m venv .venv
source .venv/bin/activate           # .venv/Script/activate for Windows

# after virtual environment setup
pip install -r requirements.txt
pip install -e .                    # Editable installation for development
```

---

## Homotopy: Generalized Geometry & Material Updates During Simulation

This fork extends Dismech with a **generalized homotopy API** that allows changing rod cross-section geometry (width, height) and material parameters *while a simulation is running*, without restarting or re-discretizing the structure.

### Motivation

In bifurcation and path-following studies, one often needs to:
- Continuously vary cross-section dimensions along a solution branch
- Switch material properties to explore different regimes
- Interpolate between configurations during quasi-static or dynamic simulations

Homotopy enables these by mutating the internal state (geometry, stiffness, mass) in place and propagating updates through springs and energy models.

### API Overview

| Component | Method | Purpose |
|-----------|--------|---------|
| **SoftRobot** | `update_rod_geometry(width=..., height=...)` | Update rod cross-section (w×h). Mutates `GeomParams`, stiffness, springs, mass matrix. |
| **SoftRobot** | `update_rod_material(density=..., youngs_rod=..., poisson_rod=...)` | Update material. Recomputes stiffness and mass, propagates to springs. |
| **SoftRobot** | `geom`, `material` | Properties to access current geometry and material. |
| **TimeStepper** | `refresh_rod_params(robot)` | Propagate rod changes into elastic energy models (e.g. Sano nn_model). Call after `update_rod_geometry` or `update_rod_material`. |

### Flow for Homotopy

1. **Before each time step** (e.g. in `before_step` callback): compute desired width/height or material from time or other logic.
2. Call `robot.update_rod_geometry(width=w, height=h)` and/or `robot.update_rod_material(...)`.
3. Call `stepper.refresh_rod_params(robot)` so the analytical energy model (e.g. Sano) gets updated parameters.
4. Run the step as usual.

Geometry and material changes are applied in place; DOF structure (nodes, edges) remains unchanged.

### Implementation Details

- **`SoftRobot`** stores `__material` and exposes `geom` and `material` properties. `update_rod_geometry` recomputes `rod_r0`, `axs`, `ixs1`, `ixs2`, `jxs`, then updates all rod springs (StretchBendTwist, BendTwist, Stretch, StretchNode) and the mass matrix. `update_rod_material` updates material, recomputes stiffness, and propagates to the same springs.
- **`AnalyticalSanosElasticEnergy`** gains `update_params(EA, EI1, EI2, GJ, delta_l, zeta, h)` to refresh its internal state for homotopy.
- **`GeneralElasticEnergySano`** gains `refresh_params_from_springs(springs, sano_zeta=...)` to pull current stiffness and geometry from springs and update the analytical nn_model.
- **`TimeStepper.refresh_rod_params(robot)`** iterates over elastic energies and calls `refresh_params_from_springs` where available, computing `sano_zeta` from geometry when using the Sano model.

---

## Shear-Induced Homotopy Demo

The `shear_induced_homotopy/` directory demonstrates homotopy in a shear-induced bifurcation setting:

1. **Phase 1–2**: Apply boundary shear (move clamped end nodes in x, then y).
2. **Phase 3 (homotopy)**: No boundary motion; *ramp rod width* from start to end using `update_rod_geometry(width=...)` and `refresh_rod_params()` at each step.
3. **Phase 4**: Reverse shear with the new width.

The homotopy phase smoothly interpolates cross-section width while the simulation runs, enabling path-following and bifurcation studies.

### Demo commands

```bash
# Run simulation (generates pkl in out/pkls, plots in out/plots)
cd shear_induced_homotopy
python simulate.py --config simulate_shear_homotopy_config.yaml --pkl-dir out/pkls --plot-dir out/plots

# Override discretization or energy model
python simulate.py --config simulate_shear_homotopy_config.yaml --pkl-dir out/pkls --plot-dir out/plots --nodes 45
python simulate.py --config simulate_shear_homotopy_config.yaml --pkl-dir out/pkls --plot-dir out/plots --energy-model sano

# Plot 2×2 subplots (Hmid vs ΔW, with FEA reference and shear force)
python plot_shear_homotopy.py --config config_shear_homotopy_plot.yaml --plot-dir out/plots --output shear_homotopy_sano.png
```

### Key config parameters (`simulate_shear_homotopy_config.yaml`)

| Section | Parameter | Description |
|---------|-----------|-------------|
| `geometry` | `L`, `w_by_l_ratio`, `h` | Rod length, initial width (as fraction of L), thickness |
| `homotopy` | `start_time`, `end_time` | Time window for width ramp (no boundary motion) |
| `homotopy` | `start_width`, `end_width` | Width at start/end of homotopy (`null` = use initial) |
| `boundary_condition` | `motion_phases` | List of `{start_time, end_time, direction, reverse}` for node motion |
| `simulation` | `dt`, `total_time`, `max_iter` | Time step, duration, Newton iteration limit |
| `adaptive_dt` | `enabled`, `max_dq_threshold` | Enable adaptive dt, displacement threshold |

The `before_step` callback applies gravity ramp, motion phases, and during the homotopy window calls `robot.update_rod_geometry(width=...)` and `stepper.refresh_rod_params(robot)` at each step.

---

## Features

- 3D discrete elastic rod stretching, bending, and twisting.
- 3D discrete elastic shell hinge and mid-edge bending.
- **Homotopic updates**: change rod width, height, and material during simulation.
- Implicit integration schemes (Euler, Newmark-beta).
- Dense and sparse (PyPardiso) solvers with robust regularization.
