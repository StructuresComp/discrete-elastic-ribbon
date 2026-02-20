# Discrete Elastic Ribbon

A discrete differential geometry deformable structure simulator for elastic ribbons with **adaptive time-stepping**, **generalized energy formulations**, and **homotopic geometry/material updates** during simulation.

This codebase was forked from [PyDismech](https://github.com/StructuresComp/dismech-python) at commit [4f0eb69](https://github.com/StructuresComp/dismech-python/commit/4f0eb69d9dcf21bbbc634d3eaa9b3b36235a2b2a) and refactored for Discrete Elastic Ribbon.

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

## Additional Features in Discrete Elastic Ribbon

### 1. Adaptive Time-Stepping and Robust Solver

The time stepper in `src/dismech/time_steppers/time_stepper.py` implements **adaptive implicit Euler integration** and a **robust linear solver** to handle stiff systems and near-singular Jacobians during bifurcation and large deformations.

#### Adaptive Implicit Euler Integration

```
REQUIRE: Initial state q₀, q̇₀; time step bounds h_min, h_max; tolerances ε_force, ε_disp
ENSURE: Equilibrium configuration q*

WHILE t < T:
    Newton-Raphson iteration:
    REPEAT
        Compute F_int = -∇_q E, H_E = ∇²_q E from generalized energy
        Assemble residual r = M Δq - h M q̇ - h² (F_int + F_ext)
        Assemble Jacobian J = M - h² (H_E + ∂F_ext/∂q)
        Δq_free ← RobustSolve(J_free, r_free)   [Algorithm 2]
        q ← q - Δq_free | free DOFs
    UNTIL ‖r_free‖ < ε_force OR ‖Δq‖_∞/h < ε_disp

    Adaptive time-stepping:
    IF not converged:
        h ← max(h/2, h_min); retry step
    ELSIF ‖Δq⁽⁰⁾‖ small and few iterations:
        h ← min(1.5h, h_max)   [Increase step]
    ENDIF
    Update q̇, m₁ⁱ, m₂ⁱ, m_refⁱ; t ← t + h
RETURN q
```

#### Robust Linear Solve with Regularization

```
REQUIRE: Jacobian J, residual r; condition threshold κ_max; regularization λ₀
ENSURE: Newton step Δq

IF cond(J) < κ_max:
    RETURN J⁻¹ r   [Direct solve]
λ ← λ₀
WHILE cond(J + λI) ≥ κ_max:
    λ ← 10λ   [Tikhonov-Miller regularization]
ENDWHILE
IF regularization sufficient:
    RETURN (J + λI)⁻¹ r
ELSE:
    RETURN J⁺ r   [Truncated SVD pseudo-inverse fallback]
ENDIF
```

The **RobustSolver** (`src/dismech/solvers/solver.py`) applies Tikhonov regularization when the Jacobian is ill-conditioned (default threshold `cond > 1e12`) and falls back to an SVD-based pseudo-inverse if the regularized solve fails. Adaptive time-stepping reduces `dt` when Newton fails or when displacement per step exceeds `max_dq_threshold`, and increases `dt` when the system is stable.

---

### 2. Generalized Energy Formulation

Discrete Elastic Ribbon uses a **generalized elastic energy** framework that separates geometry-dependent strain computation from analytical strain-energy models. This allows plugging in different rod energy models (Kirchhoff, Sadowsky, Wunderlich, Sano, Audoly) by implementing only the analytical energy as a function of strains.

#### Algorithm: Generalized Elastic Energy Gradient and Hessian Assembly

```
REQUIRE: State q, material frame (m₁, m₂), natural strains ε_nat
REQUIRE: Analytical model E(ε) with ∇_ε E and ∇²_ε E
ENSURE: Global force F, stiffness K

Initialize F ← 0, K ← 0
FOR each element k = 1, …, N:
    Step 1: Compute local strains from geometry
        Extract node positions (x₀, x₁, x₂)_k and twist angles (θ_e, θ_f)_k
        Compute strain vector ε_k = [ε, κ⁽¹⁾, κ⁽²⁾, τ]_kᵀ
        Compute strain gradients ∂ε_k/∂q_loc and Hessians ∂²ε_k/∂q_loc²

    Step 2: Query analytical model in strain space
        Δε_k ← ε_k - ε_nat,k   [Delta strain]
        Normalize: ε̃_k ← [Δε, Δκ⁽¹⁾(h/ℓ), Δκ⁽²⁾(h/ℓ), Δτ(h/ℓ)]_kᵀ
        Evaluate E_k, ∇_ε̃ E_k, ∇²_ε̃ E_k from analytical model

    Step 3: Chain rule to local DOF stencil
        ∇_q_loc E_k ← (∂ε_k/∂q_loc)ᵀ ∇_ε̃ E_k
        ∇²_q_loc E_k ← (∂ε_k/∂q_loc)ᵀ ∇²_ε̃ E_k (∂ε_k/∂q_loc) + Σ_i (∂E_k/∂ε_i)(∂²ε_i/∂q_loc²)

    Step 4: Assemble to global DOFs
        F[I_k] ← F[I_k] - ∇_q_loc E_k
        K[I_k, I_k] ← K[I_k, I_k] - ∇²_q_loc E_k
ENDFOR
```

Implemented in `src/dismech/elastics/general_elastic_energy.py` and the model-specific `GeneralElasticEnergy*` classes.

---

### 3. Supported Energy Models

| Model | Analytical Energy | General Elastic Energy |
|-------|-------------------|------------------------|
| **Kirchhoff** | `analytical_kirchhoff_elastic_energy.py` | `general_elastic_energy_kirchhoff.py` |
| **Sadowsky** | `analytical_sadowsky_elastic_energy.py` | `general_elastic_energy_sadowsky.py` |
| **Wunderlich** | `analytical_wunderlichs_elastic_energy.py` | `general_elastic_energy_wunderlich.py` |
| **Sano** | `analytical_sanos_elastic_energy.py` | `general_elastic_energy_sano.py` |
| **Audoly** | `analytical_audoly_elastic_energy.py` | `general_elastic_energy_audoly.py` |

Each analytical module provides the energy density E as a function of normalized strains **ε̃ = [ϵ, κ₁, κ₂, τ]**, plus gradient and Hessian.

---

### 4. Adding a New Energy Model

To add a new rod energy model:

1. **Implement the analytical energy** in `src/dismech/elastics/`:
   - Create `analytical_<name>_elastic_energy.py` with a class that provides:
     - `forward(x)` or equivalent: energy density E(ε̃) for input `x` of shape `(batch, 4)`.
     - `compute_energy_grad_hess(x)`: returns (E, gradient, Hessian) w.r.t. normalized strains.

2. **Implement the general elastic energy wrapper**:
   - Create `general_elastic_energy_<name>.py` that:
     - Extends the base generalized energy framework.
     - Calls your analytical model with the normalized strain vector ε̃ = [ϵ, κ₁, κ₂, τ].
     - Assembles forces and stiffness via the chain rule as in Algorithm 3.

3. **Register in the time stepper** (`time_stepper.py`):
   - Add the new model to the `energy_model_type` dispatch and construct the analytical model with geometry-derived parameters (EA, EI1, EI2, GJ, delta_l, h, and any model-specific parameters like Sano's zeta or Wunderlich's W).

Only the analytical energy expression E(ε̃) needs to be implemented; the rest is handled by the generalized framework.

---

## Shear-Induced Bifurcation Simulation

The `shear_induced_homotopy/simulate.py` script implements the boundary value problem that isolates the **shear-induced supercritical pitchfork bifurcation** of a pre-buckled elastic ribbon.

**Boundary condition:** A ribbon (length L = 0.1 m, width W, thickness b) is clamped at both ends. First, one clamp is moved toward the other along the ribbon axis to apply longitudinal compression (about 25% of L), which buckles the ribbon. Then, with that pre-buckled state fixed, a transverse displacement (delta-W) is applied at one clamp to impose shear. The simulation drives these two loading phases via prescribed motion of the boundary nodes.

The same BVP is implemented in both `shear_induced_bifurcation/simulate.py` (compression + shear only, with optional tracking) and `shear_induced_homotopy/simulate.py` (adds a homotopy width-ramp phase; see [Shear-Induced Homotopy Demo](#shear-induced-homotopy-demo)).

### Usage (`shear_induced_bifurcation/simulate.py`)

```bash
cd shear_induced_bifurcation
python simulate.py --config simulate_shear_induced_bifurcation_config.yaml --pkl-dir out/pkls --plot-dir out/plots

# Override W/L ratio(s) or node count(s) from config
python simulate.py --config simulate_shear_induced_bifurcation_config.yaml --pkl-dir out/pkls --plot-dir out/plots --wbyl 1/14 1/18 --nodes 21 63

# Override energy model (sano, wunderlich, kirchhoff, sadowsky, audoly)
python simulate.py --config simulate_shear_induced_bifurcation_config.yaml --pkl-dir out/pkls --plot-dir out/plots --energy-model sano
```

Outputs are written to `--pkl-dir`; plots (if generated) go to `--plot-dir`. Use `--plot` to run only missing cases then plot, instead of rerunning all.

### In the Code

- **Phase 1 (move in x)**: The “start” nodes (e.g. x ≤ 0.01) are displaced along the longitudinal direction, inducing compression and pre-buckling.
- **Phase 2 (move in y)**: The same boundary nodes are displaced transversely, imposing delta-W incrementally and inducing shear.
- **Boundary conditions**: `boundary_condition.start_x_threshold` and `end_x_threshold` define which nodes are clamped/ driven. `u0` is the displacement per step; `motion_phases` define the time windows and directions (0=x, 1=y, 2=z) for each phase.
- **Gravity ramp**: Strong gravity during `gravity_ramp_end_time` settles the rod; then `gravity_after_ramp` (often zero) isolates the elastic response.

### Key Config Parameters (`simulate_shear_induced_bifurcation_config.yaml`)

| Section | Parameter | Description |
|---------|-----------|-------------|
| `geometry` | `L`, `w_by_l_ratio`, `h` | Ribbon length, width as fraction of L, thickness |
| `boundary_condition` | `start_x_threshold`, `end_x_threshold` | Distance thresholds for start/end nodes |
| `boundary_condition` | `u0` | Prescribed displacement per step (m) |
| `boundary_condition` | `motion_phases` | List of `{start_time, end_time, direction, reverse}` |
| `boundary_condition` | `gravity_ramp_end_time`, `gravity_during_ramp`, `gravity_after_ramp` | Gravity for settling vs. elastic phase |
| `simulation` | `dt`, `total_time`, `max_iter` | Time step, duration, Newton iteration limit |
| `adaptive_dt` | `enabled`, `max_dq_threshold`, `dt_reduction_factor` | Adaptive time-stepping |
| `energy_model` | `name` | One of: sano, wunderlich, kirchhoff, sadowsky, audoly |

---

## Homotopy: Generalized Geometry & Material Updates During Simulation

Discrete Elastic Ribbon extends PyDismech with a **homotopy API** that allows changing rod cross-section geometry (width, height) and material parameters *during* a simulation, without restarting or re-discretizing.

### Motivation

In bifurcation and path-following studies, one often needs to:
- Continuously vary cross-section dimensions along a solution branch
- Switch material properties to explore different regimes
- Interpolate between configurations during quasi-static or dynamic simulations

Homotopy enables this by mutating geometry, stiffness, and mass in place and propagating updates through springs and energy models.

### API Overview

| Component | Method | Purpose |
|-----------|--------|---------|
| **SoftRobot** | `update_rod_geometry(width=..., height=...)` | Update rod cross-section (w×h). Mutates GeomParams, stiffness, springs, mass matrix. |
| **SoftRobot** | `update_rod_material(density=..., youngs_rod=..., poisson_rod=...)` | Update material. Recomputes stiffness and mass, propagates to springs. |
| **TimeStepper** | `refresh_rod_params(robot)` | Propagate rod changes into elastic energy models. Call after `update_rod_geometry` or `update_rod_material`. |

### Flow for Homotopy

1. In a `before_step` callback: compute desired width/height or material from time or other logic.
2. Call `robot.update_rod_geometry(width=w, height=h)` and/or `robot.update_rod_material(...)`.
3. Call `stepper.refresh_rod_params(robot)` so the analytical energy model (e.g. Sano) receives updated parameters.
4. Run the step as usual.

---

## Shear-Induced Homotopy Demo

The `shear_induced_homotopy/` directory demonstrates homotopy on top of the shear-induced bifurcation setup:

1. **Phases 1–2**: Apply compression and shear as above.
2. **Phase 3 (homotopy)**: No boundary motion; *ramp rod width* from `start_width` to `end_width` using `update_rod_geometry(width=...)` and `refresh_rod_params()` at each step.
3. **Phase 4**: Reverse shear with the new width.

The homotopy phase smoothly interpolates cross-section width during the simulation, enabling path-following and bifurcation studies.

### Demo commands

```bash
cd shear_induced_homotopy
python simulate.py --config simulate_shear_homotopy_config.yaml --pkl-dir out/pkls --plot-dir out/plots

# Override discretization or energy model
python simulate.py --config simulate_shear_homotopy_config.yaml --pkl-dir out/pkls --plot-dir out/plots --nodes 45
python simulate.py --config simulate_shear_homotopy_config.yaml --pkl-dir out/pkls --plot-dir out/plots --energy-model sano

# Plot Hmid vs delta-W (with FEA reference and shear force)
python plot_shear_homotopy.py --config config_shear_homotopy_plot.yaml --plot-dir out/plots --output shear_homotopy_sano.png
```

### Homotopy config parameters (`simulate_shear_homotopy_config.yaml`)

| Section | Parameter | Description |
|---------|-----------|-------------|
| `homotopy` | `start_time`, `end_time` | Time window for width ramp (no boundary motion) |
| `homotopy` | `start_width`, `end_width` | Width at start/end of homotopy (`null` = use initial) |

The `before_step` callback applies gravity ramp, motion phases, and during the homotopy window calls `robot.update_rod_geometry(width=...)` and `stepper.refresh_rod_params(robot)`.

---

## Features

- 3D discrete elastic rod stretching, bending, and twisting.
- 3D discrete elastic shell hinge and mid-edge bending.
- **Adaptive implicit Euler** with robust linear solve (Tikhonov regularization, SVD fallback).
- **Generalized elastic energy** framework with pluggable analytical models (Kirchhoff, Sadowsky, Wunderlich, Sano, Audoly).
- **Homotopic updates**: change rod width, height, and material during simulation.
- Implicit integration schemes (Euler, Newmark-beta).
- Dense and sparse (PyPardiso) solvers.
