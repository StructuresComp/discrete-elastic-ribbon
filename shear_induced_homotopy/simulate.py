#!/usr/bin/env python3
"""
Shear-induced bifurcation with homotopy: width ramp between phase 3 and phase 4.

Phases:
  1) Move boundary nodes in x
  2) Move boundary nodes in y (shear)
  3) Homotopy: no boundary motion, interpolate rod width from start_width to end_width
  4) Reverse shear in y (with end width)

Uses update_rod_geometry() and refresh_rod_params() for homotopic width change.

Usage:
  python simulate.py --config simulate_shear_homotopy_config.yaml --pkl-dir out/pkls --plot-dir out/plots
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import pickle
import dismech
import yaml


def load_config(config_path):
    """Load YAML config."""
    path = Path(config_path)
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def ensure_initial_geometry(geometry_dir, n_nodes, L):
    """Create initial geometry file for rod along x-axis if not present."""
    geometry_dir = Path(geometry_dir)
    geometry_dir.mkdir(parents=True, exist_ok=True)
    path = geometry_dir / f"horizontal_rod_n{n_nodes}.txt"
    if path.exists():
        return path
    dl = L / (n_nodes - 1)
    lines = ["*Nodes"]
    for i in range(n_nodes):
        x = i * dl
        lines.append(f"{x},0,0")
    lines.append("*Edges")
    for i in range(1, n_nodes):
        lines.append(f"{i},{i+1}")
    path.write_text("\n".join(lines) + "\n")
    return path


def parse_w_by_l_ratio(ratio_str):
    """Parse W/L ratio string like '1/14' into float."""
    if isinstance(ratio_str, (int, float)):
        return float(ratio_str)
    s = str(ratio_str)
    if '/' in s:
        parts = s.split('/')
        return float(parts[0]) / float(parts[1])
    return float(s)


def infer_n_nodes(q0, geo):
    """Infer number of nodes from state vector or geometry."""
    try:
        if hasattr(geo, 'nodes') and geo.nodes is not None:
            return len(geo.nodes)
    except Exception:
        pass
    n_dof = len(q0)
    max_n_nodes = n_dof // 3
    for n_nodes in range(max_n_nodes, 0, -1):
        node_dofs = 3 * n_nodes
        if node_dofs > n_dof:
            continue
        node_positions = q0[:node_dofs].reshape(-1, 3)
        if np.any(np.abs(node_positions) > 1e-10) and np.all(np.abs(node_positions) < 1e6):
            return n_nodes
    return max_n_nodes


def get_node_positions(q, n_nodes):
    """Extract node positions from state vector."""
    return q[: 3 * n_nodes].reshape(-1, 3)


def find_middle_node(q0, n_nodes, start_thresh=0.01, end_thresh=0.09):
    """Find the middle node (closest to L/2 along x-axis)."""
    node_positions = get_node_positions(q0, n_nodes)
    x_coords = node_positions[:, 0]
    L_val = x_coords.max() - x_coords.min()
    L_half = x_coords.min() + L_val / 2.0
    return np.argmin(np.abs(x_coords - L_half))


def find_start_nodes(q0, n_nodes, start_thresh=0.01):
    """Find start nodes (nodes at x <= start_thresh)."""
    node_positions = get_node_positions(q0, n_nodes)
    return np.where(node_positions[:, 0] <= start_thresh)[0]


def extract_delta_h_w(qs, n_nodes, middle_node_idx, start_node_indices):
    """Extract DeltaH and DeltaW from state trajectory. Returns delta_h, delta_w, L."""
    q0 = qs[0]
    node_positions_0 = get_node_positions(q0, n_nodes)
    z_middle_initial = node_positions_0[middle_node_idx, 2]
    y_start_avg_initial = np.mean(node_positions_0[start_node_indices, 1])
    L_val = node_positions_0[:, 0].max() - node_positions_0[:, 0].min()
    delta_h, delta_w = [], []
    for q in qs:
        node_positions = get_node_positions(q, n_nodes)
        delta_h.append(node_positions[middle_node_idx, 2] - z_middle_initial)
        delta_w.append(np.mean(node_positions[start_node_indices, 1]) - y_start_avg_initial)
    return np.array(delta_h), np.array(delta_w), L_val


def split_phases(delta_w):
    """
    Split data into Phase 3 (ΔW increasing) and Phase 4 (ΔW decreasing).
    Returns phase3_mask, phase4_mask, phase3_start_idx, switch_idx.
    """
    delta_delta_w = np.diff(delta_w)
    delta_delta_w_full = np.zeros_like(delta_w)
    delta_delta_w_full[:-1] = delta_delta_w
    if len(delta_delta_w) > 0:
        delta_delta_w_full[-1] = delta_delta_w[-1]
    threshold = 1e-10
    delta_delta_w_sign = np.sign(delta_delta_w_full)
    for i in range(len(delta_delta_w_sign)):
        if abs(delta_delta_w_full[i]) < threshold and i > 0:
            delta_delta_w_sign[i] = delta_delta_w_sign[i - 1]
    min_run = 5
    dw_mag_threshold = max(1e-10, 1e-6 * (np.max(delta_w) - np.min(delta_w) + 1e-12))
    phase3_start_idx = 0
    for i in range(len(delta_delta_w_sign) - min_run):
        if np.all(delta_delta_w_sign[i : i + min_run] > 0) and abs(delta_w[i]) >= dw_mag_threshold:
            phase3_start_idx = i
            break
    negative_deriv_indices = np.where(delta_delta_w_sign < 0)[0]
    positive_deriv_indices = np.where(delta_delta_w_sign > 0)[0]
    if len(negative_deriv_indices) == 0:
        phase3_mask = np.zeros(len(delta_w), dtype=bool)
        phase3_mask[phase3_start_idx:] = True
        phase4_mask = np.zeros(len(delta_w), dtype=bool)
        return phase3_mask, phase4_mask, phase3_start_idx, len(delta_w)
    if len(positive_deriv_indices) == 0:
        phase3_mask = np.zeros(len(delta_w), dtype=bool)
        phase4_mask = np.ones(len(delta_w), dtype=bool)
        return phase3_mask, phase4_mask, 0, 0
    switch_idx = None
    for i in range(1, len(delta_delta_w_sign)):
        if delta_delta_w_sign[i - 1] > 0 and delta_delta_w_sign[i] < 0:
            switch_idx = i
            break
    if switch_idx is None:
        if delta_delta_w_sign[0] > 0:
            switch_idx = negative_deriv_indices[0] if len(negative_deriv_indices) > 0 else len(delta_w)
        else:
            switch_idx = 0
    phase3_mask = np.zeros(len(delta_w), dtype=bool)
    phase3_mask[phase3_start_idx:switch_idx] = True
    phase4_mask = np.zeros(len(delta_w), dtype=bool)
    phase4_mask[switch_idx:] = True
    return phase3_mask, phase4_mask, phase3_start_idx, switch_idx


def plot_midpoint_deflection(pkl_dir, plot_dir, n_nodes_list, energy_model, start_thresh=0.01):
    """
    Plot |H_mid|/L vs ΔW/L (mid-point deflection vs shear) for each discretization.
    Two subplots: ΔW increasing, ΔW decreasing.
    """
    pkl_dir = Path(pkl_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    linestyles = ['-', '--', '-.', ':']
    labels_n = {n: f'n={n}' for n in n_nodes_list}

    fig, (ax_inc, ax_dec) = plt.subplots(1, 2, figsize=(12, 5))

    for idx, n_nodes in enumerate(n_nodes_list):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        pkl_path = pkl_dir / f"shear_homotopy_n{n_nodes}.pkl"
        if not pkl_path.exists():
            print(f"  Skip plot for n={n_nodes}: missing {pkl_path}")
            continue
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            qs = data['qs']
            geo = data['geo']
            cfg = data.get('config', {})
            bc_cfg = cfg.get('boundary_condition', {})
            st = float(bc_cfg.get('start_x_threshold', 0.01))
        except Exception as e:
            print(f"  Error loading {pkl_path}: {e}")
            continue

        q0 = qs[0]
        n_infer = infer_n_nodes(q0, geo)
        middle_node_idx = find_middle_node(q0, n_infer, start_thresh=st)
        start_node_indices = find_start_nodes(q0, n_infer, start_thresh=st)
        delta_h, delta_w, L_val = extract_delta_h_w(
            qs, n_infer, middle_node_idx, start_node_indices
        )
        delta_h_norm = delta_h / L_val
        delta_w_norm = delta_w / L_val
        phase3_mask, phase4_mask, _, _ = split_phases(delta_w)
        label = labels_n[n_nodes]

        if np.any(phase3_mask):
            dw_inc = delta_w_norm[phase3_mask]
            dh_inc = delta_h_norm[phase3_mask]
            if len(dw_inc) > 0:
                ax_inc.plot(
                    dw_inc, np.abs(dh_inc),
                    color=color, linestyle=linestyle, linewidth=2.5, label=label, alpha=0.95
                )
        if np.any(phase4_mask):
            dw_dec = delta_w_norm[phase4_mask]
            dh_dec = delta_h_norm[phase4_mask]
            if len(dw_dec) > 0:
                ax_dec.plot(
                    dw_dec, np.abs(dh_dec),
                    color=color, linestyle=linestyle, linewidth=2.5, label=label, alpha=0.95
                )

    ax_inc.set_xlabel('ΔW/L', fontsize=12)
    ax_inc.set_ylabel(r'$|H_{mid}/L|$', fontsize=12)
    ax_inc.set_title('ΔW increasing', fontsize=12)
    ax_inc.grid(True, alpha=0.3)
    ax_inc.set_ylim(-0.1, 0.35)
    ax_inc.legend(loc='best', fontsize=10)

    ax_dec.set_xlabel('ΔW/L', fontsize=12)
    ax_dec.set_ylabel(r'$|H_{mid}/L|$', fontsize=12)
    ax_dec.set_title('ΔW decreasing', fontsize=12)
    ax_dec.grid(True, alpha=0.3)
    ax_dec.set_ylim(-0.1, 0.35)
    ax_dec.legend(loc='best', fontsize=10)

    fig.suptitle(f"Shear homotopy — {energy_model.capitalize()} Energy Model", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = plot_dir / f"{energy_model}_shear_homotopy_midpoint.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def create_boundary_condition_with_homotopy(cfg, start_nodes, stepper, initial_width):
    """
    Create BC function that:
    - Applies gravity ramp
    - Moves nodes during motion_phases
    - During homotopy phase: interpolates width, calls update_rod_geometry + refresh_rod_params
    """
    bc = cfg['boundary_condition']
    u0 = float(bc['u0'])
    ramp_end = float(bc['gravity_ramp_end_time'])
    g_ramp = np.array(bc['gravity_during_ramp'], dtype=float)
    g_after = np.array(bc['gravity_after_ramp'], dtype=float)
    phases = bc['motion_phases']

    hom = cfg.get('homotopy', {})
    hom_start = float(hom.get('start_time', 7.55))
    hom_end = float(hom.get('end_time', 8.55))
    start_width = hom.get('start_width')
    end_width = float(hom.get('end_width', 0.012))
    if start_width is None:
        start_width = initial_width
    else:
        start_width = float(start_width)

    def move_twist_and_homotopy(robot: dismech.SoftRobot, t: float):
        if t == robot.sim_params.dt:
            return robot

        # Gravity
        if t <= ramp_end:
            robot.env.g = g_ramp.copy()
        else:
            robot.env.g = g_after.copy()

        # Homotopy phase and beyond: set width (interpolate during homotopy, then end_width)
        if t >= hom_start:
            if t < hom_end:
                frac = (t - hom_start) / (hom_end - hom_start) if hom_end > hom_start else 1.0
                width = start_width + frac * (end_width - start_width)
            else:
                width = end_width
            robot.update_rod_geometry(width=width)
            stepper.refresh_rod_params(robot)

        # Motion phases (also apply during/after homotopy for phase 4)
        for ph in phases:
            t_start = float(ph['start_time'])
            t_end = float(ph['end_time'])
            direction = int(ph['direction'])
            reverse = ph.get('reverse', False)
            sign = -1 if reverse else 1
            if t >= t_start and t < t_end:
                robot = robot.move_nodes(start_nodes, u0 * robot.sim_params.dt * sign, direction)
                break

        return robot

    return move_twist_and_homotopy


def main():
    parser = argparse.ArgumentParser(
        description='Shear-induced bifurcation with homotopy (width ramp between phases)'
    )
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--pkl-dir', type=str, required=True, help='Directory for pkl output')
    parser.add_argument('--plot-dir', type=str, required=True, help='Directory for plots')
    parser.add_argument('--nodes', type=int, default=None, help='Override node count')
    parser.add_argument('--energy-model', type=str, default=None,
                        choices=['sano', 'wunderlich', 'kirchhoff', 'sadowsky', 'audoly'],
                        help='Override energy model from config')
    parser.add_argument('--plot', action='store_true',
                        help='If set: run simulation only for missing pkls, then plot mid-point deflection. '
                             'If not set: run all simulations then plot.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    L = float(cfg['geometry']['L'])
    w_by_l_cfg = cfg['geometry']['w_by_l_ratio']
    w_by_l = parse_w_by_l_ratio(w_by_l_cfg[0] if isinstance(w_by_l_cfg, list) else w_by_l_cfg)
    w_initial = w_by_l * L
    h = float(cfg['geometry']['h'])
    shell_h = float(cfg['geometry'].get('shell_h', 0))

    n_nodes_list = args.nodes if args.nodes is not None else cfg.get('nodes', [21])
    if isinstance(n_nodes_list, int):
        n_nodes_list = [n_nodes_list]

    geom = dismech.GeomParams(
        rod_r0=h,
        shell_h=shell_h,
        axs=w_initial * h,
        jxs=w_initial * h**3 / 3,
        ixs1=w_initial * h**3 / 12,
        ixs2=h * w_initial**3 / 12,
    )
    material = dismech.Material(
        density=float(cfg['material']['density']),
        youngs_rod=float(cfg['material']['youngs_rod']),
        youngs_shell=float(cfg['material']['youngs_shell']),
        poisson_rod=float(cfg['material']['poisson_rod']),
        poisson_shell=float(cfg['material']['poisson_shell']),
    )
    sim_params = dismech.SimParams(
        static_sim=bool(cfg['simulation']['static_sim']),
        two_d_sim=bool(cfg['simulation']['two_d_sim']),
        use_mid_edge=bool(cfg['simulation']['use_mid_edge']),
        use_line_search=bool(cfg['simulation']['use_line_search']),
        show_floor=bool(cfg['simulation']['show_floor']),
        log_data=bool(cfg['simulation']['log_data']),
        log_step=int(cfg['simulation']['log_step']),
        dt=float(cfg['simulation']['dt']),
        max_iter=int(cfg['simulation']['max_iter']),
        total_time=float(cfg['simulation']['total_time']),
        plot_step=int(cfg['simulation']['plot_step']),
        tol=float(cfg['simulation']['tol']),
        ftol=float(cfg['simulation']['ftol']),
        dtol=float(cfg['simulation']['dtol']),
    )
    env = dismech.Environment()
    env.add_force('gravity', g=np.array(cfg['environment']['gravity'], dtype=float))

    em_cfg = cfg.get('energy_model', {})
    energy_model = (args.energy_model or em_cfg.get('name', 'sano')).lower()
    sano_zeta = em_cfg.get('sano_zeta')
    wunderlich_W = em_cfg.get('wunderlich_W')
    if sano_zeta is not None:
        sano_zeta = float(sano_zeta)
    if wunderlich_W is not None:
        wunderlich_W = float(wunderlich_W)

    script_dir = Path(__file__).parent
    geometry_dir = script_dir / 'initial_geometry'
    pkl_dir = Path(args.pkl_dir)
    plot_dir = Path(args.plot_dir)
    pkl_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    run_all = not args.plot
    start_thresh = float(cfg['boundary_condition'].get('start_x_threshold', 0.01))

    for n_nodes in n_nodes_list:
        pkl_path = pkl_dir / f"shear_homotopy_n{n_nodes}.pkl"
        if pkl_path.exists() and not run_all:
            print(f"Skip (--plot): {pkl_path.name}")
            continue

        geo_path = ensure_initial_geometry(geometry_dir, n_nodes, L)
        geo = dismech.Geometry.from_txt(str(geo_path))

        robot = dismech.SoftRobot(geom, material, geo, sim_params, env)
        bc_cfg = cfg['boundary_condition']
        node_positions = robot.state.q[robot.node_dof_indices].reshape(-1, 3)
        start_thresh = float(bc_cfg['start_x_threshold'])
        end_thresh = float(bc_cfg['end_x_threshold'])
        start = np.where(node_positions[:, 0] <= start_thresh)[0]
        end = np.where(node_positions[:, 0] >= end_thresh)[0]
        fixed_nodes = np.union1d(start, end)
        robot = robot.fix_nodes(fixed_nodes)

        stepper = dismech.ImplicitEulerTimeStepper(
            robot,
            energy_model=energy_model,
            sano_zeta=sano_zeta,
            wunderlich_W=wunderlich_W,
        )
        stepper.before_step = create_boundary_condition_with_homotopy(
            cfg, start, stepper, initial_width=w_initial
        )

        adt = cfg.get('adaptive_dt', {})
        if adt.get('enabled', True):
            stepper.adaptive_dt = True
            stepper.max_dq_threshold = float(adt.get('max_dq_threshold', 0.1))
            stepper.dt_reduction_factor = float(adt.get('dt_reduction_factor', 0.5))
            base_dt = sim_params.dt
            stepper.min_dt = base_dt / float(adt.get('min_dt_ratio', 1e6))
            stepper.max_dt = base_dt * float(adt.get('max_dt_ratio', 2.0))
            stepper.max_dt_reductions = int(adt.get('max_dt_reductions', 40))

        tr = cfg.get('tracking', {})
        if tr.get('track_forces', True):
            stepper.set_nodes_to_track_forces(fixed_nodes)
        if tr.get('track_elastic_energy', False):
            stepper.enable_elastic_energy_tracking()

        print(f"Running shear homotopy n={n_nodes} ...")
        result = stepper.simulate()
        (robots, tracked_forces_list, tracked_forces_times,
         condition_numbers, condition_number_times,
         elastic_energies, elastic_energy_times,
         material_directors_list, material_directors_times) = result

        qs = np.stack([r.state.q for r in robots])
        save_path = pkl_path
        save_data = {
            'qs': qs,
            'sim_params': sim_params,
            'geom_params': geom,
            'material': material,
            'env': env,
            'geo': geo,
            'fixed_nodes': fixed_nodes,
            'config': cfg,
        }
        if tracked_forces_list:
            save_data['tracked_forces'] = tracked_forces_list
            save_data['tracked_forces_times'] = tracked_forces_times
        if elastic_energies:
            save_data['elastic_energies'] = elastic_energies
            save_data['elastic_energy_times'] = elastic_energy_times
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved {save_path}")

    # Plot mid-point deflection |H_mid|/L vs ΔW/L
    plot_midpoint_deflection(
        pkl_dir, plot_dir, n_nodes_list, energy_model,
        start_thresh=start_thresh,
    )
    print("Done.")


if __name__ == '__main__':
    main()
