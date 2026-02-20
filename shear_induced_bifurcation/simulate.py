#!/usr/bin/env python3
"""
Shear-induced bifurcation simulation with config-driven parameters and optional tracking.

All geometry, material, simulation, and tracking options are read from a YAML config file.
Tracking: forces (boundary or all nodes), Hessian condition number, total elastic energy,
and material directors (a1, a2, m1, m2) can be enabled/disabled in the config.

Usage:
  python simulate.py --config simulate_shear_induced_bifurcation_config.yaml --pkl-dir out/pkls --plot-dir out/plots
  Optional: --wbyl 1/14 1/18  --nodes 21 63  (override config)
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
import re
import yaml
from fractions import Fraction


def load_config(config_path):
    """Load YAML config. Supports optional C-style comments by stripping them if needed."""
    path = Path(config_path)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data


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


def setup_geometry_from_config(cfg):
    """Build GeomParams from config geometry section."""
    g = cfg['geometry']
    L = float(g['L'])
    w_by_l = g['w_by_l_ratio']
    if isinstance(w_by_l, list):
        w_by_l = w_by_l[0]  # use first for single-run
    w = float(w_by_l) * L
    h = float(g['h'])
    A = w * h
    I1 = w * h**3 / 12
    I2 = h * w**3 / 12
    J = w * h**3 / 3
    return dismech.GeomParams(
        rod_r0=h,
        shell_h=float(g.get('shell_h', 0)),
        axs=A,
        jxs=J,
        ixs1=I1,
        ixs2=I2
    ), L


def setup_material_from_config(cfg):
    """Build Material from config material section."""
    m = cfg['material']
    return dismech.Material(
        density=float(m['density']),
        youngs_rod=float(m['youngs_rod']),
        youngs_shell=float(m['youngs_shell']),
        poisson_rod=float(m['poisson_rod']),
        poisson_shell=float(m['poisson_shell'])
    )


def setup_simulation_params_from_config(cfg):
    """Build SimParams from config simulation section."""
    s = cfg['simulation']
    return dismech.SimParams(
        static_sim=bool(s['static_sim']),
        two_d_sim=bool(s['two_d_sim']),
        use_mid_edge=bool(s['use_mid_edge']),
        use_line_search=bool(s['use_line_search']),
        show_floor=bool(s['show_floor']),
        log_data=bool(s['log_data']),
        log_step=int(s['log_step']),
        dt=float(s['dt']),
        max_iter=int(s['max_iter']),
        total_time=float(s['total_time']),
        plot_step=int(s['plot_step']),
        tol=float(s['tol']),
        ftol=float(s['ftol']),
        dtol=float(s['dtol'])
    )


def setup_environment_from_config(cfg):
    """Build Environment from config environment section."""
    env = dismech.Environment()
    g = cfg['environment']['gravity']
    env.add_force('gravity', g=np.array(g, dtype=float))
    return env


def load_geometry(geometry_file):
    """Load geometry from file."""
    return dismech.Geometry.from_txt(geometry_file)


def create_boundary_condition_function_from_config(cfg, start_nodes):
    """Create the time-dependent BC from config boundary_condition section.
    Each phase has start_time, end_time, direction (0=x, 1=y, 2=z). Optional 'reverse': true
    means use -u0 instead of +u0 for that phase (e.g. to reverse shear).
    """
    bc = cfg['boundary_condition']
    u0 = float(bc['u0'])
    ramp_end = float(bc['gravity_ramp_end_time'])
    g_ramp = np.array(bc['gravity_during_ramp'], dtype=float)
    g_after = np.array(bc['gravity_after_ramp'], dtype=float)
    phases = bc['motion_phases']

    def move_and_twist(robot: dismech.SoftRobot, t: float):
        if t == robot.sim_params.dt:
            return robot
        if t <= ramp_end:
            robot.env.g = g_ramp.copy()
        else:
            robot.env.g = g_after.copy()
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

    return move_and_twist


def run_simulation(geom, material, geo, sim_params, env, cfg, w_by_l_ratio):
    """Run simulation with tracking options from config."""
    robot = dismech.SoftRobot(geom, material, geo, sim_params, env)
    bc_cfg = cfg['boundary_condition']
    node_positions = robot.state.q[robot.node_dof_indices].reshape(-1, 3)
    start_thresh = float(bc_cfg['start_x_threshold'])
    end_thresh = float(bc_cfg['end_x_threshold'])
    start = np.where(node_positions[:, 0] <= start_thresh)[0]
    end = np.where(node_positions[:, 0] >= end_thresh)[0]
    fixed_nodes = np.union1d(start, end)
    robot = robot.fix_nodes(fixed_nodes)

    em_cfg = cfg.get('energy_model', {})
    energy_model = em_cfg.get('name', 'sano')
    sano_zeta = em_cfg.get('sano_zeta')
    wunderlich_W = em_cfg.get('wunderlich_W')
    if sano_zeta is not None:
        sano_zeta = float(sano_zeta)
    if wunderlich_W is not None:
        wunderlich_W = float(wunderlich_W)

    stepper = dismech.ImplicitEulerTimeStepper(
        robot,
        energy_model=energy_model,
        sano_zeta=sano_zeta,
        wunderlich_W=wunderlich_W,
    )
    stepper.before_step = create_boundary_condition_function_from_config(cfg, start)

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
    track_forces = tr.get('track_forces', True)
    track_forces_all_nodes = tr.get('track_forces_all_nodes', False)
    if track_forces:
        if track_forces_all_nodes:
            n_nodes_all = node_positions.shape[0]
            stepper.set_nodes_to_track_forces(np.arange(n_nodes_all, dtype=int))
        else:
            stepper.set_nodes_to_track_forces(fixed_nodes)
    if tr.get('track_condition_number', False):
        stepper.enable_condition_number_tracking()
    if tr.get('track_elastic_energy', False):
        stepper.enable_elastic_energy_tracking()
    if tr.get('track_material_directors', False):
        stepper.enable_material_director_tracking()

    result = stepper.simulate()
    (robots, tracked_forces_list, tracked_forces_times,
     condition_numbers, condition_number_times,
     elastic_energies, elastic_energy_times,
     material_directors_list, material_directors_times) = result

    if not track_forces:
        tracked_forces_list = []
        tracked_forces_times = []
    if not tr.get('track_condition_number', False):
        condition_numbers = None
        condition_number_times = None
    if not tr.get('track_elastic_energy', False):
        elastic_energies = []
        elastic_energy_times = []
    if not tr.get('track_material_directors', False):
        material_directors_list = []
        material_directors_times = []

    qs = np.stack([r.state.q for r in robots])
    return (robots, qs, fixed_nodes,
            tracked_forces_list, tracked_forces_times,
            condition_numbers, condition_number_times,
            elastic_energies, elastic_energy_times,
            material_directors_list, material_directors_times)


def save_results(save_path, qs, geom, material, geo, sim_params, env, fixed_nodes,
                tracked_forces_list=None, tracked_forces_times=None,
                condition_numbers=None, condition_number_times=None,
                elastic_energies=None, elastic_energy_times=None,
                material_directors_list=None, material_directors_times=None):
    """Save simulation results to disk, including all optional tracking data."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        'qs': qs,
        'sim_params': sim_params,
        'geom_params': geom,
        'material': material,
        'env': env,
        'geo': geo,
        'fixed_nodes': fixed_nodes,
    }
    if tracked_forces_list is not None:
        save_data['tracked_forces'] = tracked_forces_list
    if tracked_forces_times is not None:
        save_data['tracked_forces_times'] = tracked_forces_times
    if condition_numbers is not None:
        save_data['condition_numbers'] = condition_numbers
    if condition_number_times is not None:
        save_data['condition_number_times'] = condition_number_times
    if elastic_energies is not None and len(elastic_energies) > 0:
        save_data['elastic_energies'] = elastic_energies
    if elastic_energy_times is not None and len(elastic_energy_times) > 0:
        save_data['elastic_energy_times'] = elastic_energy_times
    if material_directors_list is not None and len(material_directors_list) > 0:
        save_data['material_directors_list'] = material_directors_list
    if material_directors_times is not None and len(material_directors_times) > 0:
        save_data['material_directors_times'] = material_directors_times
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)


def parse_w_by_l_ratio(ratio_str):
    """Parse W/L ratio string like '1/14' into float."""
    if isinstance(ratio_str, (int, float)):
        return float(ratio_str)
    s = str(ratio_str)
    if '/' in s:
        parts = s.split('/')
        return float(parts[0]) / float(parts[1])
    return float(s)


def w_by_l_to_filename(w_by_l_ratio, n_nodes=None):
    """Convert W/L ratio to filename base like '1b14', optionally with _nXX suffix."""
    frac = Fraction(w_by_l_ratio).limit_denominator(1000)
    base = f"1b{frac.denominator}"
    if n_nodes is not None:
        return f"{base}_n{n_nodes}.pkl"
    return f"{base}.pkl"


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
    return q[: 3 * n_nodes].reshape(-1, 3)


def find_middle_node(q0, n_nodes):
    node_positions = get_node_positions(q0, n_nodes)
    x_coords = node_positions[:, 0]
    L_val = x_coords.max()
    return np.argmin(np.abs(x_coords - L_val / 2.0))


def find_start_nodes(q0, n_nodes, start_x_threshold=0.01):
    node_positions = get_node_positions(q0, n_nodes)
    return np.where(node_positions[:, 0] <= start_x_threshold)[0]


def extract_delta_h_w(qs, n_nodes, middle_node_idx, start_node_indices):
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
        switch_idx = negative_deriv_indices[0] if len(negative_deriv_indices) > 0 else len(delta_w)
        if delta_delta_w_sign[0] <= 0:
            switch_idx = 0
    phase3_mask = np.zeros(len(delta_w), dtype=bool)
    phase3_mask[phase3_start_idx:switch_idx] = True
    phase4_mask = np.zeros(len(delta_w), dtype=bool)
    phase4_mask[switch_idx:] = True
    return phase3_mask, phase4_mask, phase3_start_idx, switch_idx


def plot_discretization_validation(pkl_dir, plot_dir, w_by_l_ratios, energy_model, n_nodes_list, cfg):
    """Plot |H_mid|/L vs ΔW/L for each W/L and discretization (same as simulate_track)."""
    pkl_dir = Path(pkl_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    start_thresh = float(cfg.get('boundary_condition', {}).get('start_x_threshold', 0.01))
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    linestyles = ['-', '--', '-.', ':']
    def style_for_idx(i):
        return colors[i % len(colors)], linestyles[i % len(linestyles)]
    labels_n = {n: f'n={n}' for n in n_nodes_list}

    for w_by_l_ratio in w_by_l_ratios:
        frac = Fraction(w_by_l_ratio).limit_denominator(1000)
        wl_label = f"W/L = 1/{frac.denominator}"
        fig, (ax_inc, ax_dec) = plt.subplots(1, 2, figsize=(12, 5))

        for idx, n_nodes in enumerate(n_nodes_list):
            color, linestyle = style_for_idx(idx)
            pkl_name = w_by_l_to_filename(w_by_l_ratio, n_nodes)
            pkl_path = pkl_dir / pkl_name
            if not pkl_path.exists():
                print(f"  Skip plot for {wl_label} n={n_nodes}: missing {pkl_path}")
                continue
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                qs = data['qs']
                geo = data['geo']
                fixed_nodes = data['fixed_nodes']
            except Exception as e:
                print(f"  Error loading {pkl_path}: {e}")
                continue

            q0 = qs[0]
            n_infer = infer_n_nodes(q0, geo)
            middle_node_idx = find_middle_node(q0, n_infer)
            start_node_indices = find_start_nodes(q0, n_infer, start_x_threshold=start_thresh)
            delta_h, delta_w, L_val = extract_delta_h_w(qs, n_infer, middle_node_idx, start_node_indices)
            delta_h_norm = delta_h / L_val
            delta_w_norm = delta_w / L_val
            phase3_mask, phase4_mask, _, _ = split_phases(delta_w)
            label = labels_n[n_nodes]
            if np.any(phase3_mask):
                dw_inc = delta_w_norm[phase3_mask]
                dh_inc = delta_h_norm[phase3_mask]
                if len(dw_inc) > 0:
                    ax_inc.plot(dw_inc, np.abs(dh_inc), color=color, linestyle=linestyle, linewidth=2.5, label=label, alpha=0.95)
            if np.any(phase4_mask):
                dw_dec = delta_w_norm[phase4_mask]
                dh_dec = delta_h_norm[phase4_mask]
                if len(dw_dec) > 0:
                    ax_dec.plot(dw_dec, np.abs(dh_dec), color=color, linestyle=linestyle, linewidth=2.5, label=label, alpha=0.95)

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
        fig.suptitle(f"{wl_label} — {energy_model.capitalize()} Energy Model", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        safe_name = wl_label.replace('/', '_').replace(' ', '_')
        out_path = plot_dir / f"{energy_model}_discretization_validation_{safe_name}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Shear-induced bifurcation: config-driven simulation with tracking')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--pkl-dir', type=str, required=True, help='Directory to save/load pkl files')
    parser.add_argument('--plot-dir', type=str, required=True, help='Directory to save plots')
    parser.add_argument('--wbyl', type=str, nargs='+', default=None, help='Override W/L ratios (e.g. 1/14 1/18)')
    parser.add_argument('--nodes', type=int, nargs='+', default=None, help='Override node counts (e.g. 21 63)')
    parser.add_argument('--energy-model', type=str, default=None,
                        choices=['sano', 'wunderlich', 'kirchhoff', 'sadowsky', 'audoly'],
                        help='Override energy model from config')
    parser.add_argument('--plot', action='store_true', help='Only run missing cases then plot; else rerun all then plot')
    args = parser.parse_args()

    cfg = load_config(args.config)
    L = float(cfg['geometry']['L'])
    w_by_l_ratio_cfg = cfg['geometry']['w_by_l_ratio']
    if isinstance(w_by_l_ratio_cfg, list):
        w_by_l_ratios = [float(x) if isinstance(x, (int, float)) else parse_w_by_l_ratio(x) for x in w_by_l_ratio_cfg]
    else:
        w_by_l_ratios = [parse_w_by_l_ratio(w_by_l_ratio_cfg)]

    if args.wbyl:
        w_by_l_ratios = [parse_w_by_l_ratio(s) for s in args.wbyl]
    n_nodes_list = args.nodes if args.nodes is not None else cfg.get('nodes', [21, 63, 105])
    if isinstance(n_nodes_list, int):
        n_nodes_list = [n_nodes_list]
    n_nodes_list = sorted(n_nodes_list)

    script_dir = Path(__file__).parent
    geometry_dir = script_dir / 'initial_geometry'
    for n in n_nodes_list:
        ensure_initial_geometry(geometry_dir, n, L)

    material = setup_material_from_config(cfg)
    sim_params = setup_simulation_params_from_config(cfg)
    env = setup_environment_from_config(cfg)
    pkl_dir = Path(args.pkl_dir)
    pkl_dir.mkdir(parents=True, exist_ok=True)
    run_all = not args.plot
    energy_model = (args.energy_model or cfg.get('energy_model', {}).get('name', 'sano')).lower()
    if args.energy_model is not None:
        if 'energy_model' not in cfg:
            cfg['energy_model'] = {}
        cfg['energy_model']['name'] = energy_model

    shell_h = float(cfg['geometry'].get('shell_h', 0))
    h = float(cfg['geometry']['h'])
    for w_by_l_ratio in w_by_l_ratios:
        w = w_by_l_ratio * L
        geom = dismech.GeomParams(rod_r0=h, shell_h=shell_h, axs=w * h, jxs=w * h**3 / 3,
                                  ixs1=w * h**3 / 12, ixs2=h * w**3 / 12)
        for n_nodes in n_nodes_list:
            pkl_path = pkl_dir / w_by_l_to_filename(w_by_l_ratio, n_nodes)
            if pkl_path.exists() and not run_all:
                print(f"Skip (--plot): {pkl_path.name}")
                continue
            geo_path = ensure_initial_geometry(geometry_dir, n_nodes, L)
            geo = load_geometry(str(geo_path))
            print(f"Running W/L={w_by_l_ratio} n={n_nodes} -> {pkl_path.name}")
            try:
                (robots, qs, fixed_nodes,
                 tracked_forces_list, tracked_forces_times,
                 condition_numbers, condition_number_times,
                 elastic_energies, elastic_energy_times,
                 material_directors_list, material_directors_times) = run_simulation(
                    geom, material, geo, sim_params, env, cfg, w_by_l_ratio)
                save_results(
                    pkl_path, qs, geom, material, geo, sim_params, env, fixed_nodes,
                    tracked_forces_list, tracked_forces_times,
                    condition_numbers, condition_number_times,
                    elastic_energies, elastic_energy_times,
                    material_directors_list, material_directors_times,
                )
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

    plot_discretization_validation(
        args.pkl_dir, args.plot_dir, w_by_l_ratios, energy_model, n_nodes_list, cfg)
    print("Done.")


if __name__ == '__main__':
    main()
