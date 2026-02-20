#!/usr/bin/env python3
"""
Plot |H_mid|/L and normalized Fy (FL/(Yh³)) vs ΔW/L from config-driven pkl paths.

Single energy model, 4 pkls -> 4 subplots (2x2). Each subplot: left axis = |H_mid|/L,
right axis = Fy normalized by L/(Y*h³). Config defines per subplot: pkl, W/L subtitle,
increasing/decreasing branch, and which side (start or end) for forces.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml


def load_pkl_data(pkl_path):
    """Load simulation data from a pkl file (validate_discretization format)."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return (
        data['qs'],
        data.get('sim_params'),
        data.get('geom_params'),
        data.get('material'),
        data.get('env'),
        data['geo'],
        data['fixed_nodes'],
        data.get('tracked_forces', None),
        data.get('tracked_forces_times', None),
    )


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


def find_middle_node(q0, n_nodes):
    """Find the middle node (closest to L/2 along x-axis)."""
    node_positions = get_node_positions(q0, n_nodes)
    x_coords = node_positions[:, 0]
    L_val = x_coords.max()
    L_half = L_val / 2.0
    return np.argmin(np.abs(x_coords - L_half))


def find_start_nodes(q0, n_nodes):
    """Find start nodes (nodes at x <= 0.01)."""
    node_positions = get_node_positions(q0, n_nodes)
    return np.where(node_positions[:, 0] <= 0.01)[0]


def find_end_nodes(q0, n_nodes):
    """Find end nodes (nodes at x >= 0.09)."""
    node_positions = get_node_positions(q0, n_nodes)
    return np.where(node_positions[:, 0] >= 0.09)[0]


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
    """Split into Phase 3 (ΔW increasing) and Phase 4 (ΔW decreasing). Returns phase3_mask, phase4_mask, phase3_start_idx, switch_idx."""
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


def compute_summed_forces_at_nodes(tracked_forces_list, node_indices, geo):
    """
    Sum reaction forces (and edge torques) at the given nodes.
    Returns array of shape (n_timesteps, 4): [Fx_sum, Fy_sum, Fz_sum, Mθ_sum] per timestep.
    """
    forces_summed = []
    edges = None
    if hasattr(geo, 'rod_edges') and geo.rod_edges.size > 0:
        edges = geo.rod_edges
    elif hasattr(geo, 'edges') and geo.edges.size > 0:
        edges = geo.edges
    node_to_edges = {}
    if edges is not None:
        for edge_idx in range(len(edges)):
            edge = edges[edge_idx]
            if len(edge) >= 2:
                node0, node1 = int(edge[0]), int(edge[1])
                if node0 not in node_to_edges:
                    node_to_edges[node0] = []
                if node1 not in node_to_edges:
                    node_to_edges[node1] = []
                node_to_edges[node0].append(edge_idx)
                node_to_edges[node1].append(edge_idx)
    for tracked_forces in tracked_forces_list:
        node_forces_dict = tracked_forces.get('node_forces', {})
        edge_torques_dict = tracked_forces.get('edge_torques', {})
        Fx_sum = Fy_sum = Fz_sum = Mθ_sum = 0.0
        unique_edge_indices = set()
        for node_idx in node_indices:
            if node_idx in node_forces_dict:
                nf = node_forces_dict[node_idx]
                Fx_sum += nf[0]
                Fy_sum += nf[1]
                Fz_sum += nf[2]
            if node_idx in node_to_edges:
                for edge_idx in node_to_edges[node_idx]:
                    unique_edge_indices.add(edge_idx)
        for edge_idx in unique_edge_indices:
            if edge_idx in edge_torques_dict:
                Mθ_sum += edge_torques_dict[edge_idx]
        forces_summed.append(np.array([Fx_sum, Fy_sum, Fz_sum, Mθ_sum]))
    return np.array(forces_summed)


def load_config(config_path):
    """Load YAML config and return (config dict, config_dir for resolving paths)."""
    config_path = Path(config_path).resolve()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config_dir = config_path.parent
    return config, config_dir


def resolve_pkl_path(pkl_ref, config_dir, base_dir_ref):
    """Resolve pkl path: if absolute, use as-is; else relative to config_dir / base_dir."""
    p = Path(pkl_ref)
    if p.is_absolute():
        return p
    base = config_dir / base_dir_ref if base_dir_ref else config_dir
    return (base / pkl_ref).resolve()


def run_plot(config_path, plot_dir, out_filename='hmid_fy.png'):
    """Build 2x2 subplots of |H_mid|/L and normalized Fy vs ΔW/L from config."""
    config, config_dir = load_config(config_path)
    base_dir_ref = config.get('base_dir', '')
    xlim = config.get('xlim', [0, 0.5])
    ylim = config.get('ylim', [-0.05, 0.35])
    ylim_fy = config.get('ylim_fy')  # None or [min, max] for right axis
    title = config.get('title', 'H_mid and Fy')
    figsize = config.get('figsize', [14, 10])
    wspace = config.get('wspace', 0.28)
    hspace = config.get('hspace', 0.35)
    tight_layout_rect = config.get('tight_layout_rect', [0, 0, 1, 0.82])
    shear_resultant = config.get('shear_resultant', False)  # False = Fy only, True = sqrt(Fy² + Fz²)
    subplots_config = config.get('subplots', [])
    if len(subplots_config) != 4:
        raise ValueError("Config must have exactly 4 entries under 'subplots'.")

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(figsize[0], figsize[1]), gridspec_kw={'wspace': wspace, 'hspace': hspace})
    axes = axes.flatten()
    subplot_data = [None] * 4

    for subplot_idx in range(4):
        ax = axes[subplot_idx]
        entry = subplots_config[subplot_idx]
        pkl_ref = entry.get('pkl')
        subplot_title = entry.get('subplot_title', f'Subplot {subplot_idx + 1}')
        increasing = entry.get('increasing', True)
        force_side = entry.get('force_side', 'start').lower()
        if force_side not in ('start', 'end'):
            force_side = 'start'
        # Per-subplot ylim for Fy (right axis); fall back to global ylim_fy
        entry_ylim_fy = entry.get('ylim_fy')
        subplot_ylim_fy = entry_ylim_fy if entry_ylim_fy is not None else ylim_fy

        if not pkl_ref:
            ax.set_title(subplot_title, fontsize=12, pad=6)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_xlabel('ΔW/L', fontsize=12)
            ax.set_ylabel(r'$|H_{mid}/L|$', fontsize=12)
            ax.grid(True, alpha=0.3)
            continue

        pkl_path = resolve_pkl_path(pkl_ref, config_dir, base_dir_ref)
        if not pkl_path.exists():
            print(f"  Skip subplot {subplot_idx}: not found {pkl_path}")
            ax.set_title(subplot_title, fontsize=12, pad=6)
            ax.axis('off')
            continue

        try:
            (qs, _, geom_params, material, _, geo, _, tracked_forces_list, _) = load_pkl_data(pkl_path)
        except Exception as e:
            print(f"  Error loading {pkl_path}: {e}")
            ax.set_title(subplot_title, fontsize=12, pad=6)
            ax.axis('off')
            continue

        q0 = qs[0]
        n_nodes = infer_n_nodes(q0, geo)
        middle_node_idx = find_middle_node(q0, n_nodes)
        start_node_indices = find_start_nodes(q0, n_nodes)
        end_node_indices = find_end_nodes(q0, n_nodes)
        delta_h, delta_w, L_val = extract_delta_h_w(qs, n_nodes, middle_node_idx, start_node_indices)
        delta_h_norm = delta_h / L_val
        delta_w_norm = delta_w / L_val
        phase3_mask, phase4_mask, _, _ = split_phases(delta_w)

        mask = phase3_mask if increasing else phase4_mask
        if not np.any(mask):
            ax.set_title(subplot_title, fontsize=12, pad=6)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_xlabel('ΔW/L', fontsize=12)
            ax.set_ylabel(r'$|H_{mid}/L|$', fontsize=12)
            ax.grid(True, alpha=0.3)
            continue

        dw_plot = delta_w_norm[mask]
        dh_plot = np.abs(delta_h_norm[mask])
        if len(dw_plot) == 0:
            continue

        # Left axis: |H_mid|/L vs ΔW/L
        ax.plot(dw_plot, dh_plot, color='#008B8B', linestyle='-.', linewidth=2, label=r'$|H_{mid}/L|$', alpha=0.9)
        ax.set_xlabel('ΔW/L', fontsize=12)
        ax.set_ylabel(r'$|H_{mid}/L|$', fontsize=12)
        ax.set_title(subplot_title, fontsize=12, pad=6)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.grid(True, alpha=0.3)

        # Right axis: normalized Fy (only Fy, normalized by L/(Y*h³))
        fy_x, fy_y, ylim_fy_save = None, None, subplot_ylim_fy
        if tracked_forces_list is not None and len(tracked_forces_list) == len(delta_w):
            node_indices = start_node_indices if force_side == 'start' else end_node_indices
            forces_summed = compute_summed_forces_at_nodes(tracked_forces_list, node_indices, geo)
            h = float(geom_params.rod_r0) if hasattr(geom_params, 'rod_r0') else 0.001
            Y = float(material.youngs_rod) if hasattr(material, 'youngs_rod') else 1e10
            force_norm_factor = L_val / (Y * h**3)
            forces_normalized = forces_summed * force_norm_factor
            if shear_resultant:
                shear_plot = np.sqrt(forces_normalized[mask, 1]**2 + forces_normalized[mask, 2]**2)
            else:
                shear_plot = np.abs(forces_normalized[mask, 1])

            if len(shear_plot) == len(dw_plot):
                fy_x, fy_y = np.asarray(dw_plot), np.asarray(shear_plot)
                ax2 = ax.twinx()
                ax2.plot(dw_plot, shear_plot, color='#CC0000', linestyle='--', linewidth=1.5, label=r'$\hat{F}_{shear}$', alpha=0.8)
                ax2.set_ylabel(r'$\hat{F}_{shear}$ (FL/(Yh³))', fontsize=11, color='#CC0000')
                ax2.tick_params(axis='y', labelcolor='#CC0000')
                if subplot_ylim_fy is not None and len(subplot_ylim_fy) == 2:
                    ax2.set_ylim(subplot_ylim_fy[0], subplot_ylim_fy[1])
                ax2.grid(True, alpha=0.2, linestyle=':')
        else:
            if tracked_forces_list is None or len(tracked_forces_list) == 0:
                print(f"  No tracked forces in {pkl_path.name} for subplot {subplot_idx}")
        subplot_data[subplot_idx] = {
            'subplot_title': subplot_title,
            'hmid_x': np.asarray(dw_plot), 'hmid_y': np.asarray(dh_plot),
            'fy_x': fy_x, 'fy_y': fy_y, 'ylim_fy': ylim_fy_save,
        }

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    # Single legend at top, side-by-side (same as plot_hmid.py)
    all_handles = []
    all_labels = []
    seen = set()
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label and label not in seen:
                all_handles.append(handle)
                all_labels.append(label)
                seen.add(label)
    if all_handles and all_labels:
        ncol_legend = min(len(all_handles), 6)
        fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 0.96),
                   ncol=ncol_legend, fontsize=11, frameon=True, fancybox=True, shadow=True,
                   columnspacing=1.5, handlelength=2.0, handletextpad=0.5)
    reproduce_pkl = config.get('reproduce_pkl', False)
    reproduce_path_ref = config.get('reproduce_pkl_path', '')
    if reproduce_pkl and reproduce_path_ref:
        out_reproduce = resolve_pkl_path(reproduce_path_ref, config_dir, '')
        out_reproduce = Path(out_reproduce)
        out_reproduce.parent.mkdir(parents=True, exist_ok=True)
        plot_config = {
            'title': title, 'xlim': xlim, 'ylim': ylim, 'ylim_fy': ylim_fy,
            'figsize': figsize, 'wspace': wspace, 'hspace': hspace,
            'tight_layout_rect': tight_layout_rect,
        }
        payload = {'version': 1, 'subplots': subplot_data, 'plot_config': plot_config}
        with open(out_reproduce, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved reproduce pkl: {out_reproduce}")
    plt.tight_layout(rect=tight_layout_rect)
    out_path = plot_dir / out_filename
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Plot |H_mid|/L and normalized Fy vs ΔW/L (4 subplots) from config.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML. Default: validate_discretization/plot_scripts/config_hmid_fy.yaml'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        required=True,
        help='Directory to save the output plot'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='hmid_fy.png',
        help='Output filename'
    )
    args = parser.parse_args()
    script_dir = Path(__file__).parent.resolve()
    config_path = Path(args.config) if args.config else script_dir / 'config_hmid_fy.yaml'
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    run_plot(config_path, args.plot_dir, out_filename=args.output)


if __name__ == '__main__':
    main()
