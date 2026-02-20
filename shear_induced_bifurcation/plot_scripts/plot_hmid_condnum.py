#!/usr/bin/env python3
"""
Plot |H_mid|/L and Hessian condition number vs ΔW/L from config-driven pkl paths.

Single energy model, 4 pkls -> 4 subplots (2x2). Each subplot: left axis = |H_mid|/L,
right axis = condition number (log scale by default). Config defines per subplot:
pkl, W/L subtitle, increasing/decreasing branch. PKL files must have been saved with
--track-all (condition_numbers, condition_number_times in pkl).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml


def load_pkl_data(pkl_path):
    """Load simulation data from a pkl file (validate_discretization format, with condition numbers)."""
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
        data.get('condition_numbers', None),
        data.get('condition_number_times', None),
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


def run_plot(config_path, plot_dir, out_filename='hmid_condnum.png'):
    """Build 2x2 subplots of |H_mid|/L and condition number vs ΔW/L from config."""
    config, config_dir = load_config(config_path)
    base_dir_ref = config.get('base_dir', '')
    xlim = config.get('xlim', [0, 0.5])
    ylim = config.get('ylim', [-0.05, 0.35])
    ylim_condnum = config.get('ylim_condnum')  # None or [min, max] for right axis
    cond_scale = config.get('cond_scale', 'log').lower()  # 'log' or 'linear'
    if cond_scale not in ('log', 'linear'):
        cond_scale = 'log'
    title = config.get('title', 'H_mid and Condition Number')
    figsize = config.get('figsize', [14, 10])
    wspace = config.get('wspace', 0.28)
    hspace = config.get('hspace', 0.35)
    tight_layout_rect = config.get('tight_layout_rect', [0, 0, 1, 0.82])
    subplots_config = config.get('subplots', [])
    if len(subplots_config) != 4:
        raise ValueError("Config must have exactly 4 entries under 'subplots'.")

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(figsize[0], figsize[1]), gridspec_kw={'wspace': wspace, 'hspace': hspace})
    axes = axes.flatten()

    for subplot_idx in range(4):
        ax = axes[subplot_idx]
        entry = subplots_config[subplot_idx]
        pkl_ref = entry.get('pkl')
        subplot_title = entry.get('subplot_title', f'Subplot {subplot_idx + 1}')
        increasing = entry.get('increasing', True)
        entry_ylim_cond = entry.get('ylim_condnum')
        subplot_ylim_cond = entry_ylim_cond if entry_ylim_cond is not None else ylim_condnum

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
            (qs, _, _, _, _, geo, _, condition_numbers, _) = load_pkl_data(pkl_path)
        except Exception as e:
            print(f"  Error loading {pkl_path}: {e}")
            ax.set_title(subplot_title, fontsize=12, pad=6)
            ax.axis('off')
            continue

        if condition_numbers is None or len(condition_numbers) == 0:
            print(f"  No condition numbers in {pkl_path.name} for subplot {subplot_idx} (run with --track-all?)")
            ax.set_title(subplot_title, fontsize=12, pad=6)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_xlabel('ΔW/L', fontsize=12)
            ax.set_ylabel(r'$|H_{mid}/L|$', fontsize=12)
            ax.grid(True, alpha=0.3)
            continue

        q0 = qs[0]
        n_nodes = infer_n_nodes(q0, geo)
        middle_node_idx = find_middle_node(q0, n_nodes)
        start_node_indices = find_start_nodes(q0, n_nodes)
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
        cond_arr = np.asarray(condition_numbers, dtype=float)
        if len(cond_arr) != len(delta_w):
            print(f"  Condition number length mismatch in {pkl_path.name} for subplot {subplot_idx}")
            ax.set_title(subplot_title, fontsize=12, pad=6)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_xlabel('ΔW/L', fontsize=12)
            ax.set_ylabel(r'$|H_{mid}/L|$', fontsize=12)
            ax.grid(True, alpha=0.3)
            continue

        cond_plot = cond_arr[mask]
        if len(dw_plot) == 0 or len(cond_plot) != len(dw_plot):
            continue

        # Left axis: |H_mid|/L vs ΔW/L
        ax.plot(dw_plot, dh_plot, color='#008B8B', linestyle='-.', linewidth=2, label=r'$|H_{mid}/L|$', alpha=0.9)
        ax.set_xlabel('ΔW/L', fontsize=12)
        ax.set_ylabel(r'$|H_{mid}/L|$', fontsize=12)
        ax.set_title(subplot_title, fontsize=12, pad=6)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.grid(True, alpha=0.3)

        # Right axis: condition number (log or linear)
        ax2 = ax.twinx()
        if cond_scale == 'log':
            # Replace 0 or very small values for log scale
            cond_plot_safe = np.where(cond_plot > 0, cond_plot, np.nan)
            ax2.semilogy(dw_plot, cond_plot_safe, color='#CC0000', linestyle='--', linewidth=1.5, label='Condition Number', alpha=0.8)
        else:
            ax2.plot(dw_plot, cond_plot, color='#CC0000', linestyle='--', linewidth=1.5, label='Condition Number', alpha=0.8)
        ax2.set_ylabel('Condition Number', fontsize=11, color='#CC0000')
        ax2.tick_params(axis='y', labelcolor='#CC0000')
        if subplot_ylim_cond is not None and len(subplot_ylim_cond) == 2:
            ax2.set_ylim(subplot_ylim_cond[0], subplot_ylim_cond[1])
        ax2.grid(True, alpha=0.2, linestyle=':')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    # Single legend at top, side-by-side (same as plot_hmid.py / plot_hmid_fy.py)
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
    plt.tight_layout(rect=tight_layout_rect)
    out_path = plot_dir / out_filename
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Plot |H_mid|/L and condition number vs ΔW/L (4 subplots) from config.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML. Default: validate_discretization/plot_scripts/config_hmid_condnum.yaml'
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
        default='hmid_condnum.png',
        help='Output filename'
    )
    args = parser.parse_args()
    script_dir = Path(__file__).parent.resolve()
    config_path = Path(args.config) if args.config else script_dir / 'config_hmid_condnum.yaml'
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    run_plot(config_path, args.plot_dir, out_filename=args.output)


if __name__ == '__main__':
    main()
