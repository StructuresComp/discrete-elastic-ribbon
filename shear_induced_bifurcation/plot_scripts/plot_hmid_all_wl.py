#!/usr/bin/env python3
"""
Plot |H_mid|/L vs ΔW/L for all W/L (one series per pkl) in a single axes.

Uses the same series structure as config_hmid_fy subplots: pkl, subplot_title (legend label),
increasing (true = phase 3, false = phase 4). Different color and line style per series.
Thick lines (linewidth 2.5), legend at top. All options in config.
"""

import argparse
import sys
from pathlib import Path

# Allow loading pkls that contain dismech objects (validate_discretization and homotopy pkls)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root / 'src'))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import yaml
from matplotlib.patches import Patch


def load_pkl_geometry(pkl_path):
    """Load qs and geo from a pkl (validate_discretization format)."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['qs'], data['geo']


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


def find_start_nodes(q0, n_nodes, start_x_threshold=0.01):
    """Find start nodes (nodes at x <= start_x_threshold)."""
    node_positions = get_node_positions(q0, n_nodes)
    return np.where(node_positions[:, 0] <= start_x_threshold)[0]


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


# Default colors and linestyles (one per series index)
DEFAULT_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
DEFAULT_LINESTYLES = ['-', '--', '-.', ':']


def run_plot(config_path, plot_dir, out_filename='hmid_all_wl.png'):
    """Plot |H_mid|/L vs ΔW/L for all series in one axes."""
    config, config_dir = load_config(config_path)
    base_dir_ref = config.get('base_dir', '')
    labels = config.get('labels', {})
    xlabel = labels.get('xaxis', r'$\Delta W / L$')
    ylabel = labels.get('yaxis', r'$|H_{mid}/L|$')
    fonts = config.get('fonts', {})
    font_text = fonts.get('text', 'CMR10')
    font_math = fonts.get('math', 'CMMI10')
    matplotlib.rcParams['font.family'] = font_text
    matplotlib.rcParams['mathtext.fontset'] = 'cm' if font_math == 'CMMI10' else font_math
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True
    matplotlib.rcParams['axes.unicode_minus'] = False

    title = config.get('title', 'H_mid (all W/L)')
    figsize = config.get('figsize', [10, 6])
    tight_layout_rect = config.get('tight_layout_rect', [0, 0, 1, 0.90])
    suptitle_y = config.get('suptitle_y', 0.99)
    legend_bbox_y = config.get('legend_bbox_y', 0.94)
    homotopy_legend_bbox_y = config.get('homotopy_legend_bbox_y', 0.88)
    xlim = config.get('xlim', [0, 0.5])
    ylim = config.get('ylim', [-0.05, 0.35])
    linewidth = config.get('linewidth', 2.5)
    alpha = config.get('alpha', 0.95)
    colors = config.get('colors', DEFAULT_COLORS)
    linestyles = config.get('linestyles', DEFAULT_LINESTYLES)
    homotopy_colors = config.get('homotopy_colors', ['#7B2CBF', '#00B4D8'])
    homotopy_linestyles = config.get('homotopy_linestyles', ['--', '-.'])
    fontsize_labels = config.get('fontsize_labels', 12)
    fontsize_suptitle = config.get('fontsize_suptitle', 14)
    fontsize_legend = config.get('fontsize_legend', 11)

    series = config.get('series', [])
    homotopy_series = config.get('homotopy_series', [])
    homotopy_base_dir_ref = config.get('homotopy_base_dir', '')
    if not series:
        print("Config has no 'series' entries.")
        return None

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Clamp tight_layout top to 0-1 (matplotlib expects normalized coords)
    tight_layout_rect = list(tight_layout_rect)
    tight_layout_rect[3] = min(max(float(tight_layout_rect[3]), 0), 1)

    fig, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]))
    all_handles = []
    all_labels = []
    main_curve_data = []
    homotopy_curve_data = []

    for idx, entry in enumerate(series):
        pkl_ref = entry.get('pkl')
        label = entry.get('subplot_title', f'W/L #{idx+1}')
        increasing = entry.get('increasing', False)
        if not pkl_ref:
            continue
        pkl_path = resolve_pkl_path(pkl_ref, config_dir, base_dir_ref)
        if not pkl_path.exists():
            print(f"  Skip {label}: not found {pkl_path}")
            continue
        try:
            qs, geo = load_pkl_geometry(pkl_path)
        except Exception as e:
            print(f"  Error loading {pkl_path}: {e}")
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
            continue
        dw_plot = delta_w_norm[mask]
        dh_plot = np.abs(delta_h_norm[mask])
        if len(dw_plot) == 0:
            continue
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        h, = ax.plot(dw_plot, dh_plot, color=color, linestyle=linestyle, linewidth=linewidth, label=label, alpha=alpha)
        all_handles.append(h)
        all_labels.append(label)
        main_curve_data.append({'label': label, 'x': np.asarray(dw_plot), 'y': np.asarray(dh_plot), 'color': color, 'linestyle': linestyle})

    # Homotopy-aided series: same axes, phase 4 (decreasing ΔW), second legend box
    homotopy_handles = []
    homotopy_labels = []
    for idx, entry in enumerate(homotopy_series):
        pkl_ref = entry.get('pkl')
        label = entry.get('subplot_title', f'Homotopy #{idx+1}')
        if not pkl_ref:
            continue
        base_ref = homotopy_base_dir_ref if homotopy_base_dir_ref else base_dir_ref
        pkl_path = resolve_pkl_path(pkl_ref, config_dir, base_ref)
        if not pkl_path.exists():
            print(f"  Skip homotopy {label}: not found {pkl_path}")
            continue
        try:
            qs, geo = load_pkl_geometry(pkl_path)
        except Exception as e:
            print(f"  Error loading homotopy {pkl_path}: {e}")
            continue
        q0 = qs[0]
        n_nodes = infer_n_nodes(q0, geo)
        middle_node_idx = find_middle_node(q0, n_nodes)
        start_node_indices = find_start_nodes(q0, n_nodes)
        delta_h, delta_w, L_val = extract_delta_h_w(qs, n_nodes, middle_node_idx, start_node_indices)
        delta_h_norm = delta_h / L_val
        delta_w_norm = delta_w / L_val
        phase3_mask, phase4_mask, _, _ = split_phases(delta_w)
        mask = phase4_mask  # homotopy uses reverse (decreasing ΔW) branch
        if not np.any(mask):
            continue
        dw_plot = delta_w_norm[mask]
        dh_plot = np.abs(delta_h_norm[mask])
        if len(dw_plot) == 0:
            continue
        color = homotopy_colors[idx % len(homotopy_colors)]
        linestyle = homotopy_linestyles[idx % len(homotopy_linestyles)]
        h, = ax.plot(dw_plot, dh_plot, color=color, linestyle=linestyle, linewidth=linewidth,
                     label=label, alpha=alpha)
        homotopy_handles.append(h)
        homotopy_labels.append(label)
        homotopy_curve_data.append({'label': label, 'x': np.asarray(dw_plot), 'y': np.asarray(dh_plot), 'color': color, 'linestyle': linestyle})

    # Optionally save one pkl with all curve data + plot config for reproduction
    reproduce_pkl = config.get('reproduce_pkl', False)
    reproduce_path_ref = config.get('reproduce_pkl_path', '')
    if reproduce_pkl and reproduce_path_ref:
        out_reproduce = resolve_pkl_path(reproduce_path_ref, config_dir, '')
        out_reproduce = Path(out_reproduce)
        out_reproduce.parent.mkdir(parents=True, exist_ok=True)
        plot_config = {
            'title': title, 'xlabel': xlabel, 'ylabel': ylabel,
            'xlim': xlim, 'ylim': ylim, 'figsize': figsize,
            'tight_layout_rect': tight_layout_rect,
            'legend_bbox_y': legend_bbox_y, 'homotopy_legend_bbox_y': homotopy_legend_bbox_y,
            'fonts': fonts, 'fontsize_labels': fontsize_labels,
            'fontsize_suptitle': fontsize_suptitle, 'fontsize_legend': fontsize_legend,
            'linewidth': linewidth, 'alpha': alpha,
            'colors': colors, 'linestyles': linestyles,
            'homotopy_colors': homotopy_colors, 'homotopy_linestyles': homotopy_linestyles,
        }
        reproduce_payload = {
            'version': 1,
            'main_curves': main_curve_data,
            'homotopy_curves': homotopy_curve_data,
            'plot_config': plot_config,
        }
        with open(out_reproduce, 'wb') as f:
            pickle.dump(reproduce_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved reproduce pkl: {out_reproduce}")

    ax.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel, fontsize=fontsize_labels)
    ax.tick_params(axis='both', labelsize=fontsize_labels)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=fontsize_suptitle, fontweight='bold', y=0.98)
    if all_handles and all_labels:
        ncol = min(len(all_handles), 6)
        fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, legend_bbox_y),
                   ncol=ncol, fontsize=fontsize_legend, frameon=True, fancybox=True, shadow=True,
                   columnspacing=1.5, handlelength=2.0, handletextpad=0.5)
    if homotopy_handles and homotopy_labels:
        # One line: "Homotopy aided: [handle1] W/L* = 1/2 [handle2] W/L* = 1/3"
        dummy = Patch(facecolor='none', edgecolor='none', label='Homotopy aided: ')
        h_handles = [dummy] + homotopy_handles
        h_labels = ['Homotopy aided: '] + homotopy_labels
        ncol_h = len(h_handles)
        fig.legend(h_handles, h_labels, loc='upper center', bbox_to_anchor=(0.5, homotopy_legend_bbox_y),
                   ncol=ncol_h, fontsize=fontsize_legend, frameon=True, fancybox=True, shadow=True,
                   columnspacing=1.5, handlelength=2.0, handletextpad=0.5)
    plt.tight_layout(rect=tight_layout_rect)
    out_path = plot_dir / out_filename
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Plot |H_mid|/L vs ΔW/L for all W/L in one plot from config.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML. Default: config_hmid_all_wl.yaml')
    parser.add_argument('--plot-dir', type=str, required=True, help='Directory to save the output plot')
    parser.add_argument('--output', type=str, default='hmid_all_wl.png', help='Output filename')
    args = parser.parse_args()
    script_dir = Path(__file__).parent.resolve()
    config_path = Path(args.config) if args.config else script_dir / 'config_hmid_all_wl.yaml'
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    run_plot(config_path, args.plot_dir, out_filename=args.output)


if __name__ == '__main__':
    main()
