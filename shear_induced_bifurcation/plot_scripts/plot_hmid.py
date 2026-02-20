#!/usr/bin/env python3
"""
Plot |H_mid|/L vs ΔW/L for multiple energy models from config-driven pkl paths.

Uses config_hmid_plot.yaml to define, per energy model, 4 pkl files (one per W/L subplot)
and whether to plot the ΔW increasing or decreasing branch for each.
2x2 subplot layout with legend at top center (side-by-side).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml

# Energy model styles (same as clamped_clamped/simulate_all.py)
ENERGY_MODELS = {
    'kirchhoff': {'label': 'Kirchhoff', 'color': '#8B00FF', 'linestyle': '-.', 'linewidth': 2.5},
    'sadowsky': {'label': 'Sadowsky', 'color': '#0066FF', 'linestyle': '--', 'linewidth': 2.5},
    'wunderlich': {'label': 'Wunderlich', 'color': '#D2691E', 'linestyle': (0, (5, 2)), 'linewidth': 2.5},
    'sano': {'label': 'Sano', 'color': '#FF0000', 'linestyle': ':', 'linewidth': 2.5},
    'audoly': {'label': 'Audoly', 'color': '#00AA44', 'linestyle': (0, (3, 1, 1, 1)), 'linewidth': 2.0},
}


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


def load_fea_pkl(pkl_path):
    """
    Load FEA sweep curve pkl. Schema: delta_w_over_L, H_mid_over_L_abs, label, n_points.
    Returns (x, y) arrays for scattering, or (None, None) on error.
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        x = np.asarray(data['delta_w_over_L'])
        y = np.asarray(data['H_mid_over_L_abs'])
        return x, y
    except Exception as e:
        print(f"  FEA load error {pkl_path}: {e}")
        return None, None


def remove_fea_outliers_in_band(x, y, x_lo=0.0, x_hi=0.1, iqr_k=2.0):
    """
    Remove FEA points that are outliers in the x-band [x_lo, x_hi].
    In that band, points with y outside [Q1 - iqr_k*IQR, Q3 + iqr_k*IQR] are dropped.
    Points outside the band are kept. Returns (x_filtered, y_filtered).
    """
    x, y = np.asarray(x), np.asarray(y)
    in_band = (x >= x_lo) & (x <= x_hi)
    out_band = ~in_band
    if not np.any(in_band):
        return x, y
    y_band = y[in_band]
    q1, q3 = np.percentile(y_band, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        iqr = np.std(y_band) or 1e-10
    low, high = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    in_band_keep = in_band & (y >= low) & (y <= high)
    keep = out_band | in_band_keep
    return x[keep], y[keep]


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


def run_plot(config_path, plot_dir, out_filename='hmid_comparison.png'):
    """Build 2x2 subplots of |H_mid|/L vs ΔW/L for all energies from config."""
    config, config_dir = load_config(config_path)
    base_dir_ref = config.get('base_dir', '')
    xlim = config.get('xlim', [0, 0.5])
    ylim = config.get('ylim', [-0.05, 0.35])
    title = config.get('title', 'Energy Model Comparison')
    figsize = config.get('figsize', [14, 10])
    wspace = config.get('wspace', 0.25)
    hspace = config.get('hspace', 0.35)
    tight_layout_rect = config.get('tight_layout_rect', [0, 0, 1, 0.85])
    subplot_titles = config.get('subplot_titles', ['W/L = 1/2', 'W/L = 1/6', 'W/L = 1/12', 'W/L = 1/20'])
    fea_config = config.get('fea')  # optional: pkl_dir, pkl_files (list of 4), alpha, color, marker, s, label
    energies_config = config.get('energies', {})
    if not energies_config:
        raise ValueError("Config must have 'energies' with at least one model.")

    n_subplots = 4
    energy_models = list(energies_config.keys())
    for key in energy_models:
        if key not in ENERGY_MODELS:
            raise ValueError(f"Unknown energy model in config: {key}. Allowed: {list(ENERGY_MODELS.keys())}")

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(figsize[0], figsize[1]), gridspec_kw={'wspace': wspace, 'hspace': hspace})
    axes = axes.flatten()

    legend_handles = {}
    legend_labels = {}
    subplot_data = []

    for subplot_idx in range(n_subplots):
        ax = axes[subplot_idx]
        if subplot_idx < len(subplot_titles):
            ax.set_title(subplot_titles[subplot_idx], fontsize=12, pad=6)
        ax._reproduce_curves = []

        for energy_model in energy_models:
            entries = energies_config[energy_model]
            if subplot_idx >= len(entries):
                continue
            entry = entries[subplot_idx]
            pkl_ref = entry.get('pkl')
            shear_increasing = entry.get('shear_increasing', True)
            if not pkl_ref:
                continue
            pkl_path = resolve_pkl_path(pkl_ref, config_dir, base_dir_ref)
            if not pkl_path.exists():
                print(f"  Skip {energy_model} subplot {subplot_idx}: not found {pkl_path}")
                continue
            try:
                (qs, _, _, _, _, geo, _, _, _) = load_pkl_data(pkl_path)
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

            if shear_increasing:
                mask = phase3_mask
            else:
                mask = phase4_mask
            if not np.any(mask):
                continue
            dw_plot = delta_w_norm[mask]
            dh_plot = delta_h_norm[mask]
            if len(dw_plot) == 0:
                continue
            style = ENERGY_MODELS[energy_model]
            line = ax.plot(
                dw_plot, np.abs(dh_plot),
                color=style['color'], linestyle=style['linestyle'],
                linewidth=style.get('linewidth', 2.5), label=style['label'], alpha=0.9
            )
            if energy_model not in legend_handles:
                legend_handles[energy_model] = line[0]
                legend_labels[energy_model] = style['label']
            ax._reproduce_curves.append({
                'label': style['label'], 'x': np.asarray(dw_plot), 'y': np.asarray(np.abs(dh_plot)),
                'color': style['color'], 'linestyle': style['linestyle']
            })
        fea_xy = None
        # Optional FEA scatter overlay (translucent triangles, distinct color)
        if fea_config and subplot_idx < len(fea_config.get('pkl_files', [])):
            fea_dir_ref = fea_config.get('pkl_dir', '')
            fea_files = fea_config.get('pkl_files', [])
            fea_name = fea_files[subplot_idx]
            fea_path = resolve_pkl_path(Path(fea_dir_ref) / fea_name, config_dir, base_dir_ref)
            if fea_path.exists():
                x_fe, y_fe = load_fea_pkl(fea_path)
                if x_fe is not None and len(x_fe) > 0:
                    x_fe, y_fe = remove_fea_outliers_in_band(x_fe, y_fe, x_lo=0.0, x_hi=0.1, iqr_k=2.0)
                    fea_alpha = fea_config.get('alpha', 0.55)
                    fea_color = fea_config.get('color', '#2F4F4F')
                    fea_marker = fea_config.get('marker', '^')
                    fea_s = fea_config.get('s', 28)
                    fea_label = fea_config.get('label', 'FEA')
                    sc = ax.scatter(
                        x_fe, y_fe,
                        c=fea_color, marker=fea_marker, s=fea_s, alpha=fea_alpha,
                        label=fea_label, zorder=5, edgecolors='none'
                    )
                    if '_fea' not in legend_handles:
                        legend_handles['_fea'] = sc
                        legend_labels['_fea'] = fea_label
                    fea_xy = {'x': np.asarray(x_fe), 'y': np.asarray(y_fe)}
        curves_this = getattr(ax, '_reproduce_curves', [])
        subplot_data.append({'curves': curves_this, 'fea': fea_xy if fea_xy is not None else None})
        ax.set_xlabel('ΔW/L', fontsize=12)
        ax.set_ylabel(r'$|H_{mid}/L|$', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlim(xlim[0], xlim[1])

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=tight_layout_rect)
    if legend_handles and legend_labels:
        handles = [legend_handles[m] for m in energy_models if m in legend_handles]
        labels = [legend_labels[m] for m in energy_models if m in legend_labels]
        if '_fea' in legend_handles:
            handles.append(legend_handles['_fea'])
            labels.append(legend_labels['_fea'])
        if handles and labels:
            ncol_legend = min(len(handles), 6)
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96),
                       ncol=ncol_legend, fontsize=11, frameon=True, fancybox=True, shadow=True,
                       columnspacing=1.5, handlelength=2.0, handletextpad=0.5)
    reproduce_pkl = config.get('reproduce_pkl', False)
    reproduce_path_ref = config.get('reproduce_pkl_path', '')
    if reproduce_pkl and reproduce_path_ref:
        out_reproduce = resolve_pkl_path(reproduce_path_ref, config_dir, '')
        out_reproduce = Path(out_reproduce)
        out_reproduce.parent.mkdir(parents=True, exist_ok=True)
        plot_config = {
            'title': title, 'xlim': xlim, 'ylim': ylim, 'figsize': figsize,
            'wspace': wspace, 'hspace': hspace, 'tight_layout_rect': tight_layout_rect,
            'subplot_titles': subplot_titles,
        }
        payload = {'version': 1, 'subplots': subplot_data, 'plot_config': plot_config}
        with open(out_reproduce, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved reproduce pkl: {out_reproduce}")
    out_path = plot_dir / out_filename
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Plot |H_mid|/L vs ΔW/L for multiple energy models from config (2x2 subplots).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config_hmid_plot.yaml. Default: validate_discretization/plot_scripts/config_hmid_plot.yaml'
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
        default='hmid_comparison.png',
        help='Output filename (e.g. hmid_comparison.png)'
    )
    args = parser.parse_args()
    script_dir = Path(__file__).parent.resolve()
    config_path = Path(args.config) if args.config else script_dir / 'config_hmid_plot.yaml'
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    run_plot(config_path, args.plot_dir, out_filename=args.output)


if __name__ == '__main__':
    main()
