#!/usr/bin/env python3
"""
Plot shear homotopy results: 2x2 subplots (Sano, reverse direction).

  Subplot 1 & 2 (W/L=1/2, W/L=1/3): Hmid/L vs Delta W/L with FEA reference.
  Subplot 3 & 4 (W/L=1/2, W/L=1/3): Hmid/L vs Delta W/L with shear force (twin axis).

All use the reverse (phase 4, Delta W decreasing) branch after homotopy.
Pkl schema: same as simulate.py output (qs, geo, config, tracked_forces, ...).
"""

import argparse
import sys
from pathlib import Path

# Allow loading pkls that contain dismech objects
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml


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


def load_shear_homotopy_pkl(pkl_path):
    """
    Load shear homotopy simulation pkl (simulate.py output).
    Returns: qs, sim_params, geom_params, material, env, geo, fixed_nodes,
             tracked_forces, tracked_forces_times, config.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return (
        data['qs'],
        data.get('sim_params'),
        data.get('geom_params'),
        data.get('material'),
        data.get('env'),
        data['geo'],
        data.get('fixed_nodes'),
        data.get('tracked_forces', None),
        data.get('tracked_forces_times', None),
        data.get('config', {}),
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


def find_end_nodes(q0, n_nodes, end_thresh=0.09):
    """Find end nodes (nodes at x >= end_thresh)."""
    node_positions = get_node_positions(q0, n_nodes)
    return np.where(node_positions[:, 0] >= end_thresh)[0]


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
    if len(delta_w) > 0:
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
    Returns (x, y) arrays, or (None, None) on error.
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
    """Remove FEA points that are outliers in the x-band [x_lo, x_hi]. Returns (x_filtered, y_filtered)."""
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


def compute_summed_forces_at_nodes(tracked_forces_list, node_indices, geo):
    """
    Sum reaction forces at the given nodes.
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


def run_plot(config_path, plot_dir, out_filename='shear_homotopy_sano.png'):
    """Build 2x2 subplots: (1,2) Hmid vs dW + FEA; (3,4) Hmid vs dW + shear force. Reverse branch only."""
    config, config_dir = load_config(config_path)
    base_dir_ref = config.get('base_dir', '')
    xlim = config.get('xlim', [0, 0.5])
    ylim = config.get('ylim', [-0.05, 0.35])
    ylim_fy = config.get('ylim_fy')
    title = config.get('title', 'Shear homotopy — Sano')
    figsize = config.get('figsize', [14, 10])
    wspace = config.get('wspace', 0.28)
    hspace = config.get('hspace', 0.35)
    tight_layout_rect = config.get('tight_layout_rect', [0, 0, 1, 0.82])
    subplot_titles = config.get('subplot_titles', ['W/L = 1/2', 'W/L = 1/3', 'W/L = 1/2', 'W/L = 1/3'])
    labels_cfg = config.get('labels', {})
    xlabel = labels_cfg.get('xaxis', r'$\Delta W / L$')
    ylabel = labels_cfg.get('yaxis', r'$|H_{mid}/L|$')
    ylabel_right = labels_cfg.get('yaxis_right', r'$\hat{F}_{shear}$ (FL/(Yh³))')
    force_side = config.get('force_side', 'end').lower()
    if force_side not in ('start', 'end'):
        force_side = 'end'
    force_color = config.get('force_color', '#0066AA')

    # Fonts: CMR10 text, Computer Modern (cm) for math (uses CMMI10-style italic)
    fonts_cfg = config.get('fonts', {})
    font_text = fonts_cfg.get('text', 'CMR10')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [font_text]
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.formatter.use_mathtext'] = True

    # Sano pkls for W/L 1/2 and 1/3
    sano_cfg = config.get('sano', {})
    pkl_1_2 = resolve_pkl_path(sano_cfg.get('pkl_W_L_1_2', ''), config_dir, base_dir_ref)
    pkl_1_3 = resolve_pkl_path(sano_cfg.get('pkl_W_L_1_3', ''), config_dir, base_dir_ref)
    sano_pkls = [pkl_1_2, pkl_1_3]

    fea_config = config.get('fea', {})
    fea_dir_ref = fea_config.get('pkl_dir', '')
    fea_files = fea_config.get('pkl_files', [])

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(figsize[0], figsize[1]), gridspec_kw={'wspace': wspace, 'hspace': hspace})
    axes = axes.flatten()

    # Subplot index: 0,1 = Hmid + FEA; 2,3 = Hmid + Fy
    for subplot_idx in range(4):
        ax = axes[subplot_idx]
        ax.set_title(subplot_titles[subplot_idx] if subplot_idx < len(subplot_titles) else '', fontsize=12, pad=6)
        wl_idx = subplot_idx % 2  # 0 -> W/L 1/2, 1 -> W/L 1/3
        sano_path = sano_pkls[wl_idx]
        if not sano_path.exists():
            print(f"  Skip subplot {subplot_idx}: not found {sano_path}")
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.grid(True, alpha=0.3)
            continue

        try:
            (qs, _, geom_params, material, _, geo, _, tracked_forces_list, _, cfg) = load_shear_homotopy_pkl(sano_path)
        except Exception as e:
            print(f"  Error loading {sano_path}: {e}")
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.grid(True, alpha=0.3)
            continue

        bc_cfg = cfg.get('boundary_condition', {})
        start_thresh = float(bc_cfg.get('start_x_threshold', 0.01))
        end_thresh = float(bc_cfg.get('end_x_threshold', 0.09))

        q0 = qs[0]
        n_nodes = infer_n_nodes(q0, geo)
        middle_node_idx = find_middle_node(q0, n_nodes, start_thresh=start_thresh, end_thresh=end_thresh)
        start_node_indices = find_start_nodes(q0, n_nodes, start_thresh=start_thresh)
        end_node_indices = find_end_nodes(q0, n_nodes, end_thresh=end_thresh)
        delta_h, delta_w, L_val = extract_delta_h_w(qs, n_nodes, middle_node_idx, start_node_indices)
        delta_h_norm = delta_h / L_val
        delta_w_norm = delta_w / L_val
        phase3_mask, phase4_mask, _, _ = split_phases(delta_w)

        # Reverse direction = phase 4 (ΔW decreasing)
        mask = phase4_mask
        if not np.any(mask):
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.grid(True, alpha=0.3)
            continue

        dw_plot = delta_w_norm[mask]
        dh_plot = np.abs(delta_h_norm[mask])
        if len(dw_plot) == 0:
            continue

        # Left axis: Hmid/L vs Delta W/L
        ax.plot(dw_plot, dh_plot, color='#FF0000', linestyle=':', linewidth=2.5, label='Sano', alpha=0.9)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.grid(True, alpha=0.3)

        # FEA overlay on subplots 0 and 1
        if subplot_idx < 2 and fea_config and wl_idx < len(fea_files):
            fea_path = resolve_pkl_path(Path(fea_dir_ref) / fea_files[wl_idx], config_dir, base_dir_ref)
            if fea_path.exists():
                x_fe, y_fe = load_fea_pkl(fea_path)
                if x_fe is not None and len(x_fe) > 0:
                    x_fe, y_fe = remove_fea_outliers_in_band(x_fe, y_fe, x_lo=0.0, x_hi=0.1, iqr_k=2.0)
                    fea_alpha = fea_config.get('alpha', 0.55)
                    fea_color = fea_config.get('color', '#2F4F4F')
                    fea_marker = fea_config.get('marker', '^')
                    fea_s = fea_config.get('s', 28)
                    fea_label = fea_config.get('label', 'FEA')
                    ax.scatter(
                        x_fe, y_fe,
                        c=fea_color, marker=fea_marker, s=fea_s, alpha=fea_alpha,
                        label=fea_label, zorder=5, edgecolors='none'
                    )

        # Right axis: shear force on subplots 2 and 3
        if subplot_idx >= 2 and tracked_forces_list is not None and len(tracked_forces_list) == len(delta_w):
            node_indices = start_node_indices if force_side == 'start' else end_node_indices
            forces_summed = compute_summed_forces_at_nodes(tracked_forces_list, node_indices, geo)
            h = float(geom_params.rod_r0) if hasattr(geom_params, 'rod_r0') else 0.001
            Y = float(material.youngs_rod) if hasattr(material, 'youngs_rod') else 1e10
            force_norm_factor = L_val / (Y * h**3)
            forces_normalized = forces_summed * force_norm_factor
            shear_plot = np.abs(forces_normalized[mask, 1])

            if len(shear_plot) == len(dw_plot):
                ax2 = ax.twinx()
                ax2.plot(dw_plot, shear_plot, color=force_color, linestyle='--', linewidth=1.5, label=ylabel_right, alpha=0.8)
                ax2.set_ylabel(ylabel_right, fontsize=11, color=force_color)
                ax2.tick_params(axis='y', labelcolor=force_color)
                if ylim_fy is not None and len(ylim_fy) == 2:
                    ax2.set_ylim(ylim_fy[0], ylim_fy[1])
                ax2.grid(True, alpha=0.2, linestyle=':')
        elif subplot_idx >= 2 and (tracked_forces_list is None or len(tracked_forces_list) == 0):
            print(f"  No tracked forces in {sano_path.name} for subplot {subplot_idx}")

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
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
        description='Plot shear homotopy 2x2: Hmid+FEA (row 1), Hmid+Fy (row 2), Sano reverse branch.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML. Default: shear_induced_homotopy/config_shear_homotopy_plot.yaml'
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
        default='shear_homotopy_sano.png',
        help='Output filename'
    )
    args = parser.parse_args()
    script_dir = Path(__file__).parent.resolve()
    config_path = Path(args.config) if args.config else script_dir / 'config_shear_homotopy_plot.yaml'
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    run_plot(config_path, args.plot_dir, out_filename=args.output)


if __name__ == '__main__':
    main()
