import typing
import numpy as np
import scipy.sparse as sp
import torch
import dataclasses

from .elastic_energy import ElasticEnergy
from ..state import RobotState
from .elastic_energy_nn import ElasticEnergyRod3DNN # Import the NN model class
# from .analytical_sanos_elastic_energy import AnalyticalSanosElasticEnergy # Import the analytical model class
from .analytical_wunderlichs_elastic_energy import AnalyticalWunderlichElasticEnergy
from ..springs import StretchBendTwistSpring # Import the new combined spring

# --- Strain Calculation Helpers (Adapted from Notebook) ---
# Note: These functions are adapted to work with batches of springs (N springs)
#       and assume inputs (node positions, material directors) are shaped (N, 3) or (N,).

def _get_strain_stretch3D_batch(node0, node1, node2, l_eff):
    """Computes stretch strain for a batch of nodes."""
    inv_l_eff = 1.0 / l_eff
    # Calculate stretch for the first edge (node0 to node1)
    edge0 = node1 - node0
    edge_len0 = np.linalg.norm(edge0, axis=1)
    stretch0 = edge_len0 * inv_l_eff - 1.0

    # Calculate stretch for the second edge (node1 to node2)
    edge1 = node2 - node1
    edge_len1 = np.linalg.norm(edge1, axis=1)
    stretch1 = edge_len1 * inv_l_eff - 1.0

    # Node strain is the average of the two edge stretches
    epsilon_node = 0.5 * stretch0 + 0.5 * stretch1
    return epsilon_node # Shape (N,)

def _grad_hess_strain_stretch3D_batch(node0, node1, node2, l_eff):
    """Computes gradient and Hessian of stretch strain for a batch."""
    N = node0.shape[0]
    Id3 = np.eye(3)
    inv_l_eff = 1.0 / l_eff

    grad_eps = np.zeros((N, 9))
    hess_eps = np.zeros((N, 9, 9))

    # --- Edge 0-1 ---
    edge0 = node1 - node0
    edge_len0 = np.linalg.norm(edge0, axis=1)
    valid_len0 = edge_len0 > 1e-12
    tangent0 = np.zeros_like(edge0)
    tangent0[valid_len0] = edge0[valid_len0] / edge_len0[valid_len0, None]
    eps0 = edge_len0 * inv_l_eff - 1.0
    dF_unit0 = tangent0 * inv_l_eff[:, None]

    M0_term1 = np.zeros((N, 3, 3))
    M0_term2 = np.zeros((N, 3, 3))
    if np.any(valid_len0):
         M0_term1[valid_len0] = (inv_l_eff[valid_len0, None, None] - 1.0 / edge_len0[valid_len0, None, None]) * Id3
         M0_term2[valid_len0] = np.einsum('...i,...j->...ij', edge0[valid_len0], edge0[valid_len0]) / (edge_len0[valid_len0]**3)[:, None, None]
    M0 = 2.0 * inv_l_eff[:, None, None] * (M0_term1 + M0_term2)
    M2_0 = M0 - 2.0 * np.einsum('...i,...j->...ij', dF_unit0, dF_unit0)
    M2_0 *= 0.5
    mask0 = eps0 != 0
    M2_0 = np.divide(M2_0, eps0[:, None, None], where=mask0[:, None, None], out=np.zeros_like(M2_0))

    # --- Edge 1-2 ---
    edge1 = node2 - node1
    edge_len1 = np.linalg.norm(edge1, axis=1)
    valid_len1 = edge_len1 > 1e-12
    tangent1 = np.zeros_like(edge1)
    tangent1[valid_len1] = edge1[valid_len1] / edge_len1[valid_len1, None]
    eps1 = edge_len1 * inv_l_eff - 1.0
    dF_unit1 = tangent1 * inv_l_eff[:, None]

    M1_term1 = np.zeros((N, 3, 3))
    M1_term2 = np.zeros((N, 3, 3))
    if np.any(valid_len1):
        M1_term1[valid_len1] = (inv_l_eff[valid_len1, None, None] - 1.0 / edge_len1[valid_len1, None, None]) * Id3
        M1_term2[valid_len1] = np.einsum('...i,...j->...ij', edge1[valid_len1], edge1[valid_len1]) / (edge_len1[valid_len1]**3)[:, None, None]
    M1 = 2.0 * inv_l_eff[:, None, None] * (M1_term1 + M1_term2)
    M2_1 = M1 - 2.0 * np.einsum('...i,...j->...ij', dF_unit1, dF_unit1)
    M2_1 *= 0.5
    mask1 = eps1 != 0
    M2_1 = np.divide(M2_1, eps1[:, None, None], where=mask1[:, None, None], out=np.zeros_like(M2_1))

    # --- Combine ---
    grad_eps[:, 0:3] = -0.5 * dF_unit0
    grad_eps[:, 3:6] = 0.5 * dF_unit0 - 0.5 * dF_unit1
    grad_eps[:, 6:9] = 0.5 * dF_unit1
    hess_eps[:, 0:3, 0:3] = 0.5 * M2_0
    hess_eps[:, 3:6, 3:6] = 0.5 * M2_0 + 0.5 * M2_1
    hess_eps[:, 6:9, 6:9] = 0.5 * M2_1
    hess_eps[:, 0:3, 3:6] = -0.5 * M2_0
    hess_eps[:, 3:6, 0:3] = -0.5 * M2_0
    hess_eps[:, 3:6, 6:9] = -0.5 * M2_1
    hess_eps[:, 6:9, 3:6] = -0.5 * M2_1

    return grad_eps, hess_eps # Shapes (N, 9), (N, 9, 9)

def _get_strain_curvature3D_batch(node0, node1, node2, m1e, m2e, m1f, m2f):
    """Computes curvature strain for a batch."""
    ee = node1 - node0
    ef = node2 - node1
    norm_e = np.linalg.norm(ee, axis=1)
    norm_f = np.linalg.norm(ef, axis=1)

    # Handle potential zero length edges
    valid_edges = (norm_e > 1e-12) & (norm_f > 1e-12)
    kappa = np.zeros((node0.shape[0], 2))
    if not np.any(valid_edges):
        return kappa

    te = np.zeros_like(ee)
    tf = np.zeros_like(ef)
    te[valid_edges] = ee[valid_edges] / norm_e[valid_edges, None]
    tf[valid_edges] = ef[valid_edges] / norm_f[valid_edges, None]

    dot_prod = np.sum(te * tf, axis=1)
    # Avoid division by zero if t0 = -t1
    chi = 1.0 + dot_prod
    valid_chi = np.abs(chi) > 1e-12
    valid_indices = valid_edges & valid_chi

    if not np.any(valid_indices):
        return kappa

    chi_inv = np.zeros_like(chi)
    chi_inv[valid_indices] = 1.0 / chi[valid_indices]
    kb = np.zeros_like(te)
    kb[valid_indices] = 2.0 * np.cross(te[valid_indices], tf[valid_indices]) * chi_inv[valid_indices, None]

    kappa[valid_indices, 0] = 0.5 * np.sum(kb[valid_indices] * (m2e[valid_indices] + m2f[valid_indices]), axis=1)
    kappa[valid_indices, 1] = -0.5 * np.sum(kb[valid_indices] * (m1e[valid_indices] + m1f[valid_indices]), axis=1)

    return kappa # Shape (N, 2)

def _grad_hess_strain_curvature3D_batch(node0, node1, node2, m1e, m2e, m1f, m2f):
    """Computes gradient and Hessian of curvature strain for a batch."""
    n_springs = node0.shape[0]
    Id3 = np.eye(3)[None, :, :]  # For broadcasting

    gradKappa = np.zeros((n_springs, 11, 2))
    hessKappa = np.zeros((n_springs, 11, 11, 2))

    # Precompute common terms
    ee = node1 - node0
    ef = node2 - node1
    norm_e = np.linalg.norm(ee, axis=1)
    norm_f = np.linalg.norm(ef, axis=1)

    valid_edges = (norm_e > 1e-12) & (norm_f > 1e-12)
    if not np.any(valid_edges):
        return gradKappa, hessKappa # Return zeros if edges are invalid

    te = np.zeros_like(ee)
    tf = np.zeros_like(ef)
    te[valid_edges] = ee[valid_edges] / norm_e[valid_edges, None]
    tf[valid_edges] = ef[valid_edges] / norm_f[valid_edges, None]

    dot_prod = np.sum(te * tf, axis=1)
    chi = 1.0 + dot_prod
    valid_chi = np.abs(chi) > 1e-12
    valid_indices = valid_edges & valid_chi

    if not np.any(valid_indices):
         return gradKappa, hessKappa # Return zeros

    # --- Calculate only for valid indices ---
    vi = valid_indices # Shorthand
    n0p_v, n1p_v, n2p_v = node0[vi], node1[vi], node2[vi]
    m1e_v, m2e_v, m1f_v, m2f_v = m1e[vi], m2e[vi], m1f[vi], m2f[vi]
    norm_e_v, norm_f_v = norm_e[vi], norm_f[vi]
    te_v, tf_v = te[vi], tf[vi]
    chi_v = chi[vi]
    chi_inv_v = 1.0 / chi_v
    kb_v = 2.0 * np.cross(te_v, tf_v) * chi_inv_v[:, None]

    tilde_t_v = (te_v + tf_v) * chi_inv_v[:, None]
    tilde_d1_v = (m1e_v + m1f_v) * chi_inv_v[:, None]
    tilde_d2_v = (m2e_v + m2f_v) * chi_inv_v[:, None]

    kappa1_v = 0.5 * np.sum(kb_v * (m2e_v + m2f_v), axis=1)
    kappa2_v = -0.5 * np.sum(kb_v * (m1e_v + m1f_v), axis=1)

    # First derivatives
    Dkappa1De_v = (1.0 / norm_e_v[:, None]) * (-kappa1_v[:, None] * tilde_t_v + np.cross(tf_v, tilde_d2_v))
    Dkappa1Df_v = (1.0 / norm_f_v[:, None]) * (-kappa1_v[:, None] * tilde_t_v - np.cross(te_v, tilde_d2_v))
    Dkappa2De_v = (1.0 / norm_e_v[:, None]) * (-kappa2_v[:, None] * tilde_t_v - np.cross(tf_v, tilde_d1_v))
    Dkappa2Df_v = (1.0 / norm_f_v[:, None]) * (-kappa2_v[:, None] * tilde_t_v + np.cross(te_v, tilde_d1_v))

    # Gradient assembly (Indices: 0-2: n0_pos, 3-5: n1_pos, 6-8: n2_pos, 9: th_e, 10: th_f)
    gradKappa[vi, 0:3, 0] = -Dkappa1De_v
    gradKappa[vi, 3:6, 0] = Dkappa1De_v - Dkappa1Df_v
    gradKappa[vi, 6:9, 0] = Dkappa1Df_v
    gradKappa[vi, 0:3, 1] = -Dkappa2De_v
    gradKappa[vi, 3:6, 1] = Dkappa2De_v - Dkappa2Df_v
    gradKappa[vi, 6:9, 1] = Dkappa2Df_v

    # Twist terms (DOFs 9 and 10 correspond to theta1 and theta2)
    gradKappa[vi, 9, 0] = -0.5 * np.sum(kb_v * m1e_v, axis=1) # dKappa1/dTheta_e
    gradKappa[vi, 10, 0] = -0.5 * np.sum(kb_v * m1f_v, axis=1) # dKappa1/dTheta_f
    gradKappa[vi, 9, 1] = -0.5 * np.sum(kb_v * m2e_v, axis=1) # dKappa2/dTheta_e
    gradKappa[vi, 10, 1] = -0.5 * np.sum(kb_v * m2f_v, axis=1) # dKappa2/dTheta_f

    # --- Hessian Calculation (Simplified - using notebook logic structure) ---
    norm2_e_v = norm_e_v**2
    norm2_f_v = norm_f_v**2
    Id3_v = np.eye(3)[None, :, :].repeat(n0p_v.shape[0], axis=0)

    def batch_outer(a, b): return np.einsum('...i,...j->...ij', a, b)
    def cross_mat_batch(v):
        zeros = np.zeros_like(v[:, 0])
        return np.array([[zeros, -v[:, 2], v[:, 1]], [v[:, 2], zeros, -v[:, 0]], [-v[:, 1], v[:, 0], zeros]]).transpose(2, 0, 1)

    # Kappa1 second derivatives
    tt_o_tt_v = batch_outer(tilde_t_v, tilde_t_v)
    tf_c_d2t_v = np.cross(tf_v, tilde_d2_v)
    tf_c_d2t_o_tt_v = batch_outer(tf_c_d2t_v, tilde_t_v)
    tt_o_tf_c_d2t_v = batch_outer(tilde_t_v, tf_c_d2t_v) # Transpose of above
    kb_o_d2e_v = batch_outer(kb_v, m2e_v)

    D2kappa1De2_v = (1/norm2_e_v[:,None,None])*(2*kappa1_v[:,None,None]*tt_o_tt_v - tf_c_d2t_o_tt_v - tt_o_tf_c_d2t_v) \
        - (kappa1_v[:,None,None]/(chi_v[:,None,None]*norm2_e_v[:,None,None]))*(Id3_v - batch_outer(te_v, te_v)) \
        + (1/(2*norm2_e_v[:,None,None]))*kb_o_d2e_v

    te_c_d2t_v = np.cross(te_v, tilde_d2_v)
    te_c_d2t_o_tt_v = batch_outer(te_c_d2t_v, tilde_t_v)
    tt_o_te_c_d2t_v = batch_outer(tilde_t_v, te_c_d2t_v) # Transpose of above
    kb_o_d2f_v = batch_outer(kb_v, m2f_v)

    D2kappa1Df2_v = (1/norm2_f_v[:,None,None])*(2*kappa1_v[:,None,None]*tt_o_tt_v + te_c_d2t_o_tt_v + tt_o_te_c_d2t_v) \
        - (kappa1_v[:,None,None]/(chi_v[:,None,None]*norm2_f_v[:,None,None]))*(Id3_v - batch_outer(tf_v, tf_v)) \
        + (1/(2*norm2_f_v[:,None,None]))*kb_o_d2f_v

    te_o_tf_v = batch_outer(te_v, tf_v)
    D2kappa1DeDf_v = (-kappa1_v[:,None,None]/(chi_v[:,None,None]*norm_e_v[:,None,None]*norm_f_v[:,None,None]))*(Id3_v + te_o_tf_v) \
        + (1/(norm_e_v[:,None,None]*norm_f_v[:,None,None]))*(2*kappa1_v[:,None,None]*tt_o_tt_v
                                                             - tf_c_d2t_o_tt_v + tt_o_te_c_d2t_v - cross_mat_batch(tilde_d2_v))

    # Kappa2 second derivatives
    tf_c_d1t_v = np.cross(tf_v, tilde_d1_v)
    tf_c_d1t_o_tt_v = batch_outer(tf_c_d1t_v, tilde_t_v)
    tt_o_tf_c_d1t_v = batch_outer(tilde_t_v, tf_c_d1t_v) # Transpose
    kb_o_d1e_v = batch_outer(kb_v, m1e_v)

    D2kappa2De2_v = (1/norm2_e_v[:,None,None])*(2*kappa2_v[:,None,None]*tt_o_tt_v + tf_c_d1t_o_tt_v + tt_o_tf_c_d1t_v) \
        - (kappa2_v[:,None,None]/(chi_v[:,None,None]*norm2_e_v[:,None,None]))*(Id3_v - batch_outer(te_v, te_v)) \
        - (1/(2*norm2_e_v[:,None,None]))*kb_o_d1e_v

    te_c_d1t_v = np.cross(te_v, tilde_d1_v)
    te_c_d1t_o_tt_v = batch_outer(te_c_d1t_v, tilde_t_v)
    tt_o_te_c_d1t_v = batch_outer(tilde_t_v, te_c_d1t_v) # Transpose
    kb_o_d1f_v = batch_outer(kb_v, m1f_v)

    D2kappa2Df2_v = (1/norm2_f_v[:,None,None])*(2*kappa2_v[:,None,None]*tt_o_tt_v - te_c_d1t_o_tt_v - tt_o_te_c_d1t_v) \
        - (kappa2_v[:,None,None]/(chi_v[:,None,None]*norm2_f_v[:,None,None]))*(Id3_v - batch_outer(tf_v, tf_v)) \
        - (1/(2*norm2_f_v[:,None,None]))*kb_o_d1f_v

    D2kappa2DeDf_v = (-kappa2_v[:,None,None]/(chi_v[:,None,None]*norm_e_v[:,None,None]*norm_f_v[:,None,None]))*(Id3_v + te_o_tf_v) \
        + (1/(norm_e_v[:,None,None]*norm_f_v[:,None,None]))*(2*kappa2_v[:,None,None]*tt_o_tt_v
                                                             + tf_c_d1t_o_tt_v - tt_o_te_c_d1t_v + cross_mat_batch(tilde_d1_v))

    # Twist terms
    D2kappa1Dthetae2_v = -0.5 * np.sum(kb_v * m2e_v, axis=1)
    D2kappa1Dthetaf2_v = -0.5 * np.sum(kb_v * m2f_v, axis=1)
    D2kappa2Dthetae2_v = 0.5 * np.sum(kb_v * m1e_v, axis=1)
    D2kappa2Dthetaf2_v = 0.5 * np.sum(kb_v * m1f_v, axis=1)

    # Coupled terms (Position-Theta) - Simplified structure
    D2kappa1DeDthetae_v = (1/norm_e_v[:,None]) * (0.5 * np.sum(kb_v * m1e_v, axis=1)[:,None] * tilde_t_v - (1/chi_v[:,None]) * np.cross(tf_v, m1e_v))
    D2kappa1DeDthetaf_v = (1/norm_e_v[:,None]) * (0.5 * np.sum(kb_v * m1f_v, axis=1)[:,None] * tilde_t_v - (1/chi_v[:,None]) * np.cross(tf_v, m1f_v))
    D2kappa1DfDthetae_v = (1/norm_f_v[:,None]) * (0.5 * np.sum(kb_v * m1e_v, axis=1)[:,None] * tilde_t_v + (1/chi_v[:,None]) * np.cross(te_v, m1e_v))
    D2kappa1DfDthetaf_v = (1/norm_f_v[:,None]) * (0.5 * np.sum(kb_v * m1f_v, axis=1)[:,None] * tilde_t_v + (1/chi_v[:,None]) * np.cross(te_v, m1f_v))

    D2kappa2DeDthetae_v = (1/norm_e_v[:,None]) * (0.5 * np.sum(kb_v * m2e_v, axis=1)[:,None] * tilde_t_v - (1/chi_v[:,None]) * np.cross(tf_v, m2e_v))
    D2kappa2DeDthetaf_v = (1/norm_e_v[:,None]) * (0.5 * np.sum(kb_v * m2f_v, axis=1)[:,None] * tilde_t_v - (1/chi_v[:,None]) * np.cross(tf_v, m2f_v))
    D2kappa2DfDthetae_v = (1/norm_f_v[:,None]) * (0.5 * np.sum(kb_v * m2e_v, axis=1)[:,None] * tilde_t_v + (1/chi_v[:,None]) * np.cross(te_v, m2e_v))
    D2kappa2DfDthetaf_v = (1/norm_f_v[:,None]) * (0.5 * np.sum(kb_v * m2f_v, axis=1)[:,None] * tilde_t_v + (1/chi_v[:,None]) * np.cross(te_v, m2f_v))


    # Assemble Hessian blocks for valid indices
    # Position blocks (0:3, 3:6, 6:9)
    hessKappa[vi, 0:3, 0:3, 0] = D2kappa1De2_v
    hessKappa[vi, 0:3, 3:6, 0] = -D2kappa1De2_v + D2kappa1DeDf_v
    hessKappa[vi, 0:3, 6:9, 0] = -D2kappa1DeDf_v
    hessKappa[vi, 3:6, 0:3, 0] = -D2kappa1De2_v + D2kappa1DeDf_v.transpose(0, 2, 1)
    hessKappa[vi, 3:6, 3:6, 0] = D2kappa1De2_v - D2kappa1DeDf_v - D2kappa1DeDf_v.transpose(0, 2, 1) + D2kappa1Df2_v
    hessKappa[vi, 3:6, 6:9, 0] = D2kappa1DeDf_v - D2kappa1Df2_v
    hessKappa[vi, 6:9, 0:3, 0] = -D2kappa1DeDf_v.transpose(0, 2, 1)
    hessKappa[vi, 6:9, 3:6, 0] = D2kappa1DeDf_v.transpose(0, 2, 1) - D2kappa1Df2_v
    hessKappa[vi, 6:9, 6:9, 0] = D2kappa1Df2_v

    hessKappa[vi, 0:3, 0:3, 1] = D2kappa2De2_v
    hessKappa[vi, 0:3, 3:6, 1] = -D2kappa2De2_v + D2kappa2DeDf_v
    hessKappa[vi, 0:3, 6:9, 1] = -D2kappa2DeDf_v
    hessKappa[vi, 3:6, 0:3, 1] = -D2kappa2De2_v + D2kappa2DeDf_v.transpose(0, 2, 1)
    hessKappa[vi, 3:6, 3:6, 1] = D2kappa2De2_v - D2kappa2DeDf_v - D2kappa2DeDf_v.transpose(0, 2, 1) + D2kappa2Df2_v
    hessKappa[vi, 3:6, 6:9, 1] = D2kappa2DeDf_v - D2kappa2Df2_v
    hessKappa[vi, 6:9, 0:3, 1] = -D2kappa2DeDf_v.transpose(0, 2, 1)
    hessKappa[vi, 6:9, 3:6, 1] = D2kappa2DeDf_v.transpose(0, 2, 1) - D2kappa2Df2_v
    hessKappa[vi, 6:9, 6:9, 1] = D2kappa2Df2_v

    # Twist-Twist blocks (9, 10)
    hessKappa[vi, 9, 9, 0] = D2kappa1Dthetae2_v
    hessKappa[vi, 10, 10, 0] = D2kappa1Dthetaf2_v
    hessKappa[vi, 9, 9, 1] = D2kappa2Dthetae2_v
    hessKappa[vi, 10, 10, 1] = D2kappa2Dthetaf2_v

    # Position-Twist blocks (Indices 9, 10 for thetas)
    hessKappa[vi, 0:3, 9, 0] = -D2kappa1DeDthetae_v
    hessKappa[vi, 3:6, 9, 0] = D2kappa1DeDthetae_v - D2kappa1DfDthetae_v
    hessKappa[vi, 6:9, 9, 0] = D2kappa1DfDthetae_v
    hessKappa[vi, 9, 0:3, 0] = -D2kappa1DeDthetae_v # Transpose implicitly handled by structure
    hessKappa[vi, 9, 3:6, 0] = D2kappa1DeDthetae_v - D2kappa1DfDthetae_v
    hessKappa[vi, 9, 6:9, 0] = D2kappa1DfDthetae_v

    hessKappa[vi, 0:3, 10, 0] = -D2kappa1DeDthetaf_v
    hessKappa[vi, 3:6, 10, 0] = D2kappa1DeDthetaf_v - D2kappa1DfDthetaf_v
    hessKappa[vi, 6:9, 10, 0] = D2kappa1DfDthetaf_v
    hessKappa[vi, 10, 0:3, 0] = -D2kappa1DeDthetaf_v
    hessKappa[vi, 10, 3:6, 0] = D2kappa1DeDthetaf_v - D2kappa1DfDthetaf_v
    hessKappa[vi, 10, 6:9, 0] = D2kappa1DfDthetaf_v

    hessKappa[vi, 0:3, 9, 1] = -D2kappa2DeDthetae_v
    hessKappa[vi, 3:6, 9, 1] = D2kappa2DeDthetae_v - D2kappa2DfDthetae_v
    hessKappa[vi, 6:9, 9, 1] = D2kappa2DfDthetae_v
    hessKappa[vi, 9, 0:3, 1] = -D2kappa2DeDthetae_v
    hessKappa[vi, 9, 3:6, 1] = D2kappa2DeDthetae_v - D2kappa2DfDthetae_v
    hessKappa[vi, 9, 6:9, 1] = D2kappa2DfDthetae_v

    hessKappa[vi, 0:3, 10, 1] = -D2kappa2DeDthetaf_v
    hessKappa[vi, 3:6, 10, 1] = D2kappa2DeDthetaf_v - D2kappa2DfDthetaf_v
    hessKappa[vi, 6:9, 10, 1] = D2kappa2DfDthetaf_v
    hessKappa[vi, 10, 0:3, 1] = -D2kappa2DeDthetaf_v
    hessKappa[vi, 10, 3:6, 1] = D2kappa2DeDthetaf_v - D2kappa2DfDthetaf_v
    hessKappa[vi, 10, 6:9, 1] = D2kappa2DfDthetaf_v


    return gradKappa, hessKappa # Shapes (N, 11, 2), (N, 11, 11, 2)


def _parallel_transport_batch(u, t1, t2):
    """Parallel transport a batch of vectors u from tangent t1 to t2."""
    N = u.shape[0]
    d = u.copy() # Default to no change if tangents are parallel
    b = np.cross(t1, t2)
    norm_b = np.linalg.norm(b, axis=1)
    valid = norm_b > 1e-10
    if not np.any(valid):
        return d

    b_v = b[valid] / norm_b[valid, None]
    t1_v = t1[valid]
    t2_v = t2[valid]
    u_v = u[valid]

    # Improve numerical stability (optional but good practice)
    b_v = b_v - np.sum(b_v * t1_v, axis=1)[:, None] * t1_v
    b_v /= np.linalg.norm(b_v, axis=1)[:, None]
    b_v = b_v - np.sum(b_v * t2_v, axis=1)[:, None] * t2_v
    b_v /= np.linalg.norm(b_v, axis=1)[:, None]

    n1_v = np.cross(t1_v, b_v)
    n2_v = np.cross(t2_v, b_v)

    d[valid] = np.sum(u_v * t1_v, axis=1)[:, None] * t2_v + \
               np.sum(u_v * n1_v, axis=1)[:, None] * n2_v + \
               np.sum(u_v * b_v, axis=1)[:, None] * b_v
    return d

def _signed_angle_batch(u, v, normal):
    """Compute signed angle from batch u to v around batch normal."""
    N = u.shape[0]
    angle = np.zeros(N)
    # Ensure u and v are perpendicular to normal
    u_proj = u - np.sum(u * normal, axis=1)[:, None] * normal
    v_proj = v - np.sum(v * normal, axis=1)[:, None] * normal

    norm_u = np.linalg.norm(u_proj, axis=1)
    norm_v = np.linalg.norm(v_proj, axis=1)

    valid = (norm_u > 1e-10) & (norm_v > 1e-10)
    if not np.any(valid):
        return angle

    u_norm = u_proj[valid] / norm_u[valid, None]
    v_norm = v_proj[valid] / norm_v[valid, None]
    normal_v = normal[valid]

    cos_angle = np.clip(np.sum(u_norm * v_norm, axis=1), -1.0, 1.0)
    sin_angle = np.sum(np.cross(u_norm, v_norm) * normal_v, axis=1)

    angle[valid] = np.arctan2(sin_angle, cos_angle)
    return angle


def _get_strain_twist3D_batch(node0, node1, node2, m1e, m2e, m1f, m2f):
    """Computes twist strain for a batch."""
    N = node0.shape[0]
    tau = np.zeros(N)

    ee = node1 - node0
    ef = node2 - node1
    norm_e = np.linalg.norm(ee, axis=1)
    norm_f = np.linalg.norm(ef, axis=1)

    valid_edges = (norm_e > 1e-10) & (norm_f > 1e-10)
    if not np.any(valid_edges):
        return tau

    te = np.zeros_like(ee)
    tf = np.zeros_like(ef)
    te[valid_edges] = ee[valid_edges] / norm_e[valid_edges, None]
    tf[valid_edges] = ef[valid_edges] / norm_f[valid_edges, None]

    # Transport m1e from te to tf
    m1e_transported = _parallel_transport_batch(m1e, te, tf)

    # Project transported vector onto plane normal to tf
    m1e_transported_proj = m1e_transported - np.sum(m1e_transported * tf, axis=1)[:, None] * tf
    norm_m1e_transported = np.linalg.norm(m1e_transported_proj, axis=1)

    valid_transport = norm_m1e_transported > 1e-10
    valid_indices = valid_edges & valid_transport
    if not np.any(valid_indices):
        return tau

    m1e_t_norm = m1e_transported_proj[valid_indices] / norm_m1e_transported[valid_indices, None]
    m1f_v = m1f[valid_indices]
    tf_v = tf[valid_indices]

    # Project m1f onto plane normal to tf
    m1f_proj = m1f_v - np.sum(m1f_v * tf_v, axis=1)[:, None] * tf_v
    norm_m1f = np.linalg.norm(m1f_proj, axis=1)
    valid_m1f = norm_m1f > 1e-10

    # Indices within the valid_indices subset where m1f projection is valid
    sub_valid_indices = np.where(valid_m1f)[0]
    if sub_valid_indices.size == 0:
        return tau

    m1f_norm = m1f_proj[valid_m1f] / norm_m1f[valid_m1f, None]

    # Calculate angle only for valid m1f projections
    # Map sub_valid_indices back to original indices
    final_valid_indices = np.where(valid_indices)[0][sub_valid_indices]
    tau[final_valid_indices] = _signed_angle_batch(m1e_t_norm[sub_valid_indices], m1f_norm, tf_v[sub_valid_indices])


    return tau # Shape (N,)

def _grad_hess_strain_twist3D_batch(node0, node1, node2, m1e, m2e, m1f, m2f):
    """Computes gradient and Hessian of twist strain for a batch."""
    n_springs = node0.shape[0]
    gradTwist = np.zeros((n_springs, 11))
    hessTwist = np.zeros((n_springs, 11, 11))

    # Edge vectors and lengths
    ee = node1 - node0
    ef = node2 - node1
    norm_e = np.linalg.norm(ee, axis=1)
    norm_f = np.linalg.norm(ef, axis=1)
    norm2_e = norm_e**2
    norm2_f = norm_f**2

    valid_edges = (norm_e > 1e-12) & (norm_f > 1e-12)
    if not np.any(valid_edges):
        return gradTwist, hessTwist

    te = np.zeros_like(ee)
    tf = np.zeros_like(ef)
    te[valid_edges] = ee[valid_edges] / norm_e[valid_edges, None]
    tf[valid_edges] = ef[valid_edges] / norm_f[valid_edges, None]

    dot_prod = np.sum(te * tf, axis=1)
    chi = 1.0 + dot_prod
    valid_chi = np.abs(chi) > 1e-12
    valid_indices = valid_edges & valid_chi

    if not np.any(valid_indices):
        return gradTwist, hessTwist

    # --- Calculate only for valid indices ---
    vi = valid_indices # Shorthand
    norm_e_v, norm_f_v = norm_e[vi], norm_f[vi]
    norm2_e_v, norm2_f_v = norm2_e[vi], norm2_f[vi]
    te_v, tf_v = te[vi], tf[vi]
    chi_v = chi[vi]
    chi_inv_v = 1.0 / chi_v
    kb_v = 2.0 * np.cross(te_v, tf_v) * chi_inv_v[:, None]

    # Gradient of twist wrt DOFs - Panetta's formulation
    # Indices: 0-2: n0_pos, 3-5: n1_pos, 6-8: n2_pos, 9: th_e, 10: th_f
    gradTwist[vi, 0:3] = -0.5 / norm_e_v[:, None] * kb_v
    gradTwist[vi, 6:9] = 0.5 / norm_f_v[:, None] * kb_v
    gradTwist[vi, 3:6] = -(gradTwist[vi, 0:3] + gradTwist[vi, 6:9])
    gradTwist[vi, 9] = -1
    gradTwist[vi, 10] = 1

    # Helper terms for Hessian calculation
    tilde_t_v = (te_v + tf_v) * chi_inv_v[:, None]
    te_plus_tilde_t_v = te_v + tilde_t_v
    tf_plus_tilde_t_v = tf_v + tilde_t_v

    def cross_mat_batch(v):
        zeros = np.zeros_like(v[:, 0])
        return np.array([[zeros, -v[:, 2], v[:, 1]], [v[:, 2], zeros, -v[:, 0]], [-v[:, 1], v[:, 0], zeros]]).transpose(2, 0, 1)

    # Hessian of twist wrt DOFs - Panetta's formulation
    D2mDe2_v = -0.5 / norm2_e_v[:, None, None] * (np.einsum('...i,...j->...ij', kb_v, te_plus_tilde_t_v) + 2.0 * chi_inv_v[:, None, None] * cross_mat_batch(tf_v))
    D2mDf2_v = -0.5 / norm2_f_v[:, None, None] * (np.einsum('...i,...j->...ij', kb_v, tf_plus_tilde_t_v) - 2.0 * chi_inv_v[:, None, None] * cross_mat_batch(te_v))
    D2mDfDe_v = 0.5 / (norm_e_v * norm_f_v)[:, None, None] * (2.0 * chi_inv_v[:, None, None] * cross_mat_batch(te_v) - np.einsum('...i,...j->...ij', kb_v, tilde_t_v))
    D2mDeDf_v = 0.5 / (norm_e_v * norm_f_v)[:, None, None] * (-2.0 * chi_inv_v[:, None, None] * cross_mat_batch(tf_v) - np.einsum('...i,...j->...ij', kb_v, tilde_t_v))

    # Populate the Hessian matrix - Panetta's formulation
    hessTwist[vi, 0:3, 0:3] = D2mDe2_v
    hessTwist[vi, 0:3, 3:6] = -D2mDe2_v + D2mDfDe_v
    hessTwist[vi, 3:6, 0:3] = -D2mDe2_v + D2mDeDf_v
    hessTwist[vi, 3:6, 3:6] = D2mDe2_v - (D2mDeDf_v + D2mDfDe_v) + D2mDf2_v
    hessTwist[vi, 0:3, 6:9] = -D2mDfDe_v
    hessTwist[vi, 6:9, 0:3] = -D2mDeDf_v
    hessTwist[vi, 6:9, 3:6] = D2mDeDf_v - D2mDf2_v
    hessTwist[vi, 3:6, 6:9] = D2mDfDe_v - D2mDf2_v
    hessTwist[vi, 6:9, 6:9] = D2mDf2_v

    # Hessian wrt rotation angles (theta) and mixed terms are assumed zero in Panetta's twist Hessian

    return gradTwist, hessTwist # Shapes (N, 11), (N, 11, 11)

# --- Main Class ---

class GeneralElasticEnergyWunderlich(ElasticEnergy):
    """
    General elastic energy for 3D rods using a neural network model.
    Combines stretching, bending, and twisting effects.
    Uses StretchBendTwistSpring to provide necessary parameters like
    EA, EI1, EI2, GJ, h, ref_len, kappaBar, tauBar, nodes_ind, edges_ind, sgn, and ind.
    """
    def __init__(self,
                 springs: typing.List[StretchBendTwistSpring], # Expects the combined spring type
                 initial_state: RobotState,
                #  nn_model: ElasticEnergyRod3DNN,
                 nn_model: AnalyticalWunderlichElasticEnergy,
                 get_strain = None): # Optional override for get_strain

        # Store NN model and physical parameters from springs
        self.nn_model = nn_model
        self.EA = np.array([s.EA for s in springs], dtype=np.float64)
        self.EI1 = np.array([s.EI1 for s in springs], dtype=np.float64)
        self.EI2 = np.array([s.EI2 for s in springs], dtype=np.float64)
        self.GJ = np.array([s.GJ for s in springs], dtype=np.float64)
        self.h = np.array([s.h for s in springs], dtype=np.float64) # Characteristic dimension
        self.l_eff = np.array([s.ref_len for s in springs], dtype=np.float64) # Effective length

        # Store natural curvatures and twists
        self._kappaBar = np.array([getattr(s, 'kappaBar', [0.0, 0.0]) for s in springs], dtype=np.float64) # Shape (N, 2)
        self._tauBar = np.array([getattr(s, 'tauBar', 0.0) for s in springs], dtype=np.float64) # Shape (N,)

        # Use EA * l_eff as a placeholder stiffness 'K' for the base class.
        stiffness_k = self.EA * self.l_eff

        # Node indices (expecting Nx3) and DOF indices (expecting Nx11 for bend/twist components)
        nodes_ind = np.array([s.nodes_ind for s in springs], dtype=np.int64)
        ind = np.array([s.ind for s in springs], dtype=np.int64) # Should map to 11 DOFs

        # Store edge indices and signs if provided by the spring (like BendTwistSpring)
        if hasattr(springs[0], 'edges_ind') and hasattr(springs[0], 'sgn'):
             self._edges_ind = np.array([s.edges_ind for s in springs], dtype=np.int64)
             self._sgn = np.array([s.sgn for s in springs], dtype=np.float64)
             # Create sign matrices for base class compatibility if needed (though overridden methods don't use them)
             N = len(springs)
             self._sign_grad = np.ones((N, 11))
             for dof_idx, signs in [(9, self._sgn[:, 0]), (10, self._sgn[:, 1])]: # Assuming 9, 10 are theta DOFs
                 self._sign_grad[:, dof_idx] = signs
             self._sign_hess = self._sign_grad[:, :, None] * self._sign_grad[:, None, :]
        else:
             self._edges_ind = None # Mark as unavailable
             self._sgn = None


        super().__init__(
            stiffness_k,
            nodes_ind,
            ind,
            initial_state,
            get_strain # Pass the override if provided
        )
        # Base class __post_init__ will call self.get_strain

    def __post_init__(self):
        # Override post_init to calculate the 4-component natural strain vector
        # Check if get_strain was passed externally
        external_get_strain = self._nat_strain is not None

        if external_get_strain:
            initial_strains_calc = self.get_strain(self._initial_state)
            self._nat_strain = np.where(np.isnan(self._nat_strain), initial_strains_calc, self._nat_strain)
        else:
            self._nat_strain = self.get_strain(self._initial_state).copy()


    def _get_adjusted_material_directors(self, state: RobotState):
        """Helper to get material directors based on stored edge indices and signs."""
        if self._edges_ind is not None and self._sgn is not None:
             m1e = state.m1[self._edges_ind[:, 0]]
             m2e = state.m2[self._edges_ind[:, 0]] * self._sgn[:, 0, None]
             m1f = state.m1[self._edges_ind[:, 1]]
             m2f = state.m2[self._edges_ind[:, 1]] * self._sgn[:, 1, None]
             return m1e, m2e, m1f, m2f
        else:
             # Fallback: Requires RobotState to have appropriate edge mapping or direct access
             # This indicates a potential design dependency on how RobotState provides directors per spring
             # Raising an error might be safer than guessing.
             raise NotImplementedError("GeneralElasticEnergyRod3D requires springs with 'edges_ind' and 'sgn' attributes, or RobotState needs modification.")


    def get_strain(self, state: RobotState) -> np.ndarray:
        """
        Computes the four strain components (stretch, kappa1, kappa2, tau) for each spring.

        Args:
            state: The current state of the robot.

        Returns:
            An array of strains, shape (N, 4).
        """
        n0p, n1p, n2p = self._get_node_pos(state.q) # Shape (N, 3) each
        m1e, m2e, m1f, m2f = self._get_adjusted_material_directors(state)

        stretch = _get_strain_stretch3D_batch(n0p, n1p, n2p, self.l_eff) # Shape (N,)
        kappa = _get_strain_curvature3D_batch(n0p, n1p, n2p, m1e, m2e, m1f, m2f) # Shape (N, 2)
        tau = _get_strain_twist3D_batch(n0p, n1p, n2p, m1e, m2e, m1f, m2f) # Shape (N,)

        # Combine into (N, 4) array
        strains = np.stack([stretch, kappa[:, 0], kappa[:, 1], tau], axis=-1)
        return strains

    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Computes the gradient and Hessian of the four strain components w.r.t. DOFs.

        Args:
            state: The current state of the robot.

        Returns:
            A tuple containing:
                - grad_strains (np.ndarray): Gradient of strains (shape N, 11, 4).
                 - hess_strains (np.ndarray): Hessian of strains (shape N, 11, 11, 4).
        """
        n_springs = self._ind.shape[0]
        n_dof_per_spring = self._ind.shape[1] # Should be 11

        grad_strains = np.zeros((n_springs, n_dof_per_spring, 4))
        hess_strains = np.zeros((n_springs, n_dof_per_spring, n_dof_per_spring, 4))

        n0p, n1p, n2p = self._get_node_pos(state.q)
        m1e, m2e, m1f, m2f = self._get_adjusted_material_directors(state)

        # Stretch Gradient/Hessian (9 DOFs)
        grad_stretch_9, hess_stretch_9 = _grad_hess_strain_stretch3D_batch(n0p, n1p, n2p, self.l_eff)

        # Pad stretch gradient/Hessian to 11 DOFs
        # Indices: 0-2(n0_pos), 3-5(n1_pos), 6-8(n2_pos), 9(th_e), 10(th_f) # Corrected DOF order
        # Stretch grad/hess indices: 0-2(n0_pos), 3-5(n1_pos), 6-8(n2_pos)
        grad_stretch_11 = np.zeros((n_springs, 11))
        hess_stretch_11 = np.zeros((n_springs, 11, 11))

        pos_indices_9 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # The first 9 DOFs correspond to the positions n0, n1, n2
        pos_indices_11 = [0, 1, 2, 3, 4, 5, 6, 7, 8] # Corrected mapping

        grad_stretch_11[:, pos_indices_11] = grad_stretch_9[:, pos_indices_9]
        # Efficiently map the 9x9 Hessian blocks into the 11x11 structure
        ix_ = np.ix_(np.arange(n_springs), pos_indices_11, pos_indices_11) # Use corrected indices
        hess_stretch_11[ix_] = hess_stretch_9

        grad_strains[:, :, 0] = grad_stretch_11
        hess_strains[:, :, :, 0] = hess_stretch_11

        # Curvature Gradient/Hessian (11 DOFs)
        grad_kappa, hess_kappa = _grad_hess_strain_curvature3D_batch(n0p, n1p, n2p, m1e, m2e, m1f, m2f)
        grad_strains[:, :, 1] = grad_kappa[:, :, 0] # Kappa1
        grad_strains[:, :, 2] = grad_kappa[:, :, 1] # Kappa2
        hess_strains[:, :, :, 1] = hess_kappa[:, :, :, 0] # Kappa1
        hess_strains[:, :, :, 2] = hess_kappa[:, :, :, 1] # Kappa2

        # Twist Gradient/Hessian (11 DOFs)
        grad_twist, hess_twist = _grad_hess_strain_twist3D_batch(n0p, n1p, n2p, m1e, m2e, m1f, m2f)
        grad_strains[:, :, 3] = grad_twist
        hess_strains[:, :, :, 3] = hess_twist

        return grad_strains, hess_strains

    # --- Override Base Class Methods ---

    def get_energy_linear_elastic(self, state: RobotState, output_scalar: bool = True):
        """
        Computes the total elastic energy using the NN model.
        Overrides the base class method.
        """
        n_springs = self._ind.shape[0]
        energies = np.zeros(n_springs)

        # Get current strains (stretch, k1, k2, tau)
        current_strains = self.get_strain(state) # Shape (N, 4)
        nat_strains = self._nat_strain # Shape (N, 4)

        # Calculate delta strains
        delta_strains = current_strains - nat_strains

        # Scale kappa and tau components of delta strain for NN input
        strains_input_nn = delta_strains.copy()
        strains_input_nn[:, 1] *= self.h / self.l_eff # Scale delta kappa1
        strains_input_nn[:, 2] *= self.h / self.l_eff # Scale delta kappa2
        strains_input_nn[:, 3] *= self.h / self.l_eff # Scale delta tau
        # Stretch component (index 0) remains unscaled delta stretch

        # Convert to physical strains for finite difference calculation
        physical_strains = current_strains.copy()  # Already in physical units

        # Batch computation: compute η' for all elements at once
        eta_prime_batch = self.nn_model.compute_eta_prime_batch(physical_strains)  # Shape (n_springs,)
        eta_prime_tensor = torch.tensor(eta_prime_batch, dtype=torch.float64)

        # Convert to torch tensors for batch processing
        strains_input_torch = torch.tensor(strains_input_nn, dtype=torch.float64)

        # Compute energy for all elements in batch
        E_nn_batch = self.nn_model.forward(strains_input_torch, eta_prime_tensor).squeeze()  # Shape (n_springs,)

        # Scale by physical parameters (0.5 * EA * l_eff) for each element
        scale_factors = 0.5 * self.EA * self.l_eff  # Shape (n_springs,)
        energies = (scale_factors * E_nn_batch.numpy())

        if output_scalar:
            return np.sum(energies)
        else:
            return energies # Return energy per spring

    def grad_hess_energy_linear_elastic(self, state: RobotState, sparse: bool = False) -> typing.Tuple[np.ndarray, np.ndarray] | typing.Tuple[np.ndarray, sp.csr_array]:
        """
        Computes the gradient and Hessian of the total elastic energy using the NN model.
        Overrides the base class method.
        """
        n_springs = self._ind.shape[0]
        n_dof_total = state.q.shape[0]
        n_dof_per_spring = self._ind.shape[1] # Should be 11

        # Initialize global gradient and Hessian components
        Fs = np.zeros(n_dof_total)
        Js_data = []
        Js_rows = []
        Js_cols = []

        # 1. Get current strains and their gradients/Hessians w.r.t DOFs
        current_strains = self.get_strain(state) # Shape (N, 4)
        nat_strains = self._nat_strain # Shape (N, 4)
        grad_strains_dof, hess_strains_dof = self.grad_hess_strain(state)
        # Shapes: (N, 11, 4), (N, 11, 11, 4)

        # 2. Calculate delta strains and scale kappa/tau components for NN input
        delta_strains = current_strains - nat_strains
        strains_input_nn = delta_strains.copy()
        strains_input_nn[:, 1] *= self.h / self.l_eff # Scale delta kappa1
        strains_input_nn[:, 2] *= self.h / self.l_eff # Scale delta kappa2
        strains_input_nn[:, 3] *= self.h / self.l_eff # Scale delta tau

        # 3. Scale strain gradients/Hessians for chain rule application
        grad_strains_dof_scaled = grad_strains_dof.copy()
        hess_strains_dof_scaled = hess_strains_dof.copy()
        # Scaling factors (N,) -> (N, 1) or (N, 1, 1) for broadcasting
        h_l = (self.h / self.l_eff)[:, None]
        h_l_hess = h_l[:, :, None]
        # Scale kappa1 components (index 1)
        grad_strains_dof_scaled[:, :, 1] *= h_l
        hess_strains_dof_scaled[:, :, :, 1] *= h_l_hess
        # Scale kappa2 components (index 2)
        grad_strains_dof_scaled[:, :, 2] *= h_l
        hess_strains_dof_scaled[:, :, :, 2] *= h_l_hess
        # Scale tau components (index 3)
        grad_strains_dof_scaled[:, :, 3] *= h_l
        hess_strains_dof_scaled[:, :, :, 3] *= h_l_hess
        # Stretch components (index 0) remain unscaled

        # Convert to physical strains for finite difference calculation
        physical_strains = current_strains.copy()

        # 4. Batch computation: compute η' for all elements at once
        eta_prime_batch = self.nn_model.compute_eta_prime_batch(physical_strains)  # Shape (n_springs,)
        eta_prime_tensor = torch.tensor(eta_prime_batch, dtype=torch.float64)

        # Convert to torch tensors for batch processing
        strains_input_torch = torch.tensor(strains_input_nn, dtype=torch.float64, requires_grad=True)

        # Compute energy, gradient, and Hessian for all elements in batch
        _, gradE_strain_nn_batch, hessE_strain_nn_batch, gradE_eta_prime_batch = self.nn_model.compute_energy_grad_hess_batch(
            strains_input_torch, eta_prime_tensor
        )
        # Shapes: (n_springs, 4), (n_springs, 4, 4), (n_springs,)

        # 5. Apply chain rule for all springs in batch (vectorized)
        # gradE_dof_local: (n_springs, 11)
        # gradE_strain_nn_batch: (n_springs, 4), grad_strains_dof_scaled: (n_springs, 11, 4)
        # Sum over strain components k: sum_k gradE[i,k] * grad_strain[i,j,k]
        gradE_dof_local = np.einsum('ik,ijk->ij', gradE_strain_nn_batch, grad_strains_dof_scaled)
        
        # hessE_dof_local: (n_springs, 11, 11)
        # hess_strains_dof_scaled: (n_springs, 11, 11, 4) where last dim is strain component
        # Term 1: sum over strain components k
        # sum_k gradE[i,k] * hess_strain[i,j,l,k]
        term1 = np.einsum('ik,ijlk->ijl', gradE_strain_nn_batch, hess_strains_dof_scaled)
        
        # Term 2: sum over strain components k, l
        # For each spring s: sum_{k,l} H_skl * G_sik * G_sjl
        # hessE_strain_nn_batch: (n_springs, 4, 4), grad_strains_dof_scaled: (n_springs, 11, 4)
        # sum_{k,l} H_skl * G_sik * G_sjl -> (n_springs, 11, 11)
        term2 = np.einsum('skl,sik,sjl->sij', 
                         hessE_strain_nn_batch, 
                         grad_strains_dof_scaled, 
                         grad_strains_dof_scaled)
        
        hessE_dof_local = term1 + term2  # Shape (n_springs, 11, 11)

        # Scale by physical factors (0.5 * EA * l_eff) for all springs
        scale_factors_1d = 0.5 * self.EA * self.l_eff  # (n_springs,)
        scale_factors_2d = scale_factors_1d[:, None]  # (n_springs, 1)
        scale_factors_3d = scale_factors_1d[:, None, None]  # (n_springs, 1, 1)
        
        gradE_dof_local *= scale_factors_2d  # Broadcast to (n_springs, 11)
        hessE_dof_local *= scale_factors_3d  # Broadcast to (n_springs, 11, 11)

        # 5b. Add contributions from eta_prime chain rule (VECTORIZED)
        # For each element k, eta_prime_k depends on strains of k-1, k, k+1
        # We need to compute: ∂E_k/∂η'_k * ∂η'_k/∂q and add to global gradient/Hessian
        # ∂η'_k/∂q = Σ_m (∂η'_k/∂strain_m * ∂strain_m/∂q) where m ∈ {k-1, k, k+1}
        
        # Scale gradE_eta_prime by physical factors
        gradE_eta_prime_scaled = gradE_eta_prime_batch * scale_factors_1d  # (n_springs,)
        
        # Extract kappa1 and tau for all elements
        kappa1_all = physical_strains[:, 1]  # (n_springs,)
        tau_all = physical_strains[:, 3]     # (n_springs,)
        
        # Mask for valid elements (non-zero kappa1 and non-negligible gradient)
        valid_mask = (np.abs(kappa1_all) > self.nn_model.eps) & (np.abs(gradE_eta_prime_scaled) > 1e-15)
        
        if not np.any(valid_mask):
            # No contributions to add
            pass
        else:
            # Vectorized computation for middle elements (central difference)
            if n_springs >= 3:
                mid_indices = np.arange(1, n_springs - 1)  # Indices 1 to n_springs-2
                mid_valid = valid_mask[mid_indices]
                
                if np.any(mid_valid):
                    k_mid = mid_indices[mid_valid]
                    
                    # Extract values for middle elements
                    kappa1_k = kappa1_all[k_mid]
                    tau_k = tau_all[k_mid]
                    kappa1_km1 = kappa1_all[k_mid - 1]
                    kappa1_kp1 = kappa1_all[k_mid + 1]
                    tau_km1 = tau_all[k_mid - 1]
                    tau_kp1 = tau_all[k_mid + 1]
                    gradE_eta_k = gradE_eta_prime_scaled[k_mid]
                    
                    # Compute derivatives (vectorized)
                    kappa1_k_sq = kappa1_k**2
                    kappa1_k_cubed = kappa1_k**3
                    factor = self.nn_model.delta_l / 2.0
                    
                    # ∂η'_k/∂κ₁_{k-1}, ∂η'_k/∂τ_{k-1}
                    d_eta_prime_d_kappa1_m1 = factor * tau_k / kappa1_k_sq
                    d_eta_prime_d_tau_m1 = factor * (-kappa1_k) / kappa1_k_sq
                    
                    # ∂η'_k/∂κ₁_k, ∂η'_k/∂τ_k
                    d_eta_prime_d_kappa1_k = factor * (tau_kp1 - tau_km1 - 2 * tau_k * (kappa1_kp1 - kappa1_km1) / kappa1_k) / kappa1_k_sq
                    d_eta_prime_d_tau_k = factor * (kappa1_kp1 - kappa1_km1) / kappa1_k_sq
                    
                    # ∂η'_k/∂κ₁_{k+1}, ∂η'_k/∂τ_{k+1}
                    d_eta_prime_d_kappa1_p1 = factor * (-tau_k) / kappa1_k_sq
                    d_eta_prime_d_tau_p1 = factor * kappa1_k / kappa1_k_sq
                    
                    # Vectorized computation of contributions
                    # Shape: (len(k_mid), 11) for each contribution type
                    n_mid = len(k_mid)
                    
                    # Contributions to element k-1: (n_mid, 11)
                    contrib_km1 = (d_eta_prime_d_kappa1_m1[:, None] * grad_strains_dof[k_mid - 1, :, 1] + 
                                 d_eta_prime_d_tau_m1[:, None] * grad_strains_dof[k_mid - 1, :, 3]) * gradE_eta_k[:, None]
                    
                    # Contributions to element k: (n_mid, 11)
                    contrib_k = (d_eta_prime_d_kappa1_k[:, None] * grad_strains_dof[k_mid, :, 1] + 
                               d_eta_prime_d_tau_k[:, None] * grad_strains_dof[k_mid, :, 3]) * gradE_eta_k[:, None]
                    
                    # Contributions to element k+1: (n_mid, 11)
                    contrib_kp1 = (d_eta_prime_d_kappa1_p1[:, None] * grad_strains_dof[k_mid + 1, :, 1] + 
                                  d_eta_prime_d_tau_p1[:, None] * grad_strains_dof[k_mid + 1, :, 3]) * gradE_eta_k[:, None]
                    
                    # Fully vectorized accumulation using np.add.at
                    # Flatten contributions and corresponding DOF indices
                    dof_indices_km1 = self._ind[k_mid - 1, :].flatten()  # (n_mid * 11,)
                    dof_indices_k = self._ind[k_mid, :].flatten()  # (n_mid * 11,)
                    dof_indices_kp1 = self._ind[k_mid + 1, :].flatten()  # (n_mid * 11,)
                    
                    contrib_km1_flat = -contrib_km1.flatten()  # (n_mid * 11,)
                    contrib_k_flat = -contrib_k.flatten()  # (n_mid * 11,)
                    contrib_kp1_flat = -contrib_kp1.flatten()  # (n_mid * 11,)
                    
                    # Accumulate all contributions at once
                    np.add.at(Fs, dof_indices_km1, contrib_km1_flat)
                    np.add.at(Fs, dof_indices_k, contrib_k_flat)
                    np.add.at(Fs, dof_indices_kp1, contrib_kp1_flat)
            
            # Handle boundary elements (first and last)
            if n_springs >= 2:
                # First element (forward difference)
                if valid_mask[0]:
                    k = 0
                    kappa1_k = kappa1_all[k]
                    tau_k = tau_all[k]
                    kappa1_k1 = kappa1_all[k+1]
                    tau_k1 = tau_all[k+1]
                    kappa1_k_sq = kappa1_k**2
                    kappa1_k_cubed = kappa1_k**3
                    
                    d_eta_prime_d_kappa1_0 = self.nn_model.delta_l * (-tau_k1 * kappa1_k + 2 * tau_k * kappa1_k1) / kappa1_k_cubed
                    d_eta_prime_d_tau_0 = self.nn_model.delta_l * (-kappa1_k1) / kappa1_k_sq
                    d_eta_prime_d_kappa1_1 = self.nn_model.delta_l * (-tau_k) / kappa1_k_sq
                    d_eta_prime_d_tau_1 = self.nn_model.delta_l * kappa1_k / kappa1_k_sq
                    
                    contrib_k = (d_eta_prime_d_kappa1_0 * grad_strains_dof[k, :, 1] + 
                                d_eta_prime_d_tau_0 * grad_strains_dof[k, :, 3]) * gradE_eta_prime_scaled[k]
                    Fs[self._ind[k, :]] -= contrib_k
                    
                    contrib_k1 = (d_eta_prime_d_kappa1_1 * grad_strains_dof[k+1, :, 1] + 
                                 d_eta_prime_d_tau_1 * grad_strains_dof[k+1, :, 3]) * gradE_eta_prime_scaled[k]
                    Fs[self._ind[k+1, :]] -= contrib_k1
                
                # Last element (backward difference)
                if valid_mask[-1]:
                    k = n_springs - 1
                    kappa1_k = kappa1_all[k]
                    tau_k = tau_all[k]
                    kappa1_km1 = kappa1_all[k-1]
                    tau_km1 = tau_all[k-1]
                    kappa1_k_sq = kappa1_k**2
                    kappa1_k_cubed = kappa1_k**3
                    
                    d_eta_prime_d_kappa1_m1 = self.nn_model.delta_l * tau_k / kappa1_k_sq
                    d_eta_prime_d_tau_m1 = self.nn_model.delta_l * (-kappa1_k) / kappa1_k_sq
                    d_eta_prime_d_kappa1_k = self.nn_model.delta_l * (-tau_km1 * kappa1_k + 2 * tau_k * kappa1_km1) / kappa1_k_cubed
                    d_eta_prime_d_tau_k = self.nn_model.delta_l * kappa1_km1 / kappa1_k_sq
                    
                    contrib_km1 = (d_eta_prime_d_kappa1_m1 * grad_strains_dof[k-1, :, 1] + 
                                  d_eta_prime_d_tau_m1 * grad_strains_dof[k-1, :, 3]) * gradE_eta_prime_scaled[k]
                    Fs[self._ind[k-1, :]] -= contrib_km1
                    
                    contrib_k = (d_eta_prime_d_kappa1_k * grad_strains_dof[k, :, 1] + 
                                d_eta_prime_d_tau_k * grad_strains_dof[k, :, 3]) * gradE_eta_prime_scaled[k]
                    Fs[self._ind[k, :]] -= contrib_k

        # 6. Assemble into global gradient/Hessian
        for i in range(n_springs):
            dof_indices = self._ind[i, :]  # Global DOFs for this spring
            Fs[dof_indices] -= gradE_dof_local[i, :]  # Subtract gradient for force

            # For sparse Hessian assembly
            rows_local = np.repeat(dof_indices, n_dof_per_spring)
            cols_local = np.tile(dof_indices, n_dof_per_spring)
            Js_data.append(-hessE_dof_local[i, :, :].ravel())
            Js_rows.append(rows_local)
            Js_cols.append(cols_local)

        # Finalize Hessian
        if not Js_data: # Handle case with no springs
             if sparse:
                 return Fs, sp.csr_matrix((n_dof_total, n_dof_total))
             else:
                 return Fs, np.zeros((n_dof_total, n_dof_total))

        Js_data_flat = np.concatenate(Js_data)
        Js_rows_flat = np.concatenate(Js_rows)
        Js_cols_flat = np.concatenate(Js_cols)

        if sparse:
            Js = sp.coo_matrix((Js_data_flat, (Js_rows_flat, Js_cols_flat)),
                               shape=(n_dof_total, n_dof_total)).tocsr()
            Js.sum_duplicates() # Sum contributions for shared DOFs
        else:
            Js = np.zeros((n_dof_total, n_dof_total))
            np.add.at(Js, (Js_rows_flat, Js_cols_flat), Js_data_flat)

        return Fs, Js
