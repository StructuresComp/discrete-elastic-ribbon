import typing
import numpy as np
import scipy.sparse as sp
import torch
import dataclasses

from .elastic_energy import ElasticEnergy
from ..state import RobotState
from .analytical_kirchhoff_elastic_energy import AnalyticalKirchhoffElasticEnergy
from ..springs import StretchBendTwistSpring

# Import strain calculation helpers from general_elastic_energy.py
from .general_elastic_energy import (
    _get_strain_stretch3D_batch,
    _grad_hess_strain_stretch3D_batch,
    _get_strain_curvature3D_batch,
    _grad_hess_strain_curvature3D_batch,
    _get_strain_twist3D_batch,
    _grad_hess_strain_twist3D_batch
)

# --- Main Class ---

class GeneralElasticEnergyKirchhoff(ElasticEnergy):
    """
    General elastic energy for 3D rods using the Kirchhoff analytical energy model.
    Combines stretching, bending, and twisting effects with vectorized batch processing.
    Uses StretchBendTwistSpring to provide necessary parameters like
    EA, EI1, EI2, GJ, h, ref_len, kappaBar, tauBar, nodes_ind, edges_ind, sgn, and ind.
    """
    def __init__(self,
                 springs: typing.List[StretchBendTwistSpring], # Expects the combined spring type
                 initial_state: RobotState,
                 nn_model: AnalyticalKirchhoffElasticEnergy,
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
             raise NotImplementedError("GeneralElasticEnergyKirchhoff requires springs with 'edges_ind' and 'sgn' attributes, or RobotState needs modification.")


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
        # Indices: 0-2(n0_pos), 3-5(n1_pos), 6-8(n2_pos), 9(th_e), 10(th_f)
        grad_stretch_11 = np.zeros((n_springs, 11))
        hess_stretch_11 = np.zeros((n_springs, 11, 11))

        pos_indices_9 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        pos_indices_11 = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        grad_stretch_11[:, pos_indices_11] = grad_stretch_9[:, pos_indices_9]
        ix_ = np.ix_(np.arange(n_springs), pos_indices_11, pos_indices_11)
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
        Computes the total elastic energy using the Kirchhoff analytical model with batch processing.
        Overrides the base class method.
        """
        n_springs = self._ind.shape[0]

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

        # Convert to torch tensors for batch processing
        strains_input_torch = torch.tensor(strains_input_nn, dtype=torch.float64)

        # Compute energy for all elements in batch
        E_nn_batch = self.nn_model.forward(strains_input_torch).squeeze()  # Shape (n_springs,)

        # Scale by physical parameters (0.5 * EA * l_eff) for each element
        scale_factors = 0.5 * self.EA * self.l_eff  # Shape (n_springs,)
        energies = (scale_factors * E_nn_batch.numpy())

        if output_scalar:
            return np.sum(energies)
        else:
            return energies # Return energy per spring

    def grad_hess_energy_linear_elastic(self, state: RobotState, sparse: bool = False) -> typing.Tuple[np.ndarray, np.ndarray] | typing.Tuple[np.ndarray, sp.csr_array]:
        """
        Computes the gradient and Hessian of the total elastic energy using the Kirchhoff analytical model.
        Fully vectorized batch processing implementation.
        Overrides the base class method.
        """
        n_springs = self._ind.shape[0]
        n_dof_total = state.q.shape[0]
        n_dof_per_spring = self._ind.shape[1]

        Fs = np.zeros(n_dof_total)
        Js_data, Js_rows, Js_cols = [], [], []

        current_strains = self.get_strain(state)
        nat_strains = self._nat_strain
        grad_strains_dof, hess_strains_dof = self.grad_hess_strain(state)

        delta_strains = current_strains - nat_strains
        strains_input_nn = delta_strains.copy()
        strains_input_nn[:, 1] *= self.h / self.l_eff
        strains_input_nn[:, 2] *= self.h / self.l_eff
        strains_input_nn[:, 3] *= self.h / self.l_eff

        grad_strains_dof_scaled = grad_strains_dof.copy()
        hess_strains_dof_scaled = hess_strains_dof.copy()
        h_l = (self.h / self.l_eff)[:, None]
        h_l_hess = h_l[:, :, None]
        grad_strains_dof_scaled[:, :, 1] *= h_l
        hess_strains_dof_scaled[:, :, :, 1] *= h_l_hess
        grad_strains_dof_scaled[:, :, 2] *= h_l
        hess_strains_dof_scaled[:, :, :, 2] *= h_l_hess
        grad_strains_dof_scaled[:, :, 3] *= h_l
        hess_strains_dof_scaled[:, :, :, 3] *= h_l_hess

        strains_input_torch = torch.tensor(strains_input_nn, dtype=torch.float64, requires_grad=True)

        _, gradE_strain_nn_batch, hessE_strain_nn_batch = self.nn_model.compute_energy_grad_hess_batch(
            strains_input_torch
        )

        gradE_dof_local = np.einsum('ik,ijk->ij', gradE_strain_nn_batch, grad_strains_dof_scaled)
        term1 = np.einsum('ik,ijlk->ijl', gradE_strain_nn_batch, hess_strains_dof_scaled)
        term2 = np.einsum('skl,sik,sjl->sij', 
                         hessE_strain_nn_batch, 
                         grad_strains_dof_scaled, 
                         grad_strains_dof_scaled)
        hessE_dof_local = term1 + term2

        scale_factors_1d = 0.5 * self.EA * self.l_eff
        scale_factors_2d = scale_factors_1d[:, None]
        scale_factors_3d = scale_factors_1d[:, None, None]
        
        gradE_dof_local *= scale_factors_2d
        hessE_dof_local *= scale_factors_3d

        for i in range(n_springs):
            dof_indices = self._ind[i, :]
            Fs[dof_indices] -= gradE_dof_local[i, :]

            rows_local = np.repeat(dof_indices, n_dof_per_spring)
            cols_local = np.tile(dof_indices, n_dof_per_spring)
            Js_data.append(-hessE_dof_local[i, :, :].ravel())
            Js_rows.append(rows_local)
            Js_cols.append(cols_local)

        if not Js_data:
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
            Js.sum_duplicates()
        else:
            Js = np.zeros((n_dof_total, n_dof_total))
            np.add.at(Js, (Js_rows_flat, Js_cols_flat), Js_data_flat)

        return Fs, Js

