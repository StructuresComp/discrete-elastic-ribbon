import typing
import numpy as np

from .elastic_energy import ElasticEnergy
from ..state import RobotState
from ..springs import StretchNodeSpring


class StretchNodeEnergy(ElasticEnergy):
    def __init__(self, springs: typing.List[StretchNodeSpring], initial_state: RobotState, get_strain=None):
        """
        Initializes the StretchNodeEnergy class.

        Args:
            springs: A list of StretchNodeSpring objects.
            initial_state: The initial state of the robot.
            get_strain: Optional function to compute strain (defaults to internal method).
        """
        self.l_k = np.array([s.ref_len for s in springs], dtype=np.float64)
        self.inv_l_k = 1.0 / self.l_k
        self.EA = np.array([s.EA for s in springs], dtype=np.float64)

        # The stiffness 'k' for the base class energy calculation E = 0.5 * k * strain^2
        # From the notebook: E_s = 0.5 * EA * strain_stretch**2 * l_k
        # So, k = EA * l_k
        stiffness_k = self.l_k * self.EA

        super().__init__(
            stiffness_k,
            np.array([s.nodes_ind for s in springs], dtype=np.int64), # Should be Nx3
            np.array([s.ind for s in springs], dtype=np.int64), # Should be Nx9
            initial_state,
            get_strain
        )

    def get_strain(self, state: RobotState) -> np.ndarray:
        """
        Computes the axial stretch strain at each node.
        Strain is defined as the average stretch of the two connected edges.

        Args:
            state: The current state of the robot.

        Returns:
            An array of stretch strains for each node spring (shape N, 1).
        """
        n0p, n1p, n2p = self._get_node_pos(state.q) # Get positions of node0, node1, node2 for each spring

        # Calculate stretch for the first edge (node0 to node1)
        edge0 = n1p - n0p
        edge_len0 = np.linalg.norm(edge0, axis=1)
        stretch0 = edge_len0 * self.inv_l_k - 1.0

        # Calculate stretch for the second edge (node1 to node2)
        edge1 = n2p - n1p
        edge_len1 = np.linalg.norm(edge1, axis=1)
        stretch1 = edge_len1 * self.inv_l_k - 1.0

        # Node strain is the average of the two edge stretches
        epsilon_node = 0.5 * stretch0 + 0.5 * stretch1

        return epsilon_node[:, None] # Return as column vector (N, 1)

    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Computes the gradient and Hessian of the axial stretch strain at each node
        with respect to the DOFs of the three involved nodes (9 DOFs).

        Args:
            state: The current state of the robot.

        Returns:
            A tuple containing:
                - grad_eps (np.ndarray): Gradient of strain (shape N, 9, 1).
                - hess_eps (np.ndarray): Hessian of strain (shape N, 9, 9, 1).
        """
        n0p, n1p, n2p = self._get_node_pos(state.q) # Shape (N, 3) each
        N = n0p.shape[0]
        Id3 = np.eye(3) # Identity matrix

        # --- Calculations for the first edge (node0 to node1) ---
        edge0 = n1p - n0p
        edge_len0 = np.linalg.norm(edge0, axis=1)
        # Handle potential zero length edges to avoid division by zero
        valid_len0 = edge_len0 > 1e-12
        tangent0 = np.zeros_like(edge0)
        tangent0[valid_len0] = edge0[valid_len0] / edge_len0[valid_len0, None]
        eps0 = edge_len0 * self.inv_l_k - 1.0

        # Gradient of stretch for edge 0 w.r.t edge vector
        dF_unit0 = tangent0 * self.inv_l_k[:, None]

        # Hessian of stretch for edge 0 w.r.t edge vector (M2_0)
        M0_term1 = np.zeros((N, 3, 3))
        M0_term2 = np.zeros((N, 3, 3))
        if np.any(valid_len0):
             M0_term1[valid_len0] = (self.inv_l_k[valid_len0, None, None] - 1.0 / edge_len0[valid_len0, None, None]) * Id3
             M0_term2[valid_len0] = np.einsum('...i,...j->...ij', edge0[valid_len0], edge0[valid_len0]) / (edge_len0[valid_len0]**3)[:, None, None]
        M0 = 2.0 * self.inv_l_k[:, None, None] * (M0_term1 + M0_term2)


        M2_0 = M0 - 2.0 * np.einsum('...i,...j->...ij', dF_unit0, dF_unit0)
        M2_0 *= 0.5
        mask0 = eps0 != 0
        M2_0 = np.divide(M2_0, eps0[:, None, None], where=mask0[:, None, None], out=np.zeros_like(M2_0))

        # --- Calculations for the second edge (node1 to node2) ---
        edge1 = n2p - n1p
        edge_len1 = np.linalg.norm(edge1, axis=1)
        valid_len1 = edge_len1 > 1e-12
        tangent1 = np.zeros_like(edge1)
        tangent1[valid_len1] = edge1[valid_len1] / edge_len1[valid_len1, None]
        eps1 = edge_len1 * self.inv_l_k - 1.0

        # Gradient of stretch for edge 1 w.r.t edge vector
        dF_unit1 = tangent1 * self.inv_l_k[:, None]

        # Hessian of stretch for edge 1 w.r.t edge vector (M2_1)
        M1_term1 = np.zeros((N, 3, 3))
        M1_term2 = np.zeros((N, 3, 3))
        if np.any(valid_len1):
            M1_term1[valid_len1] = (self.inv_l_k[valid_len1, None, None] - 1.0 / edge_len1[valid_len1, None, None]) * Id3
            M1_term2[valid_len1] = np.einsum('...i,...j->...ij', edge1[valid_len1], edge1[valid_len1]) / (edge_len1[valid_len1]**3)[:, None, None]
        M1 = 2.0 * self.inv_l_k[:, None, None] * (M1_term1 + M1_term2)


        M2_1 = M1 - 2.0 * np.einsum('...i,...j->...ij', dF_unit1, dF_unit1)
        M2_1 *= 0.5
        mask1 = eps1 != 0
        M2_1 = np.divide(M2_1, eps1[:, None, None], where=mask1[:, None, None], out=np.zeros_like(M2_1))

        # --- Combine gradients and Hessians for the node ---
        grad_eps = np.zeros((N, 9))
        hess_eps = np.zeros((N, 9, 9))

        # Edge 0-1 contribution (factor 0.5 for averaging)
        grad_eps[:, 0:3] = -0.5 * dF_unit0
        grad_eps[:, 3:6] = 0.5 * dF_unit0
        hess_eps[:, 0:3, 0:3] = 0.5 * M2_0
        hess_eps[:, 3:6, 3:6] = 0.5 * M2_0
        hess_eps[:, 0:3, 3:6] = -0.5 * M2_0
        hess_eps[:, 3:6, 0:3] = -0.5 * M2_0

        # Edge 1-2 contribution (factor 0.5 for averaging)
        grad_eps[:, 3:6] += -0.5 * dF_unit1 # Add to middle node's gradient
        grad_eps[:, 6:9] = 0.5 * dF_unit1
        hess_eps[:, 3:6, 3:6] += 0.5 * M2_1 # Add to middle node's Hessian block
        hess_eps[:, 6:9, 6:9] = 0.5 * M2_1
        hess_eps[:, 3:6, 6:9] = -0.5 * M2_1
        hess_eps[:, 6:9, 3:6] = -0.5 * M2_1

        # Return with added dimension for strain component
        return grad_eps[:, :, None], hess_eps[:, :, :, None]
