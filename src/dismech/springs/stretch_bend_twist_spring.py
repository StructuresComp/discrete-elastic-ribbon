import typing
import numpy as np

class StretchBendTwistSpring:
    """
    Combined spring model incorporating stretch, bending, and twisting effects.
    Intended for use with GeneralElasticEnergyRod3D.
    """
    def __init__(self,
                 nodes_edges_index: np.ndarray, # Expects [n0, e0, n1, e1, n2]
                 signs: np.ndarray, # Expects [sign_e0, sign_e1] for material frame consistency
                 ref_len: float, # Effective length for stretching (l_eff)
                 ref_len_arr: np.ndarray, # Array of all edge reference lengths in the system
                 EA: float, # Stretch stiffness
                 EI1: float, # Bending stiffness 1
                 EI2: float, # Bending stiffness 2
                 GJ: float, # Twisting stiffness
                 h: float, # Characteristic dimension (e.g., thickness) for scaling strains
                 map_node_to_dof: typing.Callable[[int], np.ndarray],
                 map_edge_to_dof: typing.Callable[[int], int],
                 kappaBar: np.ndarray = np.array([0.0, 0.0]), # Natural curvature [kappa1, kappa2]
                 tauBar: float = 0.0): # Natural twist

        # Store physical parameters
        self.EA = EA
        self.EI1 = EI1
        self.EI2 = EI2
        self.GJ = GJ
        self.h = h
        self.ref_len = ref_len # This is l_eff used in GeneralElasticEnergyRod3D
        self.kappaBar = np.asarray(kappaBar, dtype=np.float64)
        self.tauBar = float(tauBar)

        # Store topology and frame information
        self.nodes_ind = np.array([int(nodes_edges_index[0]),
                                   int(nodes_edges_index[2]),
                                   int(nodes_edges_index[4])], dtype=np.int64)
        self.edges_ind = np.array([int(nodes_edges_index[1]),
                                   int(nodes_edges_index[3])], dtype=np.int64)
        self.sgn = np.asarray(signs, dtype=np.float64)

        # Calculate DOF indices (11 DOFs: 3 nodes * 3 pos + 2 edges * 1 angle)
        self.ind = np.concatenate((
            map_node_to_dof(self.nodes_ind[0]), # 3 DOFs
            map_node_to_dof(self.nodes_ind[1]), # 3 DOFs
            map_node_to_dof(self.nodes_ind[2]), # 3 DOFs
            np.array([map_edge_to_dof(self.edges_ind[0])], dtype=np.int64), # 1 DOF (theta_e)
            np.array([map_edge_to_dof(self.edges_ind[1])], dtype=np.int64)  # 1 DOF (theta_f)
        ), axis=0)

        # Calculate Voronoi length (used internally by some original energy forms, might not be needed by NN version)
        self.voronoi_len = 0.5 * (ref_len_arr[self.edges_ind[0]] +
                                  ref_len_arr[self.edges_ind[1]])

        # --- Parameter Validation (Optional but recommended) ---
        if self.kappaBar.shape != (2,):
            raise ValueError(f"kappaBar must be a 2-element array, got shape {self.kappaBar.shape}")
        if self.sgn.shape != (2,):
             raise ValueError(f"signs must be a 2-element array, got shape {self.sgn.shape}")
        if len(self.ind) != 11:
             raise ValueError(f"Expected 11 DOFs, but calculated {len(self.ind)} based on mapping functions.")
