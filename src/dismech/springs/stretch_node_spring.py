import typing
import numpy as np


class StretchNodeSpring:

    def __init__(self,
                 nodes_edges_index: np.ndarray,
                 ref_len: float,
                 EA: float,
                 map_node_to_dof: typing.Callable[[np.ndarray], np.ndarray]):
        self.EA = EA
        self.ref_len = ref_len
        self.nodes_ind = [int(nodes_edges_index[0]),
                          int(nodes_edges_index[2]),
                          int(nodes_edges_index[4])]
        self.ind = np.concat([map_node_to_dof(self.nodes_ind[0]),
                            map_node_to_dof(self.nodes_ind[1]),
                            map_node_to_dof(self.nodes_ind[2])])
