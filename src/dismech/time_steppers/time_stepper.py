import abc
import copy
import typing
import warnings

import scipy.sparse as sp
import numpy as np

import time
from tqdm import tqdm

from ..soft_robot import SoftRobot
from ..state import RobotState
from ..elastics import (ElasticEnergy, StretchEnergy, StretchNodeEnergy, HingeEnergy, BendEnergy, TriangleEnergy, TwistEnergy, 
                        GeneralElasticEnergyRod3D, ElasticEnergyRod3DNN, 
                        AnalyticalSanosElasticEnergy, AnalyticalWunderlichElasticEnergy, 
                        AnalyticalKirchhoffElasticEnergy, AnalyticalSadowskyElasticEnergy,
                        AnalyticalAudolyElasticEnergy,
                        GeneralElasticEnergyWunderlich, GeneralElasticEnergySano,
                        GeneralElasticEnergyKirchhoff, GeneralElasticEnergySadowsky,
                        GeneralElasticEnergyAudoly)
from ..external_forces import compute_gravity_forces, compute_aerodynamic_forces_vectorized, compute_damping_force
from ..solvers import Solver, NumpySolver, PardisoSolver, RobustSolver
from ..visualizer import Visualizer

_SOLVERS: typing.Dict[str, Solver] = {
    'np': NumpySolver, 'pardiso': PardisoSolver}


class TimeStepper(metaclass=abc.ABCMeta):

    def __init__(self, robot: SoftRobot, min_force=1e-8, dtype=np.float64, energy_model='sano', 
                 sano_zeta=None, wunderlich_W=None, cond_threshold=None):
        """
        Initialize the time stepper.
        
        Args:
            robot: SoftRobot instance
            min_force: Minimum force threshold
            dtype: Data type for computations
            energy_model: Energy model to use ('sano', 'wunderlich', 'kirchhoff', 'sadowsky', or 'audoly')
            sano_zeta: Zeta parameter for Sano model (if None, computed from geometry)
            wunderlich_W: W parameter for Wunderlich model (if None, computed from geometry)
            cond_threshold: Condition number threshold for RobustSolver (if None, uses default 1e12)
        """
        self._cond_threshold = cond_threshold
        self.robot = robot
        self._min_force = min_force
        self.energy_model_type = energy_model.lower()
        
        # Check if we have rod elements (stretch_bend_twist_springs) or shell elements
        # Shell elements can be either triangle_springs (if use_mid_edge=True) or hinge_springs (if use_mid_edge=False)
        has_rod_elements = hasattr(robot, 'stretch_bend_twist_springs') and len(robot.stretch_bend_twist_springs) > 0
        
        # Check for shell elements: either triangle_springs or hinge_springs, or check face_nodes_shell
        has_triangle_springs = hasattr(robot, 'triangle_springs') and len(robot.triangle_springs) > 0
        has_hinge_springs = hasattr(robot, 'hinge_springs') and len(robot.hinge_springs) > 0
        
        # Also check face_nodes_shell as a fallback (this is always set if shell elements exist)
        has_face_nodes = hasattr(robot, 'face_nodes_shell') and robot.face_nodes_shell.size > 0
        
        has_shell_elements = has_triangle_springs or has_hinge_springs or has_face_nodes
        
        # Only compute rod energy model parameters if we have rod elements
        if has_rod_elements:
            # Compute geometry parameters for energy model calculations
            # Access geometry parameters (h = rod_r0, w computed from ixs2 or axs)
            geom = robot._SoftRobot__geom
            h = float(robot.rod_r0[0]) if hasattr(robot.rod_r0, '__len__') else float(robot.rod_r0)
            
            # Compute width w from geometry
            # From ixs2 = h * w^3 / 12, we get w = (12 * ixs2 / h)^(1/3)
            # Or from axs = w * h, we get w = axs / h
            if geom.axs is not None:
                w = float(geom.axs) / h
            elif geom.ixs2 is not None:
                w = (12.0 * float(geom.ixs2) / h) ** (1.0 / 3.0)
            else:
                raise ValueError(
                    "Cannot compute width w from geometry. "
                    "Either 'axs' (cross-sectional area) or 'ixs2' (second moment of area) "
                    "must be provided in GeomParams to compute energy model parameters."
                )
            
            # Compute energy model parameters from geometry if not provided
            if self.energy_model_type == 'sano':
                if sano_zeta is None:
                    # sano_zeta = ((1 - 0.5) * w^4 / (60 * h^2))^0.5
                    # = (0.5 * w^4 / (60 * h^2))^0.5
                    # = (w^4 / (120 * h^2))^0.5
                    sano_zeta = np.sqrt(w**4 / (120.0 * h**2))
                self.energy_model = AnalyticalSanosElasticEnergy(
                    EA=robot.EA,
                    EI1=robot.EI1,
                    EI2=robot.EI2,
                    GJ=robot.GJ,
                    delta_l=robot.ref_len[1],
                    zeta=sano_zeta,
                    h=robot.rod_r0)
                print(f"Using Sano energy model (zeta={sano_zeta:.6f}, computed from geometry: w={w:.6f}, h={h:.6f})")
            elif self.energy_model_type == 'wunderlich':
                if wunderlich_W is None:
                    # Wunderlich_W = w
                    wunderlich_W = w
                self.energy_model = AnalyticalWunderlichElasticEnergy(
                    EA=robot.EA,
                    EI1=robot.EI1,
                    EI2=robot.EI2,
                    GJ=robot.GJ,
                    delta_l=robot.ref_len[1],
                    W=wunderlich_W,
                    h=robot.rod_r0)
                print(f"Using Wunderlich energy model (W={wunderlich_W:.6f}, computed from geometry: w={w:.6f}, h={h:.6f})")
            elif self.energy_model_type == 'kirchhoff':
                self.energy_model = AnalyticalKirchhoffElasticEnergy(
                    EA=robot.EA,
                    EI1=robot.EI1,
                    EI2=robot.EI2,
                    GJ=robot.GJ,
                    delta_l=robot.ref_len[1],
                    h=robot.rod_r0)
                print(f"Using Kirchhoff energy model (computed from geometry: w={w:.6f}, h={h:.6f})")
            elif self.energy_model_type == 'sadowsky':
                self.energy_model = AnalyticalSadowskyElasticEnergy(
                    EA=robot.EA,
                    EI1=robot.EI1,
                    EI2=robot.EI2,
                    GJ=robot.GJ,
                    delta_l=robot.ref_len[1],
                    h=robot.rod_r0)
                print(f"Using Sadowsky energy model (computed from geometry: w={w:.6f}, h={h:.6f})")
            elif self.energy_model_type == 'audoly':
                # Poisson ratio will be computed from EI1 and GJ if not provided
                # nu = 2*EI1/GJ - 1 (for rectangular cross-section where J = 4*I1)
                self.energy_model = AnalyticalAudolyElasticEnergy(
                    EA=robot.EA,
                    EI1=robot.EI1,
                    EI2=robot.EI2,
                    GJ=robot.GJ,
                    delta_l=robot.ref_len[1],
                    w=w,
                    h=robot.rod_r0,
                    nu=None)  # Will be computed from EI1 and GJ
                nu_used = self.energy_model.nu
                print(f"Using Audoly energy model (computed from geometry: w={w:.6f}, h={h:.6f}, nu={nu_used:.6f})")
            else:
                raise ValueError(f"Unknown energy_model: {energy_model}. Must be 'sano', 'wunderlich', 'kirchhoff', 'sadowsky', or 'audoly'")
        else:
            # No rod elements - energy model not needed
            self.energy_model = None

        # Initialize elastics
        self.__elastic_energies: typing.List[ElasticEnergy] = []
        
        # Debug: Print element types detected
        print(f"\n{'='*60}")
        print(f"Energy Model Initialization:")
        print(f"  Rod elements detected: {has_rod_elements}")
        print(f"  Shell elements detected: {has_shell_elements}")
        print(f"    - Triangle springs: {has_triangle_springs} ({len(robot.triangle_springs) if hasattr(robot, 'triangle_springs') else 0})")
        print(f"    - Hinge springs: {has_hinge_springs} ({len(robot.hinge_springs) if hasattr(robot, 'hinge_springs') else 0})")
        print(f"    - Face nodes (face_nodes_shell): {has_face_nodes} ({robot.face_nodes_shell.shape[0] if hasattr(robot, 'face_nodes_shell') and robot.face_nodes_shell.size > 0 else 0})")
        print(f"  Energy model type: {self.energy_model_type}")
        print(f"  use_mid_edge: {robot.sim_params.use_mid_edge if hasattr(robot, 'sim_params') and hasattr(robot.sim_params, 'use_mid_edge') else 'N/A'}")
        print(f"{'='*60}")
        
        # Stretch energy for shell elements only (not rod elements)
        # robot.stretch_springs contains both rod and shell springs: [rod_springs..., shell_springs...]
        # We need to filter to only use shell stretch springs for shell-only simulations
        if has_shell_elements and robot.stretch_springs:
            # Get number of rod stretch springs
            # rod_stretch_springs are the first n_rod_stretch springs in stretch_springs
            # We can infer this from stretch_bend_twist_springs if they exist, or assume 0 if no rod elements
            if has_rod_elements:
                # If we have rod elements, count rod edges to estimate rod stretch springs
                # For now, we'll use a simpler approach: if no rod elements, all stretch_springs are shell
                n_rod_stretch = 0  # Will be updated if we can determine it
                # Try to get from stretch_bend_twist_springs count (each rod has stretch)
                if hasattr(robot, 'stretch_bend_twist_springs') and len(robot.stretch_bend_twist_springs) > 0:
                    n_rod_stretch = len(robot.stretch_bend_twist_springs)
            else:
                # No rod elements, so all stretch_springs are shell
                n_rod_stretch = 0
            
            # Only use shell stretch springs (skip the first n_rod_stretch springs which are rod springs)
            shell_stretch_springs = robot.stretch_springs[n_rod_stretch:]
            
            if len(shell_stretch_springs) > 0:
                self.__elastic_energies.append(
                    StretchEnergy(shell_stretch_springs, robot.state))
                print(f"  ✓ Added StretchEnergy (shell): {len(shell_stretch_springs)} shell stretch springs")
            else:
                print(f"  - Skipped StretchEnergy (shell): No shell stretch springs found")
        else:
            if not has_shell_elements:
                print(f"  - Skipped StretchEnergy (shell): No shell elements detected")
            elif not robot.stretch_springs:
                print(f"  - Skipped StretchEnergy (shell): No stretch springs available")
        
        # Hinge energy for shell elements (hinge_springs is already shell-only)
        # Only add if hinge_springs exist (use_mid_edge=False case)
        if has_hinge_springs:
            self.__elastic_energies.append(
                HingeEnergy(robot.hinge_springs, robot.state))
            print(f"  ✓ Added HingeEnergy (shell): {len(robot.hinge_springs)} hinge springs")
        else:
            print(f"  - Skipped HingeEnergy (shell): No hinge springs available (use_mid_edge={robot.sim_params.use_mid_edge if hasattr(robot, 'sim_params') and hasattr(robot.sim_params, 'use_mid_edge') else 'N/A'})")
        
        # Triangle springs for shell elements
        # Only add if triangle_springs exist (use_mid_edge=True case)
        if has_triangle_springs:
            self.__elastic_energies.append(
                TriangleEnergy(robot.triangle_springs, robot.state))
            print(f"  ✓ Added TriangleEnergy (shell): {len(robot.triangle_springs)} triangle springs")
        else:
            print(f"  - Skipped TriangleEnergy (shell): No triangle springs available (use_mid_edge={robot.sim_params.use_mid_edge if hasattr(robot, 'sim_params') and hasattr(robot.sim_params, 'use_mid_edge') else 'N/A'})")
        
        # if robot.bend_twist_springs:
        #     self.__elastic_energies.append(
        #         BendEnergy(robot.bend_twist_springs, robot.state))
        #     if not robot.sim_params.two_d_sim:   # if 3d
        #         self.__elastic_energies.append(
        #             TwistEnergy(robot.bend_twist_springs, robot.state))

        # Stretch, bend, twist springs - select appropriate elastic energy class (for rod elements only)
        if has_rod_elements:
            if self.energy_model_type == 'sano':
                self.__elastic_energies.append(
                    GeneralElasticEnergySano(robot.stretch_bend_twist_springs, robot.state, self.energy_model))
                print(f"  ✓ Added GeneralElasticEnergySano (rod): {len(robot.stretch_bend_twist_springs)} rod springs")
            elif self.energy_model_type == 'wunderlich':
                self.__elastic_energies.append(
                    GeneralElasticEnergyWunderlich(robot.stretch_bend_twist_springs, robot.state, self.energy_model))
                print(f"  ✓ Added GeneralElasticEnergyWunderlich (rod): {len(robot.stretch_bend_twist_springs)} rod springs")
            elif self.energy_model_type == 'kirchhoff':
                self.__elastic_energies.append(
                    GeneralElasticEnergyKirchhoff(robot.stretch_bend_twist_springs, robot.state, self.energy_model))
                print(f"  ✓ Added GeneralElasticEnergyKirchhoff (rod): {len(robot.stretch_bend_twist_springs)} rod springs")
            elif self.energy_model_type == 'sadowsky':
                self.__elastic_energies.append(
                    GeneralElasticEnergySadowsky(robot.stretch_bend_twist_springs, robot.state, self.energy_model))
                print(f"  ✓ Added GeneralElasticEnergySadowsky (rod): {len(robot.stretch_bend_twist_springs)} rod springs")
            elif self.energy_model_type == 'audoly':
                self.__elastic_energies.append(
                    GeneralElasticEnergyAudoly(robot.stretch_bend_twist_springs, robot.state, self.energy_model))
                print(f"  ✓ Added GeneralElasticEnergyAudoly (rod): {len(robot.stretch_bend_twist_springs)} rod springs")
        else:
            print(f"  - Skipped rod energy models: No rod elements detected")
        
        print(f"{'='*60}")
        print(f"Total elastic energies initialized: {len(self.__elastic_energies)}")
        print(f"{'='*60}\n")

        # Set solver
        # Use RobustSolver by default to handle singular matrices gracefully
        # Users can override by setting robot.sim_params.solver to 'np' or 'pardiso'
        solver_type = getattr(robot.sim_params, 'solver', 'robust')
        base_solver_class = _SOLVERS.get(solver_type, RobustSolver)
        
        # Enable verbose mode to see regularization warnings
        verbose_solver = getattr(robot.sim_params, 'verbose_solver', False)
        
        # Build RobustSolver kwargs
        robust_kwargs = {'verbose': verbose_solver}
        if self._cond_threshold is not None:
            robust_kwargs['cond_threshold'] = self._cond_threshold
        
        if solver_type == 'robust' or base_solver_class == RobustSolver:
            # Create RobustSolver wrapping NumpySolver as base
            base_solver = NumpySolver()
            self._solver = RobustSolver(base_solver=base_solver, **robust_kwargs)
        else:
            # Use the specified solver (np or pardiso) wrapped in RobustSolver for safety
            base_solver = base_solver_class()
            self._solver = RobustSolver(base_solver=base_solver, **robust_kwargs)

        # Simulate callbacks
        self.before_step = None
        
        # Adaptive time-stepping parameters
        self.adaptive_dt = False  # Enable/disable adaptive time-stepping
        self.max_dq_threshold = None  # Maximum allowed displacement (if None, use dtol-based)
        self.dt_reduction_factor = 0.5  # Factor to reduce dt when dq is too large
        self.dt_increase_factor = 1.2  # Factor to increase dt when stable (optional)
        self.min_dt = None  # Minimum allowed dt
        self.max_dt = None  # Maximum allowed dt
        self.max_dt_reductions = 5  # Maximum number of dt reductions per step

        # Force tracking at specified nodes
        self.nodes_to_track_forces = None  # List of node indices to track forces for
        self.tracked_forces_history = []  # List of tracked forces at tracked timesteps
        self.tracked_forces_times = []  # List of times corresponding to tracked forces
        
        # Condition number tracking (optional)
        self.track_condition_number = False  # Enable/disable condition number tracking
        self.condition_numbers = []  # List of condition numbers at each logged timestep
        self.condition_number_times = []  # List of times corresponding to condition numbers

        # Total elastic energy tracking (optional)
        self.track_elastic_energy = False
        self.elastic_energies = []  # List of total elastic energy (scalar) at each logged timestep
        self.elastic_energy_times = []  # List of times corresponding to elastic energies

        # Material directors tracking (optional): a1, a2, m1, m2 per edge at each logged timestep
        self.track_material_directors = False
        self.material_directors_list = []  # List of dicts: {'a1': ndarray, 'a2': ..., 'm1': ..., 'm2': ...}
        self.material_directors_times = []  # List of times corresponding to material directors

    def set_nodes_to_track_forces(self, nodes: typing.List[int] | np.ndarray):
        """
        Set which nodes to track forces for.
        
        Args:
            nodes: List or array of node indices to track forces
        """
        self.nodes_to_track_forces = np.asarray(nodes)
        self.tracked_forces_history = []
        self.tracked_forces_times = []
    
    def enable_condition_number_tracking(self):
        """Enable tracking of Hessian condition number during simulation."""
        self.track_condition_number = True
        self.condition_numbers = []
        self.condition_number_times = []

    def enable_elastic_energy_tracking(self):
        """Enable tracking of total elastic energy at each logged timestep."""
        self.track_elastic_energy = True
        self.elastic_energies = []
        self.elastic_energy_times = []

    def enable_material_director_tracking(self):
        """Enable tracking of material directors (a1, a2, m1, m2) at each logged timestep."""
        self.track_material_directors = True
        self.material_directors_list = []
        self.material_directors_times = []

    def refresh_rod_params(self, robot: SoftRobot):
        """
        Refresh rod elastic energy model params from robot's springs.
        Call after robot.update_rod_geometry() or robot.update_rod_material() for homotopy.
        """
        if not hasattr(robot, 'stretch_bend_twist_springs') or len(robot.stretch_bend_twist_springs) == 0:
            return
        springs = robot.stretch_bend_twist_springs
        sano_zeta = None
        if self.energy_model_type == 'sano':
            h = float(robot.rod_r0[0]) if hasattr(robot.rod_r0, '__len__') else float(robot.rod_r0)
            geom = robot.geom
            if geom.axs is not None and h > 0:
                w = float(geom.axs) / h
            elif geom.ixs2 is not None and h > 0:
                w = (12.0 * float(geom.ixs2) / h) ** (1.0 / 3.0)
            else:
                w = 0.0
            if w > 0 and h > 0:
                sano_zeta = np.sqrt(w**4 / (120.0 * h**2))
        for energy in self.__elastic_energies:
            if hasattr(energy, 'refresh_params_from_springs'):
                energy.refresh_params_from_springs(springs, sano_zeta=sano_zeta)

    def _compute_tracked_forces(self, robot: SoftRobot, nodes: np.ndarray) -> typing.Dict[str, typing.Dict]:
        """
        Compute elastic forces at specified nodes (excluding external and inertial forces).
        
        This method recomputes only elastic forces at the current robot state.
        External forces (gravity, aerodynamics, damping) and inertial forces are excluded.
        
        Args:
            robot: Current robot state
            nodes: Array of node indices to compute forces for
            
        Returns:
            Dictionary with keys:
            - 'node_forces': Dict mapping node_idx to [Fx, Fy, Fz] forces
            - 'edge_torques': Dict mapping edge_idx to Mθ torque for edges connected to tracked nodes
        """
        # Recompute only elastic forces at the current robot state
        # Reuse robot.state quantities (a1, a2, m1, m2, ref_twist) which are already 
        # computed and up-to-date in _finalize_update after step()
        # Only need to recompute tau since it's not updated in _finalize_update
        n_dof = robot.state.q.shape[0]
        q = robot.state.q
        
        # Initialize force vector (only for elastic forces)
        elastic_forces = np.zeros(n_dof)
        
        # Recompute tau (needed for elastic energy, but not updated in _finalize_update)
        tau = robot.update_pre_comp_shell(q)
        
        # Create state with updated tau, reusing other quantities from robot.state
        # This avoids expensive recomputation of a1, a2, m1, m2, ref_twist
        state_with_tau = RobotState.init(
            q, robot.state.a1, robot.state.a2, robot.state.m1, robot.state.m2, 
            robot.state.ref_twist, tau)
        
        # Compute only elastic forces (no external forces, no inertial forces)
        for energy in self.__elastic_energies:
            F, _ = energy.grad_hess_energy_linear_elastic(
                state_with_tau, robot.sim_params.sparse)
            elastic_forces -= F  # Negative gradient is force
        
        # Get fixed DOFs
        fixed_dofs = robot.fixed_dof
        
        # Extract elastic forces at fixed DOFs
        forces_at_fixed_dofs = elastic_forces[fixed_dofs]
        
        # Map DOF indices back to fixed_dofs
        fixed_dof_to_force = {dof: force for dof, force in zip(fixed_dofs, forces_at_fixed_dofs)}
        
        # Organize by nodes
        node_forces_dict = {}
        edge_torques_dict = {}
        
        # Get number of nodes from node_dof_indices property
        n_nodes = len(robot.node_dof_indices)
        
        for node_idx in nodes:
            # Get DOF indices for this node (x, y, z)
            node_dofs = robot.map_node_to_dof(node_idx)
            
            # Extract forces at node DOFs if they are fixed
            node_force = np.zeros(3)
            for i, dof in enumerate(node_dofs):
                if dof in fixed_dof_to_force:
                    node_force[i] = fixed_dof_to_force[dof]
            
            node_forces_dict[node_idx] = node_force
            
            # Find edges connected to this node (edges that have this node as start or end)
            connected_edges = np.where((robot.edges[:, 0] == node_idx) | (robot.edges[:, 1] == node_idx))[0]
            
            # For edges, we track torques at edge DOFs
            # Edge DOFs start after all node DOFs: edge_dof = 3*n_nodes + edge_idx
            for edge_idx in connected_edges:
                edge_dof = robot.map_edge_to_dof(edge_idx)
                if edge_dof < len(robot.state.q) and edge_dof in fixed_dof_to_force:
                    if edge_idx not in edge_torques_dict:
                        edge_torques_dict[edge_idx] = fixed_dof_to_force[edge_dof]
        
        return {
            'node_forces': node_forces_dict,
            'edge_torques': edge_torques_dict
        }
    
    def simulate(self, robot: SoftRobot = None, viz: Visualizer = None):
        """
        Simulate the robot dynamics.

        Returns:
            robots: List of robot states at logged timesteps
            tracked_forces_list: List of tracked forces (if tracking enabled)
            tracked_forces_times_list: List of times for tracked forces
            condition_numbers: List of condition numbers (if tracking enabled, else empty list)
            condition_number_times: List of times for condition numbers
            elastic_energies: List of total elastic energy scalars (if tracking enabled, else empty list)
            elastic_energy_times: List of times for elastic energies
            material_directors_list: List of dicts with keys 'a1','a2','m1','m2' (if tracking enabled, else empty list)
            material_directors_times: List of times for material directors
        """
        robot = robot or self.robot
        
        # Initialize adaptive time-stepping parameters
        original_dt = robot.sim_params.dt
        if self.min_dt is None:
            self.min_dt = original_dt / 100.0  # Allow dt to go down to 1/100 of original
        if self.max_dt is None:
            self.max_dt = original_dt * 2.0  # Allow dt to go up to 2x original
        if self.max_dq_threshold is None:
            # Use dtol-based threshold: max_dq/dt < dtol, so max_dq < dt * dtol
            self.max_dq_threshold = original_dt * robot.sim_params.dtol

        ret = []
        tracked_forces_list = []
        tracked_forces_times_list = []
        
        if viz is not None:
            viz.update(robot, 0)

        # Track time with variable dt
        current_time = 0.0
        step_count = 0
        dt_reduction_stats = []  # Track dt reductions for statistics
        dt_used_for_step = None  # Track the dt actually used for the current step

        # Progress bar showing time progress (not step count, since dt is variable)
        pbar = tqdm(total=robot.sim_params.total_time, 
                    desc="Simulating", 
                    unit="s",
                    bar_format='{l_bar}{bar}| {n:.3f}/{total:.3f}s [{elapsed}<{remaining}, {rate_fmt}]')

        start_time = time.time()

        while current_time < robot.sim_params.total_time:
            step_count += 1
            step_start_time = current_time
            
            # Apply boundary conditions before step
            if self.before_step is not None:
                robot = self.before_step(robot, current_time)

            # Adaptive time-stepping: try step, check displacement, retry if needed
            dt_reductions_this_step = 0
            step_successful = False
            robot_backup = copy.deepcopy(robot)
            initial_dq = None  # Store initial_dq from the step
            
            while not step_successful and dt_reductions_this_step < self.max_dt_reductions:
                try:
                    robot_step, max_dq, initial_dq = self.step(robot)
                    
                    # Check if displacement is acceptable
                    if self.adaptive_dt and max_dq > self.max_dq_threshold:
                        # Displacement too large, reduce dt and retry
                        old_dt = robot.sim_params.dt
                        new_dt = max(self.min_dt, robot.sim_params.dt * self.dt_reduction_factor)
                        
                        if new_dt >= old_dt * 0.99:  # Check if we've hit minimum
                            # Can't reduce further, accept this step
                            robot = robot_step
                            step_successful = True
                            if dt_reductions_this_step > 0:
                                dt_reduction_stats.append(dt_reductions_this_step)
                        else:
                            # Reduce dt and retry
                            robot.sim_params.dt = new_dt
                            robot = robot_backup  # Restore state before step
                            dt_reductions_this_step += 1
                            if step_count == 1:  # Only print on first step to avoid spam
                                print(f"\nStep {step_count}: Reducing dt from {old_dt:.6f} to {new_dt:.6f} (max_dq={max_dq:.6e} > threshold={self.max_dq_threshold:.6e})")
                    else:
                        # Displacement acceptable, accept step
                        robot = robot_step
                        step_successful = True
                        if dt_reductions_this_step > 0:
                            dt_reduction_stats.append(dt_reductions_this_step)
                        # Reset regularization after successful step
                        if hasattr(self._solver, 'reset_regularization'):
                            self._solver.reset_regularization()
                    
                except RuntimeError as e:
                    # Check if this is a convergence failure
                    is_convergence_failure = "Iteration limit" in str(e) or "reached before convergence" in str(e)
                    
                    if is_convergence_failure and hasattr(self._solver, 'increase_regularization'):
                        # Try increasing regularization strength and retry
                        if dt_reductions_this_step == 0:
                            # First convergence failure: try stronger regularization
                            self._solver.increase_regularization(factor=10.0)
                            robot = robot_backup  # Restore state
                            print(f"\nStep {step_count}: Convergence failure detected. Trying with stronger regularization...")
                            dt_reductions_this_step += 1
                            continue  # Retry step with stronger regularization
                        elif dt_reductions_this_step == 1:
                            # Still failing, try even stronger regularization
                            self._solver.increase_regularization(factor=100.0)
                            robot = robot_backup
                            print(f"\nStep {step_count}: Still failing. Trying with maximum regularization...")
                            dt_reductions_this_step += 1
                            continue
                    
                    if self.adaptive_dt and dt_reductions_this_step < self.max_dt_reductions - 1:
                        # Convergence failure, try reducing dt
                        old_dt = robot.sim_params.dt
                        new_dt = max(self.min_dt, robot.sim_params.dt * self.dt_reduction_factor)
                        
                        if new_dt < old_dt * 0.99:
                            robot.sim_params.dt = new_dt
                            robot = robot_backup
                            dt_reductions_this_step += 1
                            print(f"\nStep {step_count}: Convergence failure, reducing dt from {old_dt:.6f} to {new_dt:.6f}")
                            # Reset regularization for next attempt
                            if hasattr(self._solver, 'reset_regularization'):
                                self._solver.reset_regularization()
                        else:
                            # Can't reduce further, stop simulation and return partial results
                            print(f"\n\n{'='*60}")
                            print(f"SIMULATION STOPPED: Convergence failure at step {step_count}")
                            print(f"Time: {current_time:.3f}s / {robot.sim_params.total_time:.3f}s")
                            print(f"Error: {e}")
                            print(f"Cannot reduce dt further (min_dt={self.min_dt:.6e} reached)")
                            print(f"{'='*60}")
                            print(f"Returning {len(ret)} timesteps successfully computed so far.")
                            print(f"{'='*60}\n")
                            step_successful = False
                            break
                    else:
                        # Not using adaptive dt or max reductions reached, stop and return partial results
                        print(f"\n\n{'='*60}")
                        print(f"SIMULATION STOPPED: Convergence failure at step {step_count}")
                        print(f"Time: {current_time:.3f}s / {robot.sim_params.total_time:.3f}s")
                        print(f"Error: {e}")
                        print(f"{'='*60}")
                        print(f"Returning {len(ret)} timesteps successfully computed so far.")
                        print(f"{'='*60}\n")
                        step_successful = False
                        break
            
            if not step_successful:
                break  # Exit simulation loop

            # Store the dt that was actually used for this step BEFORE it might be changed
            # This ensures forces are tracked with the correct time step
            dt_used_for_step = robot.sim_params.dt
            
            # Advance time by the dt that was used for this step
            current_time += dt_used_for_step

            # Increase dt if initial dq from this step was very small (indicates stability)
            # We check the initial_dq from the step we just completed to decide dt for future steps
            if self.adaptive_dt and initial_dq is not None:
                # Increase dt if initial_dq is significantly below threshold (e.g., < 10% of threshold)
                # This helps regain efficiency when simulation becomes stable
                if initial_dq < self.max_dq_threshold * 0.1:
                    old_dt = robot.sim_params.dt
                    new_dt = min(self.max_dt, robot.sim_params.dt * self.dt_increase_factor)
                    
                    # Only increase if we can actually go higher (avoid noise from tiny differences)
                    if new_dt > old_dt * 1.01:
                        robot.sim_params.dt = new_dt
                        # Optionally print when dt increases (but less frequently to avoid spam)
                        if step_count % 100 == 0:  # Only print every 100 steps
                            print(f"\nStep {step_count}: Increasing dt from {old_dt:.6f} to {new_dt:.6f} (initial_dq={initial_dq:.6e} < {self.max_dq_threshold * 0.1:.6e})")

            if viz is not None and step_count % robot.sim_params.plot_step == 0:
                viz.update(robot, current_time)

            if robot.sim_params.log_data and step_count % robot.sim_params.log_step == 0:
                # Store robot state AFTER the step completes
                ret.append(robot)
                
                # Compute and store tracked forces if nodes are being tracked
                # IMPORTANT: Forces are computed from the same robot state that was just saved in ret
                # The time stored corresponds to current_time, which is the time AFTER the step
                # This ensures forces and robot states are synchronized at the same timesteps
                if self.nodes_to_track_forces is not None and len(self.nodes_to_track_forces) > 0:
                    # Compute forces from the current robot state (same state as saved in ret)
                    tracked_forces = self._compute_tracked_forces(robot, self.nodes_to_track_forces)
                    tracked_forces_list.append(tracked_forces)
                    # Store the time that corresponds to this robot state (after the step)
                    # current_time was advanced by dt_used_for_step, which matches the step that produced this robot state
                    tracked_forces_times_list.append(current_time)
                
                # Track condition number if enabled
                if self.track_condition_number:
                    # Compute condition number from the most recent Jacobian (Hessian)
                    # self._j_free is the free-DOF block of the Hessian from the last Newton iteration
                    if hasattr(self, '_j_free') and self._j_free is not None:
                        try:
                            # Convert to dense if sparse
                            if sp.issparse(self._j_free):
                                J_dense = self._j_free.toarray()
                            else:
                                J_dense = self._j_free
                            
                            # Compute condition number
                            cond = np.linalg.cond(J_dense)
                            self.condition_numbers.append(cond)
                            self.condition_number_times.append(current_time)
                        except (np.linalg.LinAlgError, ValueError):
                            # If condition number can't be computed, store inf
                            self.condition_numbers.append(np.inf)
                            self.condition_number_times.append(current_time)

            # Update progress bar with current time (not step count)
            pbar.n = current_time
            pbar.refresh()
            pbar.set_postfix({
                'dt': f'{robot.sim_params.dt:.6f}s',
                'steps': f'{step_count}'
            })

        pbar.close()
        elapsed = time.time() - start_time
        
        print(f"\nSimulation completed in {elapsed:.2f}s (real time) for {current_time:.2f}s (sim time)")
        print(f"Total steps: {step_count}")
        print(f"Average: {elapsed/step_count:.4f}s per timestep")
        print(f"Final dt: {robot.sim_params.dt:.6f}s (original: {original_dt:.6f}s)")
        if dt_reduction_stats:
            print(f"Steps requiring dt reduction: {len(dt_reduction_stats)}")
            print(f"Average reductions per step: {np.mean(dt_reduction_stats):.2f}")
        print(f"Total timesteps saved: {len(ret)}")

        if self.nodes_to_track_forces is not None:
            print(f"Forces tracked for {len(self.nodes_to_track_forces)} nodes at {len(tracked_forces_list)} timesteps")
            # Verify consistency: forces should be tracked at the same timesteps as robot states
            if len(tracked_forces_list) != len(ret):
                print(f"WARNING: Mismatch between tracked forces ({len(tracked_forces_list)}) and robot states ({len(ret)})")
            if len(tracked_forces_times_list) != len(tracked_forces_list):
                print(f"WARNING: Mismatch between tracked forces times ({len(tracked_forces_times_list)}) and tracked forces ({len(tracked_forces_list)})")
        
        if self.track_condition_number:
            print(f"Condition numbers tracked at {len(self.condition_numbers)} timesteps")
            if len(self.condition_numbers) > 0:
                print(f"  Min condition number: {np.min(self.condition_numbers):.2e}")
                print(f"  Max condition number: {np.max(self.condition_numbers):.2e}")
                print(f"  Mean condition number: {np.mean(self.condition_numbers):.2e}")

        if self.track_elastic_energy:
            print(f"Elastic energy tracked at {len(self.elastic_energies)} timesteps")
            if len(self.elastic_energies) > 0:
                print(f"  Min: {np.min(self.elastic_energies):.6e}, Max: {np.max(self.elastic_energies):.6e}")

        if self.track_material_directors:
            print(f"Material directors tracked at {len(self.material_directors_list)} timesteps")

        return (
            ret,
            tracked_forces_list,
            tracked_forces_times_list,
            self.condition_numbers,
            self.condition_number_times,
            self.elastic_energies if self.track_elastic_energy else [],
            self.elastic_energy_times if self.track_elastic_energy else [],
            self.material_directors_list if self.track_material_directors else [],
            self.material_directors_times if self.track_material_directors else [],
        )

    def step(self, robot: SoftRobot = None, debug: bool = False) -> typing.Tuple[SoftRobot, float, float]:
        """
        Execute one time step.
        
        Returns:
            robot: Updated robot state
            max_dq: Maximum absolute displacement in free DOFs (for adaptive time-stepping)
            initial_dq: Displacement magnitude from first Newton iteration
        """
        robot = robot or self.robot

        # Initialize iteration variables
        q = copy.deepcopy(robot.state.q)
        alpha = 1.0
        iteration = 1
        err_history = []
        solved = False

        # Preallocate matrices
        self._n_dof = robot.state.q.shape[0]
        n_free_dof = robot.state.free_dof.shape[0]

        self._forces = np.empty(self._n_dof)
        self._f_free = np.empty(n_free_dof)
        self._dq_free = np.empty(n_free_dof)

        if not robot.sim_params.sparse:
            self._jacobian = np.empty((self._n_dof, self._n_dof))
            self._j_free = np.empty((n_free_dof, n_free_dof))

        ndof_diag = np.arange(robot.state.q.shape[0])

        max_dq = 0.0  # Track maximum displacement during iteration
        initial_dq = None  # Track displacement from first Newton iteration

        while not solved:
            # Some integrators compute F and J not at q_{n+1} (midpoint)
            q_eval = self._compute_evaluation_position(robot, q)
            u_eval = self._compute_evaluation_velocity(robot, q)

            self._compute_forces_and_jacobian(robot, q_eval, u_eval)

            # Inertial force vs equilibrium
            if not robot.sim_params.static_sim:
                inertial_force, inertial_jacobian = self._compute_inertial_force_and_jacobian(
                    robot, q)
                self._forces += inertial_force

                if robot.sim_params.sparse:
                    self._jacobian += sp.diags(inertial_jacobian, format='csr')
                else:
                    self._jacobian[ndof_diag, ndof_diag] += inertial_jacobian

            # Handle free DOF components
            self._f_free[:] = self._forces[robot.state.free_dof]
            if robot.sim_params.sparse:
                self._j_free = self._jacobian[robot.state.free_dof,
                                              :][:, robot.state.free_dof]
            else:
                self._j_free[:] = self._jacobian[np.ix_(
                    robot.state.free_dof, robot.state.free_dof)]

            # Linear system solver
            if np.linalg.norm(self._f_free) < self._min_force:
                self._dq_free.fill(0.0)
            else:
                self._dq_free[:] = self._solver.solve(
                    self._j_free, self._f_free)

            # Adaptive damping and update
            self._dq_free *= self._adaptive_damping(alpha, iteration)
            dq_magnitude = np.max(np.abs(self._dq_free))
            max_dq = max(max_dq, dq_magnitude)  # Track max displacement
            if initial_dq is None:  # Store displacement from first iteration
                initial_dq = dq_magnitude
            q[robot.state.free_dof] -= self._dq_free

            # Error and convergence
            err = np.linalg.norm(self._f_free)
            err_history.append(err)

            solved = self._converged(
                err, err_history, self._dq_free, iteration, robot)

            if debug:
                print("iter: {}, error: {:.3f}".format(iteration, err))
            iteration += 1

        if iteration >= robot.sim_params.max_iter:
            raise RuntimeError(
                f"Iteration limit {robot.sim_params.max_iter} reached before convergence. "
                f"Final error: {err:.3e}"
            )

        # Final update and return
        self.robot = self._finalize_update(robot, q)
        # If no iterations occurred (shouldn't happen), set initial_dq to max_dq
        if initial_dq is None:
            initial_dq = max_dq
        return self.robot, max_dq, initial_dq

    @abc.abstractmethod
    def _compute_inertial_force_and_jacobian(self, robot: SoftRobot, q: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        pass

    def _compute_acceleration(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return np.zeros_like(q)

    def _compute_velocity(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return (q - robot.state.q) / robot.sim_params.dt

    def compute_total_elastic_energy(self, state: RobotState) -> np.ndarray:
        total = 0.0
        for energy in self.__elastic_energies:
            total += energy.get_energy_linear_elastic(state)
        return total

    def _compute_evaluation_position(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return q

    def _compute_evaluation_velocity(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return (q - robot.state.q) / robot.sim_params.dt

    def _compute_forces_and_jacobian(self, robot: SoftRobot, q, u):
        """ Sets self._forces and self._jacobian to sum of external/internal forces """
        self._forces.fill(0.0)

        if robot.sim_params.sparse:
            self._jacobian = sp.csr_matrix(
                (self._n_dof, self._n_dof), dtype=np.float64)
        else:
            self._jacobian.fill(0.0)

        # Compute reference frames and material directors
        a1_iter, a2_iter = robot.compute_time_parallel(
            robot.state.a1, robot.state.q, q)
        m1, m2 = robot.compute_material_directors(q, a1_iter, a2_iter)
        ref_twist = robot.compute_reference_twist(
            robot.bend_twist_springs, q, a1_iter, robot.state.ref_twist)
        tau = robot.update_pre_comp_shell(q)

        new_state = RobotState.init(
            q, a1_iter, a2_iter, m1, m2, ref_twist, tau)

        # Add elastic forces
        for energy in self.__elastic_energies:
            # J is now scipy crc sparse
            F, J = energy.grad_hess_energy_linear_elastic(
                new_state, robot.sim_params.sparse)
            self._forces -= F
            self._jacobian -= J

            # FDM check
            #_,_, fdm_check_grad, fdm_check_hess = energy.fdm_check_grad_hess_strain(new_state)
            #print("gradient of strain passed FDM check:", fdm_check_grad)
            #print("hessian of strain passed FDM check:", fdm_check_hess)

        # Add external forces
        # TODO: Make this also a list
        if "gravity" in robot.env.ext_force_list:
            self._forces -= compute_gravity_forces(robot)
        # ignore for now
        if "aerodynamics" in robot.env.ext_force_list:
            F, J, = compute_aerodynamic_forces_vectorized(robot, q, u)
            self._forces -= F
            self._jacobian -= J  # FIXME: Sparse option
        if "damping" in robot.env.ext_force_list:
            F, J, = compute_damping_force(robot, q, u)
            self._forces -= F
            self._jacobian -= J

    def _converged(self,
                   err: float,
                   err_history: typing.List[float],
                   dq: np.ndarray,
                   iteration: int,
                   robot: SoftRobot):
        """ Check all convergence criteria """
        disp_converged = np.max(np.abs(dq)) / \
            robot.sim_params.dt < robot.sim_params.dtol
        force_converged = err < robot.sim_params.tol
        relative_converged = err < err_history[0] * robot.sim_params.ftol
        iteration_limit = iteration >= robot.sim_params.max_iter

        return any([force_converged, relative_converged, disp_converged, iteration_limit])

    def _adaptive_damping(self, alpha, iteration):
        if iteration < 10:
            return 1.0

        return max(alpha * 0.9, 0.1)

    def _finalize_update(self, robot: SoftRobot, q):
        u = self._compute_velocity(robot, q)
        a = self._compute_acceleration(robot, q)
        a1, a2 = robot.compute_time_parallel(robot.state.a1, robot.state.q, q)
        m1, m2 = robot.compute_material_directors(q, a1, a2)
        ref_twist = robot.compute_reference_twist(
            robot.bend_twist_springs, q, a1, robot.state.ref_twist)
        return robot.update(q=q, u=u, a=a, a1=a1, a2=a2, m1=m1, m2=m2, ref_twist=ref_twist)
