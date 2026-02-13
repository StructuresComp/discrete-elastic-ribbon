import abc
import warnings

import numpy as np
import scipy.sparse as sp


class Solver(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def solve(self, J: np.ndarray, F: np.ndarray):
        pass


class NumpySolver(Solver):

    def __init__(self, **kwargs):
        pass

    def solve(self, J: np.ndarray, F: np.ndarray):
        if isinstance(J, sp.csr_matrix):
            print("[WARNING] Using numpy (a dense solver) for a sparse matrix")
            J = J.toarray()
        return np.linalg.solve(J, F)


class PardisoSolver(Solver):

    def __init__(self, **kwargs):
        try:
            import pypardiso
        except ImportError:
            raise ImportError("pypardiso is required for PardisoSolver but not installed. Please install it using:\n"
                              "pip install pypardiso"
                              )
        else:
            self.pardiso = pypardiso

    def solve(self, J: np.ndarray | sp.csr_matrix, F: np.ndarray):
        if isinstance(J, np.ndarray):
            print("[WARNING] Using Pardiso (a sparse solver) for a dense matrix")
            J = sp.csr_matrix(J)
        return self.pardiso.spsolve(J, F)


class RobustSolver(Solver):
    """
    Robust solver with adaptive regularization to handle singular/near-singular matrices.
    
    Uses Tikhonov regularization (J + lambda*I) when the matrix is ill-conditioned,
    maintaining accuracy for implicit Euler integration while ensuring stability.
    
    Parameters:
        base_solver: Underlying solver to use (NumpySolver or PardisoSolver)
        cond_threshold: Condition number threshold above which regularization is applied (default: 1e12)
        reg_factor: Regularization factor multiplier (default: 1e-8)
        max_reg_factor: Maximum regularization factor (default: 1e-4)
        verbose: Print warnings when regularization is applied (default: False)
    """
    
    def __init__(self, base_solver=None, cond_threshold=1e12, reg_factor=1e-8, 
                 max_reg_factor=1e-4, verbose=False, **kwargs):
        """
        Initialize robust solver.
        
        Args:
            base_solver: Base solver instance (NumpySolver or PardisoSolver). 
                        If None, creates NumpySolver by default.
            cond_threshold: Condition number threshold for regularization (default: 1e12)
            reg_factor: Base regularization factor (default: 1e-8)
            max_reg_factor: Maximum allowed regularization factor (default: 1e-4)
            verbose: Print diagnostic messages (default: False)
        """
        super().__init__(**kwargs)
        if base_solver is None:
            self.base_solver = NumpySolver()
        else:
            self.base_solver = base_solver
        
        self.cond_threshold = cond_threshold
        self.reg_factor = reg_factor
        self.max_reg_factor = max_reg_factor
        self.verbose = verbose
        self._regularization_count = 0
        self._max_condition_seen = 0.0
        self._original_max_reg_factor = max_reg_factor  # Store original for reset
    
    def solve(self, J: np.ndarray | sp.csr_matrix, F: np.ndarray):
        """
        Solve linear system J * x = F with adaptive regularization.
        
        Args:
            J: Jacobian matrix (dense numpy array or sparse CSR matrix)
            F: Force/residual vector
            
        Returns:
            Solution vector x
        """
        # Convert sparse to dense for condition number check if needed
        J_dense = None
        is_sparse = isinstance(J, sp.csr_matrix)
        
        if is_sparse:
            J_dense = J.toarray()
        else:
            J_dense = J
        
        # Check condition number
        try:
            cond = np.linalg.cond(J_dense)
            self._max_condition_seen = max(self._max_condition_seen, cond)
        except np.linalg.LinAlgError:
            # If condition number can't be computed, assume ill-conditioned
            cond = np.inf
        
        # Apply regularization if needed
        regularization_applied = False
        lambda_reg = 0.0
        
        if cond > self.cond_threshold or not np.isfinite(cond):
            # Compute adaptive regularization factor
            # Scale with condition number but cap at max_reg_factor
            if np.isfinite(cond):
                lambda_reg = min(self.reg_factor * np.sqrt(cond), self.max_reg_factor)
            else:
                lambda_reg = self.max_reg_factor
            
            # Apply Tikhonov regularization: J_reg = J + lambda * I
            if is_sparse:
                n = J.shape[0]
                regularization = sp.identity(n, format='csr') * lambda_reg
                J_reg = J + regularization
            else:
                n = J.shape[0]
                J_reg = J.copy()
                np.fill_diagonal(J_reg, J_reg.diagonal() + lambda_reg)
            
            regularization_applied = True
            self._regularization_count += 1
            
            if self.verbose:
                warnings.warn(
                    f"Matrix ill-conditioned (cond={cond:.2e}). "
                    f"Applying regularization lambda={lambda_reg:.2e}.",
                    RuntimeWarning
                )
        else:
            J_reg = J
        
        # Try to solve with base solver
        try:
            return self.base_solver.solve(J_reg, F)
        except (np.linalg.LinAlgError, ValueError) as e:
            # If still fails, try with stronger regularization
            if not regularization_applied:
                # First attempt failed, try with regularization
                if is_sparse:
                    n = J.shape[0]
                    regularization = sp.identity(n, format='csr') * self.max_reg_factor
                    J_reg = J + regularization
                else:
                    n = J.shape[0]
                    J_reg = J.copy()
                    np.fill_diagonal(J_reg, J_reg.diagonal() + self.max_reg_factor)
                
                if self.verbose:
                    warnings.warn(
                        f"Singular matrix detected. Applying maximum regularization "
                        f"lambda={self.max_reg_factor:.2e}.",
                        RuntimeWarning
                    )
                
                try:
                    return self.base_solver.solve(J_reg, F)
                except (np.linalg.LinAlgError, ValueError) as e2:
                    # Last resort: use pseudo-inverse
                    if self.verbose:
                        warnings.warn(
                            f"Regularized solve failed. Using pseudo-inverse as fallback.",
                            RuntimeWarning
                        )
                    
                    if is_sparse:
                        J_dense = J_reg.toarray()
                    else:
                        J_dense = J_reg
                    
                    # Use SVD-based pseudo-inverse for near-singular matrices
                    try:
                        U, s, Vt = np.linalg.svd(J_dense, full_matrices=False)
                        # Filter out very small singular values
                        s_inv = np.zeros_like(s)
                        tol = np.max(s) * np.finfo(s.dtype).eps * max(J_dense.shape)
                        s_inv[s > tol] = 1.0 / s[s > tol]
                        J_pinv = Vt.T @ np.diag(s_inv) @ U.T
                        return J_pinv @ F
                    except Exception as e3:
                        # Even pseudo-inverse failed, re-raise with context
                        raise RuntimeError(
                            f"All solution methods failed. Original error: {e2}. "
                            f"Pseudo-inverse error: {e3}. Matrix shape: {J_dense.shape}, "
                            f"Condition number: {np.linalg.cond(J_dense) if J_dense.size > 0 else 'N/A'}"
                        ) from e3
            else:
                # Regularization was already applied but still failed
                # Try pseudo-inverse as last resort
                if self.verbose:
                    warnings.warn(
                        f"Regularized solve failed even with lambda={lambda_reg:.2e}. "
                        f"Trying pseudo-inverse as fallback.",
                        RuntimeWarning
                    )
                
                try:
                    if is_sparse:
                        J_dense = J_reg.toarray()
                    else:
                        J_dense = J_reg
                    
                    U, s, Vt = np.linalg.svd(J_dense, full_matrices=False)
                    s_inv = np.zeros_like(s)
                    tol = np.max(s) * np.finfo(s.dtype).eps * max(J_dense.shape)
                    s_inv[s > tol] = 1.0 / s[s > tol]
                    J_pinv = Vt.T @ np.diag(s_inv) @ U.T
                    return J_pinv @ F
                except Exception as e3:
                    # Everything failed, re-raise with full context
                    raise RuntimeError(
                        f"All solution methods failed including pseudo-inverse. "
                        f"Original error: {e}. Regularization lambda={lambda_reg:.2e}. "
                        f"Pseudo-inverse error: {e3}. Matrix shape: {J_dense.shape}"
                    ) from e3
    
    def increase_regularization(self, factor=10.0):
        """
        Temporarily increase maximum regularization factor for difficult cases.
        
        Args:
            factor: Multiplier for max_reg_factor (default: 10.0)
        """
        self.max_reg_factor = min(self._original_max_reg_factor * factor, 1e-2)
        if self.verbose:
            print(f"Increased max regularization to {self.max_reg_factor:.2e}")
    
    def reset_regularization(self):
        """Reset regularization factor to original value."""
        self.max_reg_factor = self._original_max_reg_factor
    
    def get_stats(self):
        """Get statistics about regularization usage."""
        return {
            'regularization_count': self._regularization_count,
            'max_condition_seen': self._max_condition_seen,
            'current_max_reg_factor': self.max_reg_factor
        }
