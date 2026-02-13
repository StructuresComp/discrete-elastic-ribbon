import torch
import torch.nn as nn
import numpy as np

class AnalyticalWunderlichElasticEnergy(nn.Module):
    """
    Analytical Wunderlich elastic energy model for a 3D rod.
    
    Energy formula:
    E_k = ½ EA l̄_k ε² + ½ (EI₂/l̄_k) (κ₂)² + 
          ½ (EI₁/l̄_k) [(κ₁(1 + η²))² / (W η')] log[(1 + W η'/2)/(1 - W η'/2)]
    
    where:
    - η = τ/κ₁ (current element)
    - η'_k is computed using finite differences across neighboring elements:
      * Forward diff (k=0): η'_k = l̄_k · [(τ_k+1 - τ_k)κ₁_k - τ_k(κ₁_k+1 - κ₁_k)] / (κ₁_k)²
      * Central diff (middle): η'_k = (l̄_k/2) · [(τ_k+1 - τ_k-1)κ₁_k - τ_k(κ₁_k+1 - κ₁_k-1)] / (κ₁_k)²
      * Backward diff (k=N-3): η'_k = l̄_k · [(τ_k - τ_k-1)κ₁_k - τ_k(κ₁_k - κ₁_k-1)] / (κ₁_k)²
    
    Accepts normalized strain input x = [Δε, Δκ₁·(h/Δl), Δκ₂·(h/Δl), Δτ·(h/Δl)] of shape (batch,4).
    """
    
    def __init__(self,
                 EA: float,
                 EI1: float,
                 EI2: float,
                 GJ: float,  # Not used in Wunderlich but kept for interface compatibility
                 delta_l: float,
                 W: float,  # Wunderlich material parameter
                 h: float):
        super().__init__()
        
        # Physical parameters
        self.EA = float(EA)
        self.EI1 = float(EI1)
        self.EI2 = float(EI2)
        self.GJ = float(GJ)  # Kept for compatibility
        self.delta_l = float(delta_l)
        self.W = float(W)
        self.h = float(h)
        
        # Precompute scale factors
        self.inv_dl = 1.0 / self.delta_l
        self.scaling = self.delta_l / self.h  # For recovering physical strains
        self.energy_norm = 0.5 * self.EA * self.delta_l
        
        # Storage for all element strains (will be updated externally)
        self.all_strains = None  # Shape (N_elements, 4): [ε, κ₁, κ₂, τ] for all elements
        self.current_element_idx = 0  # Index of current element being evaluated
        
        # Small epsilon for numerical stability
        self.eps = 1e-12
        
    def set_all_strains(self, all_strains: np.ndarray, current_idx: int):
        """
        Set all element strains for finite difference calculation.
        
        Args:
            all_strains: Shape (N_elements, 4) - [ε, κ₁, κ₂, τ] for all elements
            current_idx: Index of current element being evaluated
        """
        self.all_strains = all_strains.copy()
        self.current_element_idx = current_idx
        
    def compute_eta_prime(self) -> float:
        """
        Compute η'_k using finite differences across neighboring elements.
        
        Returns:
            η'_k for the current element
        """
        if self.all_strains is None:
            return 0.0
            
        n_elements = self.all_strains.shape[0]
        k = self.current_element_idx
        
        # Extract current element values
        kappa1_k = self.all_strains[k, 1]  # κ₁ at element k
        tau_k = self.all_strains[k, 3]     # τ at element k
        
        if abs(kappa1_k) < self.eps:
            return 0.0
            
        # Compute η' based on position
        if n_elements == 1:
            # Single element case
            return 0.0
        elif k == 0:
            # Forward difference for first element
            if n_elements < 2:
                return 0.0
            kappa1_k1 = self.all_strains[k+1, 1]
            tau_k1 = self.all_strains[k+1, 3]
            
            numerator = (tau_k1 - tau_k) * kappa1_k - tau_k * (kappa1_k1 - kappa1_k)
            eta_prime = self.delta_l * numerator / (kappa1_k**2)
            
        elif k == n_elements - 1:
            # Backward difference for last element
            kappa1_km1 = self.all_strains[k-1, 1]
            tau_km1 = self.all_strains[k-1, 3]
            
            numerator = (tau_k - tau_km1) * kappa1_k - tau_k * (kappa1_k - kappa1_km1)
            eta_prime = self.delta_l * numerator / (kappa1_k**2)
            
        else:
            # Central difference for middle elements
            kappa1_km1 = self.all_strains[k-1, 1]
            kappa1_k1 = self.all_strains[k+1, 1]
            tau_km1 = self.all_strains[k-1, 3]
            tau_k1 = self.all_strains[k+1, 3]
            
            numerator = (tau_k1 - tau_km1) * kappa1_k - tau_k * (kappa1_k1 - kappa1_km1)
            eta_prime = (self.delta_l / 2.0) * numerator / (kappa1_k**2)
            
        return eta_prime
    
    def compute_eta_prime_batch(self, all_strains: np.ndarray) -> np.ndarray:
        """
        Compute η'_k for all elements using finite differences in batch.
        
        Args:
            all_strains: Shape (N_elements, 4) - [ε, κ₁, κ₂, τ] for all elements
            
        Returns:
            eta_prime: Shape (N_elements,) - η' for all elements
        """
        n_elements = all_strains.shape[0]
        eta_prime = np.zeros(n_elements, dtype=np.float64)
        
        if n_elements == 0:
            return eta_prime
        if n_elements == 1:
            return eta_prime
        
        # Extract κ₁ and τ for all elements
        kappa1 = all_strains[:, 1]  # Shape (N,)
        tau = all_strains[:, 3]     # Shape (N,)
        
        # Mask for valid κ₁ (non-zero)
        kappa1_valid = np.abs(kappa1) > self.eps
        if not np.any(kappa1_valid):
            return eta_prime
        
        # Forward difference for first element (k=0)
        if n_elements >= 2:
            kappa1_k = kappa1[0]
            tau_k = tau[0]
            kappa1_k1 = kappa1[1]
            tau_k1 = tau[1]
            
            if kappa1_valid[0]:
                numerator = (tau_k1 - tau_k) * kappa1_k - tau_k * (kappa1_k1 - kappa1_k)
                eta_prime[0] = self.delta_l * numerator / (kappa1_k**2)
        
        # Backward difference for last element (k=N-1)
        if n_elements >= 2:
            kappa1_k = kappa1[-1]
            tau_k = tau[-1]
            kappa1_km1 = kappa1[-2]
            tau_km1 = tau[-2]
            
            if kappa1_valid[-1]:
                numerator = (tau_k - tau_km1) * kappa1_k - tau_k * (kappa1_k - kappa1_km1)
                eta_prime[-1] = self.delta_l * numerator / (kappa1_k**2)
        
        # Central difference for middle elements (k=1 to k=N-2)
        if n_elements >= 3:
            # Vectorized computation for middle elements
            kappa1_mid = kappa1[1:-1]  # Shape (N-2,)
            tau_mid = tau[1:-1]        # Shape (N-2,)
            kappa1_prev = kappa1[:-2]  # Shape (N-2,)
            kappa1_next = kappa1[2:]   # Shape (N-2,)
            tau_prev = tau[:-2]        # Shape (N-2,)
            tau_next = tau[2:]         # Shape (N-2,)
            
            # Valid mask for middle elements
            valid_mid = kappa1_valid[1:-1]
            
            if np.any(valid_mid):
                # Numerator: (τ_{k+1} - τ_{k-1})κ₁_k - τ_k(κ₁_{k+1} - κ₁_{k-1})
                numerator = (tau_next - tau_prev) * kappa1_mid - tau_mid * (kappa1_next - kappa1_prev)
                # Only compute where valid
                eta_prime_mid = np.zeros_like(numerator)
                eta_prime_mid[valid_mid] = (self.delta_l / 2.0) * numerator[valid_mid] / (kappa1_mid[valid_mid]**2)
                eta_prime[1:-1] = eta_prime_mid
        
        return eta_prime
        
    def safe_log_ratio(self, W_eta_prime: torch.Tensor) -> torch.Tensor:
        """
        Safely compute log[(1 + W η'/2)/(1 - W η'/2)] with numerical stability.
        """
        # Clamp W_eta_prime to avoid singularities
        W_eta_prime_clamped = torch.clamp(W_eta_prime, -1.98, 1.98)
        
        numerator = 1.0 + W_eta_prime_clamped / 2.0
        denominator = 1.0 - W_eta_prime_clamped / 2.0
        
        # Ensure positive arguments for log
        numerator = torch.clamp(numerator, min=self.eps)
        denominator = torch.clamp(denominator, min=self.eps)
        
        return torch.log(numerator / denominator)
        
    def forward(self, x: torch.Tensor, eta_prime_batch: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch,4) normalized strains [Δε, Δκ₁·(h/Δl), Δκ₂·(h/Δl), Δτ·(h/Δl)]
        eta_prime_batch: (batch,) optional pre-computed η' values for batch processing
        returns: (batch,1) dimensionless energy density E_nn
        """
        batch_size = x.shape[0]
        
        # Unpack normalized inputs
        eps = x[:, 0]
        k1n = x[:, 1]  # κ₁ normalized
        k2n = x[:, 2]  # κ₂ normalized
        taun = x[:, 3]
        
        # Recover physical strains
        inv_s = self.scaling
        k1 = k1n * inv_s  # κ₁
        k2 = k2n * inv_s  # κ₂
        tau = taun * inv_s  # τ
        
        # Compute current η = τ/κ₁ (vectorized)
        eta = torch.zeros_like(tau)
        k1_nonzero = torch.abs(k1) > self.eps
        eta[k1_nonzero] = tau[k1_nonzero] / k1[k1_nonzero]
        
        # Get η' values - use batch if provided, otherwise compute for current element
        if eta_prime_batch is not None:
            eta_prime_tensor = eta_prime_batch.to(x.dtype).to(x.device)
        else:
            # Fallback to single element computation
            eta_prime = self.compute_eta_prime()
            eta_prime_tensor = torch.tensor(eta_prime, dtype=x.dtype, device=x.device)
            if batch_size > 1:
                # Broadcast single value to batch (for backward compatibility)
                eta_prime_tensor = eta_prime_tensor.expand(batch_size)
        
        W_eta_prime = self.W * eta_prime_tensor
        
        # Energy components
        half = 0.5
        
        # 1. Stretch energy: ½ EA l̄_k ε²
        E_stretch = half * self.EA * self.delta_l * eps**2
        
        # 2. Bending energy: ½ (EI₂/l̄_k) (κ₂)²
        E_bend1 = half * (self.EI2 * self.inv_dl) * k2**2
        
        # 3. Wunderlich nonlinear term: ½ (EI₁/l̄_k) [(κ₁(1 + η²))² / (W η')] log[(1 + W η'/2)/(1 - W η'/2)]
        # Vectorized computation for all batch elements
        eta_prime_valid = torch.abs(eta_prime_tensor) > self.eps
        k1_valid_mask = torch.abs(k1) > self.eps
        
        # Combine conditions: both k1 must be non-zero AND eta_prime must be non-zero
        valid_mask = eta_prime_valid & k1_valid_mask
        
        # Initialize Wunderlich energy
        E_wunderlich = torch.zeros_like(E_stretch)
        
        # Compute for valid elements (vectorized)
        if torch.any(valid_mask):
            k1_valid = k1[valid_mask]
            eta_valid = eta[valid_mask]
            W_eta_prime_valid = W_eta_prime[valid_mask]
            
            # [(κ₁(1 + η²))²]
            factor1 = (k1_valid * (1.0 + eta_valid**2))**2
            
            # 1 / (W η') - avoid division by zero
            factor2 = torch.where(
                torch.abs(W_eta_prime_valid) > self.eps,
                1.0 / W_eta_prime_valid,
                torch.zeros_like(W_eta_prime_valid)
            )
            
            # log[(1 + W η'/2)/(1 - W η'/2)]
            log_term = self.safe_log_ratio(W_eta_prime_valid)
            
            E_wunderlich[valid_mask] = half * (self.EI1 * self.inv_dl) * factor1 * factor2 * log_term
        
        # For non-valid cases (where eta_prime is too small), use simplified term: ½ (EI₁/l̄_k) (κ₁)²
        non_valid_mask = ~valid_mask & k1_valid_mask
        if torch.any(non_valid_mask):
            k1_non_valid = k1[non_valid_mask]
            factor1_simple = k1_non_valid**2
            E_wunderlich[non_valid_mask] = half * (self.EI1 * self.inv_dl) * factor1_simple
        
        # Total energy
        E_full = E_stretch + E_bend1 + E_wunderlich
        
        # Debug output (optional) - only for first element if single element mode
        if batch_size == 1 and self.current_element_idx == 0:
            E_total_scalar = E_full[0].item()
            if E_total_scalar > 1e-12:
                stretch_pct = (E_stretch[0].item() / E_total_scalar) * 100
                bend1_pct = (E_bend1[0].item() / E_total_scalar) * 100
                wunderlich_pct = (E_wunderlich[0].item() / E_total_scalar) * 100
                
                print(f"Wunderlich Energy Contributions (Element {self.current_element_idx}):")
                print(f"  Stretch:     {stretch_pct:6.2f}%")
                print(f"  Bend1 (κ₂):  {bend1_pct:6.2f}%")
                print(f"  Wunderlich:  {wunderlich_pct:6.2f}%")
                print(f"  η' = {eta_prime_tensor[0].item():.6f}")
        
        # Nondimensionalize
        E_nn = E_full / self.energy_norm
        return E_nn.unsqueeze(-1)
        
    def compute_energy_grad_hess(self, x, eta_prime_batch: torch.Tensor = None) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Compute dimensionless energy, gradient, and Hessian w.r.t normalized strains.
        x: numpy array or torch tensor of shape (4,) or (batch, 4)
        eta_prime_batch: optional pre-computed η' values for batch processing
        returns: (E, grad(4,), hess(4,4)) for single input, or batch results
        
        Note: The gradient and Hessian only include derivatives w.r.t. current element strains.
        The η' term is treated as constant (computed from neighboring elements).
        """
        # Handle both single and batch inputs
        is_single = len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1)
        
        # Prepare tensor with grad
        if isinstance(x, torch.Tensor):
            if is_single:
                xt = x.clone().detach().reshape(1, 4).to(torch.float64).requires_grad_(True)
            else:
                xt = x.clone().detach().to(torch.float64).requires_grad_(True)
        else:
            x_np = np.array(x)
            if is_single:
                xt = torch.tensor(x_np.reshape(1, 4), dtype=torch.float64, requires_grad=True)
            else:
                xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        
        E = self.forward(xt, eta_prime_batch).squeeze()
        
        if is_single:
            # Single element case
            # Gradient
            grad = torch.autograd.grad(E, xt, create_graph=True)[0].squeeze()
            
            # Hessian
            hess = torch.zeros((4, 4), dtype=torch.float64)
            for i in range(4):
                gi = torch.autograd.grad(grad[i], xt, retain_graph=True)[0].squeeze()
                hess[i, :] = gi
                
            return E.item(), grad.detach().numpy(), hess.detach().numpy()
        else:
            # Batch case
            batch_size = xt.shape[0]
            # Gradient
            grad = torch.autograd.grad(E.sum(), xt, create_graph=True)[0]  # Shape (batch, 4)
            
            # Hessian for each element in batch
            hess = torch.zeros((batch_size, 4, 4), dtype=torch.float64)
            for i in range(4):
                for j in range(batch_size):
                    gi = torch.autograd.grad(grad[j, i], xt, retain_graph=True)[0][j]
                    hess[j, i, :] = gi
            
            return E.detach().numpy(), grad.detach().numpy(), hess.detach().numpy()
    
    def compute_energy_grad_hess_batch(self, x_batch: torch.Tensor, eta_prime_batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Efficiently compute energy, gradient, and Hessian for a batch of elements.
        
        Args:
            x_batch: (batch, 4) normalized strains
            eta_prime_batch: (batch,) pre-computed η' values for all elements
            
        Returns:
            E: (batch,) energy for each element
            grad: (batch, 4) gradient w.r.t. strains for each element
            hess: (batch, 4, 4) Hessian w.r.t. strains for each element
            grad_eta_prime: (batch,) gradient w.r.t. eta_prime for each element
        """
        batch_size = x_batch.shape[0]
        
        # Convert to torch if needed
        if isinstance(x_batch, np.ndarray):
            xt = torch.tensor(x_batch, dtype=torch.float64, requires_grad=True)
        else:
            xt = x_batch.clone().detach().to(torch.float64).requires_grad_(True)
        
        if isinstance(eta_prime_batch, np.ndarray):
            eta_prime_t = torch.tensor(eta_prime_batch, dtype=torch.float64, requires_grad=True)
        else:
            eta_prime_t = eta_prime_batch.clone().detach().to(torch.float64).requires_grad_(True)
        
        # Compute energy for all elements
        E = self.forward(xt, eta_prime_t).squeeze()  # Shape (batch,)
        
        # Compute gradients for all elements - vectorized approach
        # Create indices for diagonal elements (each element's gradient w.r.t. its own input)
        indices = torch.arange(batch_size, dtype=torch.long)
        
        # Compute all gradients at once
        grad_all = torch.autograd.grad(
            E.sum(),  # Sum to get all gradients
            xt,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]  # Shape: (batch_size, 4)
        
        # Extract diagonal: grad[i] = grad_all[i, :] (gradient of E[i] w.r.t. xt[i])
        grad = grad_all.clone()  # Shape: (batch_size, 4)
        
        # Compute Hessians for all elements - vectorized approach
        hess = torch.zeros((batch_size, 4, 4), dtype=torch.float64)
        # For each element j, compute Hessian row by row
        for i in range(4):
            # Compute gradient of grad[:, i] w.r.t. xt for all elements
            hess_col = torch.autograd.grad(
                grad[:, i].sum(),  # Sum to get all second derivatives
                xt,
                retain_graph=True,
                allow_unused=True
            )[0]  # Shape: (batch_size, 4)
            if hess_col is not None:
                # Extract diagonal: hess[j, i, :] = hess_col[j, :]
                hess[:, i, :] = hess_col
        
        # Compute gradient w.r.t. eta_prime for all elements - vectorized
        grad_eta_prime_all = torch.autograd.grad(
            E.sum(),  # Sum to get all gradients
            eta_prime_t,
            retain_graph=True,
            allow_unused=True
        )[0]  # Shape: (batch_size,)
        grad_eta_prime = grad_eta_prime_all.clone() if grad_eta_prime_all is not None else torch.zeros((batch_size,), dtype=torch.float64)
        
        return E.detach().numpy(), grad.detach().numpy(), hess.detach().numpy(), grad_eta_prime.detach().numpy()