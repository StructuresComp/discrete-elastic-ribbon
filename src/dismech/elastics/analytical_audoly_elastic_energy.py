import torch
import torch.nn as nn
import numpy as np

class AnalyticalAudolyElasticEnergy(nn.Module):
    """
    Analytical Audoly-model elastic-energy for a 3D rod.
    Accepts normalized strain input x = [Δε, Δκ1·(h/Δl), Δκ2·(h/Δl), Δτ·(h/Δl)] of shape (batch,4).
    Computes dimensionless energy density E_nn such that
      E_full = ½ EA Δl (Δε)² + ½ (EI1/Δl)(Δκ1)² + ½ (EI2/Δl)(Δκ2)²
             + ½ (GJ/Δl)(Δτ)² + (3 EI1 w^4)/(h^2 (ΔL)^3) [ν κ₁² + τ²]² φ(v)
    and E_nn = E_full / (½ EA Δl).

    where:
      v = √(12(1-ν²)) (w²/(h ΔL)) κ₁
      φ(v) = (4/v²) [1/2 - (cosh(√(|v|/2)) - cos(√(|v|/2)))/(√(|v|/2)(sinh(√(|v|/2)) + sin(√(|v|/2))))]
      ν is the Poisson ratio

    Provides:
      - forward(x): returns E_nn as tensor (batch,1)
      - compute_energy_grad_hess(x): returns (energy, grad, hess) w.r.t normalized strains
      - compute_energy_grad_hess_batch(x): batch version
    """
    def __init__(self,
                 EA: float,
                 EI1: float,
                 EI2: float,
                 GJ: float,
                 delta_l: float,
                 w: float,  # Width of the ribbon
                 h: float,  # Thickness of the ribbon
                 nu: float = None):  # Poisson ratio (if None, computed from EI1 and GJ)
        super().__init__()
        self.EA = float(EA)
        self.EI1 = float(EI1)
        self.EI2 = float(EI2)
        self.GJ = float(GJ)
        self.delta_l = float(delta_l)
        self.w = float(w)
        self.h = float(h)
        
        # Compute Poisson ratio if not provided
        if nu is None:
            # From compute_rod_stiffness: GJ = E*J/(2(1+ν))
            # For rectangular cross-section: J = w*h^3/3, I1 = w*h^3/12, so J = 4*I1
            # Therefore: GJ = 4*EI1/(2(1+ν)) = 2*EI1/(1+ν)
            # So: 1+ν = 2*EI1/GJ, therefore: ν = 2*EI1/GJ - 1
            # This matches the user's formula: ν = 2*EI1/G - 1 where G = GJ/J = GJ/(4*I1)
            # But more directly: ν = 2*EI1/GJ - 1 (assuming rectangular cross-section where J = 4*I1)
            self.nu = 2.0 * self.EI1 / self.GJ - 1.0
            # Clamp to reasonable range for physical materials
            self.nu = np.clip(self.nu, -0.99, 0.5)
        else:
            self.nu = float(nu)

        # Precompute scale factors
        self.inv_dl = 1.0 / self.delta_l
        self.scaling = self.delta_l / self.h  # For recovering physical strains
        self.energy_norm = 0.5 * self.EA * self.delta_l
        
        # Precompute constant factors for nonlinear term
        # E_nonlinear = (3 EI1 w^4)/(h^2 (ΔL)^3) [ν κ₁² + τ²]² φ(v)
        self.nonlinear_prefactor = (3.0 * self.EI1 * self.w**4) / (self.h**2 * self.delta_l**3)
        
        # Precompute constant for v: v = √(12(1-ν²)) (w²/(h ΔL)) κ₁
        sqrt_12_one_minus_nu_sq = np.sqrt(12.0 * (1.0 - self.nu**2))
        self.v_prefactor = sqrt_12_one_minus_nu_sq * (self.w**2 / (self.h * self.delta_l))
        
        # Regularization parameter for φ(v) when v → 0
        self.eps = 1e-6

    def compute_phi(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute φ(v) = (4/v²) [1/2 - (cosh(√(|v|/2)) - cos(√(|v|/2)))/(√(|v|/2)(sinh(√(|v|/2)) + sin(√(|v|/2))))]
        
        When v → 0, the limit is 0. For small v, we use a smooth transition to 0 to maintain
        gradient flow while avoiding numerical issues.
        
        Args:
            v: Tensor of shape (batch,) containing v values
            
        Returns:
            φ(v) tensor of shape (batch,)
        """
        v_abs = torch.abs(v)
        
        # Compute √(|v|/2) for all elements - add small epsilon for numerical stability
        # This maintains gradient flow even when v is very small
        v_half_abs = v_abs / 2.0
        # Use a small regularization to avoid sqrt(0) while maintaining gradient flow
        sqrt_v_half = torch.sqrt(v_half_abs + self.eps**2)
        
        # Compute hyperbolic and trigonometric functions for all elements
        cosh_term = torch.cosh(sqrt_v_half)
        cos_term = torch.cos(sqrt_v_half)
        sinh_term = torch.sinh(sqrt_v_half)
        sin_term = torch.sin(sqrt_v_half)
        
        # Compute numerator: cosh(√(|v|/2)) - cos(√(|v|/2))
        numerator = cosh_term - cos_term
        
        # Compute denominator: √(|v|/2)(sinh(√(|v|/2)) + sin(√(|v|/2)))
        # Add small epsilon to avoid division by zero while maintaining gradients
        denominator = sqrt_v_half * (sinh_term + sin_term) + self.eps
        
        # Compute the fraction
        fraction = numerator / denominator
        
        # Compute φ(v) = (4/v²) [1/2 - fraction]
        # Regularize v² to prevent division by zero while maintaining gradient flow
        # For very small v, this smoothly approaches 0 (the limit)
        v_sq_reg = v**2 + self.eps**2
        phi_full = (4.0 / v_sq_reg) * (0.5 - fraction)
        
        # For very small v, use smooth transition to 0 to maintain gradient flow
        # The threshold-based masking ensures numerical stability without breaking gradients
        small_v_threshold = 1e-4
        small_v_mask = v_abs < small_v_threshold
        
        # Use torch.where to smoothly handle small v: returns 0 for small v, phi_full otherwise
        # This maintains gradient flow through the entire computation graph
        phi = torch.where(small_v_mask, torch.zeros_like(phi_full), phi_full)
        
        return phi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch,4) normalized strains
        returns: (batch,1) dimensionless energy density E_nn
        """
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
        
        # Energy components
        half = 0.5
        
        # 1. Stretch energy: ½ EA Δl ε²
        E_stretch = half * self.EA * self.delta_l * eps**2
        
        # 2. Bending energy: ½ (EI1/Δl) κ₁²
        E_bend1 = half * (self.EI1 * self.inv_dl) * k1**2
        
        # 3. Bending energy: ½ (EI2/Δl) κ₂²
        E_bend2 = half * (self.EI2 * self.inv_dl) * k2**2
        
        # 4. Twist energy: ½ (GJ/Δl) τ²
        E_twist = half * (self.GJ * self.inv_dl) * tau**2
        
        # 5. Audoly nonlinear term: (3 EI1 w^4)/(h^2 (ΔL)^3) [ν κ₁² + τ²]² φ(v)
        # where v = √(12(1-ν²)) (w²/(h ΔL)) κ₁
        v = self.v_prefactor * k1  # Shape (batch,)
        phi_v = self.compute_phi(v)  # Shape (batch,)
        
        # Compute [ν κ₁² + τ²]²
        nu_k1_sq_plus_tau_sq = self.nu * k1**2 + tau**2
        nu_k1_sq_plus_tau_sq_sq = nu_k1_sq_plus_tau_sq**2
        
        # Nonlinear energy
        E_nonlinear = self.nonlinear_prefactor * nu_k1_sq_plus_tau_sq_sq * phi_v
        
        # Total energy
        E_full = E_stretch + E_bend1 + E_bend2 + E_twist + E_nonlinear
        
        # Nondimensionalize
        E_nn = E_full / self.energy_norm
        return E_nn.unsqueeze(-1)

    def compute_energy_grad_hess(self, x) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Compute dimensionless energy, gradient, and Hessian w.r.t normalized strains.
        x: numpy array or torch tensor of shape (4,)
        returns: (E, grad(4,), hess(4,4))
        """
        if isinstance(x, torch.Tensor):
            xt = x.clone().detach().reshape(1, 4).to(torch.float64).requires_grad_(True)
        else:
            xt = torch.tensor(x.reshape(1, 4), dtype=torch.float64, requires_grad=True)

        E = self.forward(xt).squeeze()
        grad = torch.autograd.grad(E, xt, create_graph=True)[0].squeeze()
        hess = torch.zeros((4, 4), dtype=torch.float64)
        for i in range(4):
            gi = torch.autograd.grad(grad[i], xt, retain_graph=True)[0].squeeze()
            hess[i, :] = gi

        return E.item(), grad.detach().numpy(), hess.detach().numpy()
    
    def compute_energy_grad_hess_batch(self, x_batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Efficiently compute energy, gradient, and Hessian for a batch of elements.
        
        Args:
            x_batch: (batch, 4) normalized strains
            
        Returns:
            E: (batch,) energy for each element
            grad: (batch, 4) gradient w.r.t. strains for each element
            hess: (batch, 4, 4) Hessian w.r.t. strains for each element
        """
        batch_size = x_batch.shape[0]
        
        if isinstance(x_batch, np.ndarray):
            xt = torch.tensor(x_batch, dtype=torch.float64, requires_grad=True)
        else:
            xt = x_batch.clone().detach().to(torch.float64).requires_grad_(True)
        
        E = self.forward(xt).squeeze()
        
        grad_all = torch.autograd.grad(
            E.sum(),
            xt,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        grad = grad_all.clone()
        
        hess = torch.zeros((batch_size, 4, 4), dtype=torch.float64)
        for i in range(4):
            hess_col = torch.autograd.grad(
                grad[:, i].sum(),
                xt,
                retain_graph=True,
                allow_unused=True
            )[0]
            if hess_col is not None:
                hess[:, i, :] = hess_col
        
        return E.detach().numpy(), grad.detach().numpy(), hess.detach().numpy()

