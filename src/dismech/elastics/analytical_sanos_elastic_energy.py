import torch
import torch.nn as nn
import numpy as np

class AnalyticalSanosElasticEnergy(nn.Module):
    """
    Analytical Sano-model elastic-energy for a 3D rod.
    Accepts normalized strain input x = [Δε, Δκ1·(h/Δl), Δκ2·(h/Δl), Δτ·(h/Δl)] of shape (batch,4).
    Computes dimensionless energy density E_nn such that
      E_full = ½ EA Δl (Δε)² + ½ (EI1/Δl)(Δκ1)² + ½ (EI2/Δl)(Δκ2)²
             + ½ (GJ/Δl)(Δτ)² + ½ (GJ/Δl)·(Δτ)⁴/(1/(ζ/Δl)² + (Δκ2)²)
    and E_nn = E_full / (½ EA Δl).

    Provides:
      - forward(x): returns E_nn as tensor (batch,1)
      - compute_energy_grad_hess(x): returns (energy, grad, hess) w.r.t normalized strains
    """
    def __init__(self,
                 EA: float,
                 EI1: float,
                 EI2: float,
                 GJ: float,
                 delta_l: float,
                 zeta: float,
                 h: float):
        super().__init__()
        self.EA = float(EA)
        self.EI1 = float(EI1)
        self.EI2 = float(EI2)
        self.GJ = float(GJ)
        self.delta_l = float(delta_l)
        self.zeta = float(zeta)
        self.h = float(h)

        self.inv_dl = 1.0 / self.delta_l
        self.scaling = self.delta_l / self.h
        self.energy_norm = 0.5 * self.EA * self.delta_l

    def update_params(self, EA=None, EI1=None, EI2=None, GJ=None,
                      delta_l=None, zeta=None, h=None):
        """Update parameters for homotopy (geometry/material changes during simulation)."""
        if EA is not None:
            self.EA = float(EA)
        if EI1 is not None:
            self.EI1 = float(EI1)
        if EI2 is not None:
            self.EI2 = float(EI2)
        if GJ is not None:
            self.GJ = float(GJ)
        if delta_l is not None:
            self.delta_l = float(delta_l)
        if zeta is not None:
            self.zeta = float(zeta)
        if h is not None:
            self.h = float(h)
        self.inv_dl = 1.0 / self.delta_l
        self.scaling = self.delta_l / self.h
        self.energy_norm = 0.5 * self.EA * self.delta_l

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch,4) normalized strains
        returns: (batch,1) dimensionless energy density E_nn
        """
        eps = x[:,0]
        k1n = x[:,1]
        k2n = x[:,2]
        taun = x[:,3]

        inv_s = self.scaling
        k1 = k1n * inv_s
        k2 = k2n * inv_s
        tau = taun * inv_s

        half = 0.5
        E_stretch = half * self.EA * self.delta_l * eps**2
        E_bend1 = half * (self.EI1 * self.inv_dl) * k1**2
        E_bend2 = half * (self.EI2 * self.inv_dl) * k2**2
        E_twist = half * (self.GJ * self.inv_dl) * tau**2
        E_nonlinear = half * (self.EI1 * self.inv_dl) * tau**4 / (1/(self.zeta*self.inv_dl)**2 + k1**2)

        E_full = E_stretch + E_bend1 + E_bend2 + E_twist + E_nonlinear
        E_nn = E_full / self.energy_norm
        return E_nn.unsqueeze(-1)

    def compute_energy_grad_hess(self, x) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Compute dimensionless energy, gradient, and Hessian w.r.t normalized strains.
        x: numpy array or torch tensor of shape (4,)
        returns: (E, grad(4,), hess(4,4))
        """
        if isinstance(x, torch.Tensor):
            xt = x.clone().detach().reshape(1,4).to(torch.float64).requires_grad_(True)
        else:
            xt = torch.tensor(x.reshape(1,4), dtype=torch.float64, requires_grad=True)

        E = self.forward(xt).squeeze()
        grad = torch.autograd.grad(E, xt, create_graph=True)[0].squeeze()
        hess = torch.zeros((4,4), dtype=torch.float64)
        for i in range(4):
            gi = torch.autograd.grad(grad[i], xt, retain_graph=True)[0].squeeze()
            hess[i,:] = gi

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