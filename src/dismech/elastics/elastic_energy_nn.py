import torch
import torch.nn as nn
import numpy as np
import os # Import os for path checking

# Define the custom SquareActivation function
class SquareActivation(nn.Module):
    """
    Custom activation function that squares the input.
    f(x) = x^2
    """
    def forward(self, x):
        return x**2

class ElasticEnergyRod3DNN(nn.Module):
    """
    Neural Network model to approximate the elastic energy density of a 3D rod.
    Takes strain measures (stretch, kappa1, kappa2, tau) as input.
    Can be initialized with weights from a checkpoint.
    """
    def __init__(self, input_size=4, hidden_size=16, layers=2, checkpoint_path='/data/shivam/Ribbon/dismech/dismech-python-general/notebooks/checkpoints/elastic_energy_rod3D_DER_model.pth'):
        """
        Initializes the neural network model.

        Args:
            input_size (int): Number of input features (typically 4: stretch, k1, k2, tau).
            hidden_size (int): Number of neurons in each hidden layer.
            layers (int): Number of hidden layers.
            checkpoint_path (str, optional): Path to the model checkpoint file (.pth).
                                            If provided, the model's state_dict will be loaded.
        """
        super(ElasticEnergyRod3DNN, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers

        # Create the network architecture
        network_layers = []
        # Applying SquareActivation as per user's provided code snippet
        network_layers.append(SquareActivation())
        network_layers.append(nn.Linear(input_size, hidden_size, dtype=torch.float64))
        network_layers.append(nn.ReLU())

        for _ in range(layers - 1):
            network_layers.append(nn.Linear(hidden_size, hidden_size, dtype=torch.float64))
            network_layers.append(nn.ReLU())

        network_layers.append(nn.Linear(hidden_size, 1, dtype=torch.float64)) # Output layer (scalar energy)

        self.net = nn.Sequential(*network_layers)

        # --- Add checkpoint loading ---
        if checkpoint_path is not None:
            if os.path.exists(checkpoint_path):
                print(f"Loading model weights from checkpoint: {checkpoint_path}")
                # Load the state dictionary
                state_dict = torch.load(checkpoint_path)
                # Load the state dictionary into the model
                self.load_state_dict(state_dict)
                print("Model weights loaded successfully.")
            else:
                print(f"Warning: Checkpoint file not found at {checkpoint_path}. Model initialized with random weights.")
        # --- End checkpoint loading ---

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.net(x)

    def compute_energy_grad_hess(self, x):
        """
        Compute energy, gradient, and Hessian with respect to the input strain measures.

        Parameters:
            x (numpy.ndarray or torch.Tensor): Input vector (stretch, kappa1, kappa2, tau), shape (4,).

        Returns:
            tuple: (energy, gradient, Hessian), where:
                - energy (float): Energy value at the input.
                - gradient (numpy.ndarray): Gradient vector w.r.t. input strains, shape (4,).
                - Hessian (numpy.ndarray): Hessian matrix w.r.t. input strains, shape (4, 4).
        """
        # Ensure the input is a 2D tensor of shape (1, 4) and requires grad
        if isinstance(x, torch.Tensor):
            # Ensure it's float64 and requires grad
            x_tensor = x.to(torch.float64).clone().detach().reshape(1, -1).requires_grad_(True)
        else:
            x_tensor = torch.tensor(x.reshape(1, -1), requires_grad=True, dtype=torch.float64)

        # Forward pass to compute energy (output is scalar)
        energy = self.forward(x_tensor).squeeze()  # Remove batch dimension

        # Compute gradient (dE/dx)
        # create_graph=True allows computing higher-order derivatives (Hessian)
        grad = torch.autograd.grad(energy, x_tensor, create_graph=True)[0].squeeze()  # Shape: (4,)

        # Compute Hessian (d²E/dx²)
        hess = torch.zeros((4, 4), dtype=torch.float64)  # Initialize Hessian
        for i in range(4):  # Iterate over each dimension of the gradient
            # Compute the gradient of the i-th component of the gradient vector
            # retain_graph=True is needed because we reuse parts of the graph
            grad_i = torch.autograd.grad(grad[i], x_tensor, retain_graph=True)[0].squeeze()  # Shape: (4,)
            hess[i, :] = grad_i  # Fill row i of the Hessian

        # Return results as numpy arrays
        return energy.item(), grad.detach().numpy(), hess.detach().numpy()
