# config.py
from dataclasses import dataclass

@dataclass
class PIDEConfig:
    # Training-related
    epochs: int = 20000
    batch_size: int = 128
    learning_rate: float = 1e-4
    lr_milestones: tuple = (7000, 14000, 17000)
    lr_gamma: float = 0.1

    # Problem-related
    dim: int = 100         # Dimension d
    T: float = 1.0         # Terminal time
    n_steps: int = 50      # Number of time steps for discretization

    # PIDE Coefficients
    lambda_: float = 0.01
    tau: float = 0.1
    mu_phi: float = 1.0
    sigma_phi: float = 0.05
    epsilon: float = 0.01
    
    # Model-related
    hidden_dims: tuple = (256, 256, 128, 1)

    @property
    def dt(self) -> float:
        return self.T / self.n_steps