import jax.numpy as jnp
import jax

class HighDimPIDE:
    def __init__(self, config):
        self.config = config

    def terminal_condition(self, x):
        # g(x) = (1/d) * ||x||^2
        return (1.0 / self.config.dim) * jnp.sum(x**2, axis=-1, keepdims=True)

    def sample_initial_conditions(self, key, batch_size):
        # X0 = 1 (d차원 벡터), t0 = 0
        x0 = jnp.ones((batch_size, self.config.dim))
        t0 = jnp.zeros((batch_size, 1))
        return t0, x0