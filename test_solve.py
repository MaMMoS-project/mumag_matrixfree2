import os
os.environ["JAX_ENABLE_X64"] = "True"
import jax
import jax.numpy as jnp
import numpy as np

# Load a checkpoint or something? No, it's easier to modify poisson_solve.py directly to print!
