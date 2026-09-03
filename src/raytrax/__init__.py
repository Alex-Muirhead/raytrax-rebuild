"""Monte-Carlo volumetric radiation transport by ray-marching an unstructured mesh."""

import jax

# Physics, not graphics
jax.config.update("jax_enable_x64", True)
