"""Monte-Carlo volumetric radiation transport by ray-marching an unstructured mesh."""

import jax

# Physics, not graphics: float32 leaves only ~7 digits, which the axisymmetric
# crossing spends entirely on a ray passing close to the axis. Set before any
# array is made, so importing raytrax is what fixes the precision.
jax.config.update("jax_enable_x64", True)
