"""Named jaxtyping aliases for the shapes that recur across the device side."""

from jaxtyping import Array, Bool, Float, Int, Key

# Plain assignments, not PEP 695 `type` statements: jaxtyping has to resolve
# these at runtime under `install_import_hook`.

# Every key in the codebase comes from `jax.random.key`, i.e. a typed key.
KeyArray = Key[Array, ""]

Scalar = Float[Array, ""]
Index = Int[Array, ""]
Flag = Bool[Array, ""]

# A single point or direction in the modelled plane.
Vector = Float[Array, " ndim"]

# One cell's half-spaces, over its padded face slots.
Normals = Float[Array, "nslot ndim"]
Offsets = Float[Array, " nslot"]
