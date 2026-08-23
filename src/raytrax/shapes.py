"""Named jaxtyping aliases for the shapes that recur across the device side.

Spelled out, an override's signature is long enough that nobody repeats it, and
an unannotated override is never checked against its base. Naming them is what
makes repeating them cheap.

Plain assignments rather than PEP 695 `type` statements, so jaxtyping can still
resolve them at runtime under `install_import_hook`.
"""

from jaxtyping import Array, Bool, Float, Int, Key

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
