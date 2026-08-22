from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class Absorptivity(eqx.Module):
    """Absorption coefficient per cell.

    The march only ever asks for `coefficient()`, so where the field comes from
    -- a constant, a column read out of the input file, or a model evaluated on
    the temperature -- stays entirely on this side of the boundary.
    """

    def coefficient(self) -> Float[Array, " ncell"]:
        raise NotImplementedError


class Uniform(Absorptivity):
    value: Float[Array, ""]
    num_cells: int = eqx.field(static=True)

    def coefficient(self):
        return jnp.full(self.num_cells, self.value)


class Tabulated(Absorptivity):
    """A field supplied directly, e.g. read from the mesh file."""

    values: Float[Array, " ncell"]

    def coefficient(self):
        return self.values


class FromTemperature(Absorptivity):
    """A field generated from the temperature by any traceable model."""

    temperature: Float[Array, " ncell"]
    model: Callable[[Float[Array, " ncell"]], Float[Array, " ncell"]] = eqx.field(static=True)

    def coefficient(self):
        return self.model(self.temperature)
