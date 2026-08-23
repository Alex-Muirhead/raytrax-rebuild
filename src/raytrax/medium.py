from collections.abc import Callable
from typing import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class Absorptivity(eqx.Module):
    """Absorption coefficient per cell."""

    def coefficient(self) -> Float[Array, " ncell"]:
        raise NotImplementedError


class Uniform(Absorptivity):
    value: Float[Array, ""]
    num_cells: int = eqx.field(static=True)

    @override
    def coefficient(self):
        return jnp.full(self.num_cells, self.value)


class Tabulated(Absorptivity):
    """A field supplied directly, e.g. read from the mesh file."""

    values: Float[Array, " ncell"]

    @override
    def coefficient(self):
        return self.values


class FromTemperature(Absorptivity):
    """A field generated from the temperature by any traceable model."""

    temperature: Float[Array, " ncell"]
    model: Callable[[Float[Array, " ncell"]], Float[Array, " ncell"]] = eqx.field(static=True)

    @override
    def coefficient(self):
        return self.model(self.temperature)
