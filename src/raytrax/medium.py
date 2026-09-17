from collections.abc import Callable
from typing import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class Absorptivity(eqx.Module):
    """A source of the absorption coefficient for each cell."""

    def coefficient(self) -> Float[Array, " ncell"]:
        """Give the absorption coefficient of each cell.

        Returns
        -------
        coefficient
            The absorption coefficient of each cell, in inverse mesh length units.
        """
        raise NotImplementedError


class Uniform(Absorptivity):
    """One absorption coefficient for all cells.

    Attributes
    ----------
    value
        The absorption coefficient.
    num_cells
        The number of cells in the mesh.
    """

    value: Float[Array, ""]
    num_cells: int = eqx.field(static=True)

    @override
    def coefficient(self):
        return jnp.full(self.num_cells, self.value)


class Tabulated(Absorptivity):
    """An absorption coefficient that the caller gives for each cell.

    Attributes
    ----------
    values
        The absorption coefficient of each cell, for example a field from the mesh file.
    """

    values: Float[Array, " ncell"]

    @override
    def coefficient(self):
        return self.values


class FromTemperature(Absorptivity):
    """An absorption coefficient that a model calculates from the temperature.

    Attributes
    ----------
    temperature
        The temperature of each cell.
    model
        A traceable function from the temperature of each cell to its absorption coefficient.
    """

    temperature: Float[Array, " ncell"]
    model: Callable[[Float[Array, " ncell"]], Float[Array, " ncell"]] = eqx.field(static=True)

    @override
    def coefficient(self):
        return self.model(self.temperature)
