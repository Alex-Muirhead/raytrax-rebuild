from typing import Any

import equinox as eqx
from jaxtyping import Array, Float

from raytrax.interactions import Interactions
from raytrax.paths import Path
from raytrax.scene import Scene
from raytrax.sources import Sources


class Problem(eqx.Module):
    """The radiation problem to simulate: mesh, kinematics, boundaries, medium and emission.

    Attributes
    ----------
    scene
        The device arrays of the mesh.
    path
        The ray kinematics. It must agree with the geometry of `scene`.
    interactions
        The interaction table of the boundary groups.
    absorptivity
        The absorption coefficient of each cell.
    sources
        The emission distribution.
    """

    scene: Scene
    path: Path[Any]
    interactions: Interactions
    absorptivity: Float[Array, " ncell"]
    sources: Sources
