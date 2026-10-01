"""The problems the examples and scripts run, one module per geometry.

Each module builds its own mesh with gmsh, cached under `examples/meshes/`, and
reads it into a `Case`. Delete a cached `.msh` to build it again.
"""

from pathlib import Path
from typing import NamedTuple

from raytrax.io import vtk
from raytrax.mesh import MeshGeometry
from raytrax.problem import Problem

# The editable install leaves this file at examples/src/raytrax_examples/cases.
MESHES = Path(__file__).resolve().parents[3] / "meshes"


class Case(NamedTuple):
    """One problem, with the mesh it was built from.

    Attributes
    ----------
    mesh
        The mesh as read. Its geometry holds the positions, in `(x, r)` when revolved.
    geom
        The geometry the problem uses, ring-weighted when revolved.
    axisymmetric
        Whether the mesh is read as the `(x, r)` half-plane.
    problem
        The problem to simulate.
    """

    mesh: vtk.Mesh
    geom: MeshGeometry
    axisymmetric: bool
    problem: Problem

    @property
    def mean_heating(self) -> float:
        """The volumetric heating if every emitted joule landed uniformly in the gas."""
        return float(self.problem.sources.total) / float(self.geom.cells.volume.sum())
