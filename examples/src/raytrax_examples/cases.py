"""The named problems the examples and scripts run."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from raytrax import interactions, sources
from raytrax import scene as scene_module
from raytrax.interactions import Kind
from raytrax.io import vtk
from raytrax.mesh import MeshGeometry, geometry
from raytrax.paths import AxisymmetricPath, LinearPath
from raytrax.problem import Problem
from raytrax_examples import meshes

# Mesh name, and whether it is read as the (x, r) half-plane.
CASES = {
    "slab_planar": ("slab", False),
    "slab_axisymmetric": ("slab", True),
    "plane": ("plane", False),
    "plane_axisymmetric": ("plane", True),
    "cylinder_coarse": ("cylinder_coarse", False),
    "cylinder": ("cylinder", False),
    "reentry": ("reentry", True),
}


class Case(NamedTuple):
    """One named problem, with the mesh it was built from.

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


def group_faces(mesh: vtk.Mesh, name: str) -> np.ndarray:
    """Give the face rows of one boundary group.

    Parameters
    ----------
    mesh
        The mesh that holds the group.
    name
        The name of the group.

    Returns
    -------
    faces
        The face rows.
    """
    return np.flatnonzero(mesh.groups.face_group == mesh.groups.index(name))


def prepare(name: str, kappa: float = 1.0, emissive_power: float = 1.0) -> Case:
    """Read one named case into the problem to simulate.

    The tangent-slab cases have a black hot end, a black cold wall and a
    mirrored lateral surface. The re-entry case radiates from the shock, absorbs
    at the shock and the wall, and lets rays escape through the outflow.

    Parameters
    ----------
    name
        The key of the case in `CASES`.
    kappa
        The absorption coefficient, uniform over the cells.
    emissive_power
        The power per unit area of the emitting boundary.

    Returns
    -------
    case
        The problem and the mesh it was built from.
    """
    stem, axisymmetric = CASES[name]
    mesh = vtk.read(meshes.path(stem))
    geom = geometry.revolve(mesh.geometry) if axisymmetric else mesh.geometry
    path = AxisymmetricPath() if axisymmetric else LinearPath()

    scene = scene_module.build(mesh.topology, geom, mesh.groups)
    scene_module.validate(mesh.topology, geom)

    if stem == "reentry":
        # ABSORB rather than ESCAPE on the shock and the wall, so that their
        # deposits can be read back as a flux.
        kinds = {"shock": Kind.ABSORB, "wall": Kind.ABSORB, "outflow": Kind.ESCAPE, "axis": Kind.ABSORB}
        emitting = "shock"
    else:
        kinds = {"hot_end": Kind.ABSORB, "wall": Kind.ABSORB, "lateral": Kind.SPECULAR}
        if "axis" in mesh.groups.names:
            # Revolved, the y = 0 edge is the axis and nothing should reach it.
            # Extruded it is just another side, and has to match `lateral` or
            # the slab stops being symmetric.
            kinds["axis"] = Kind.ABSORB if axisymmetric else Kind.SPECULAR
        emitting = "hot_end"
    table = interactions.build(mesh.groups, kinds)

    faces = group_faces(mesh, emitting)
    # Revolved, `faces.area` is already the area of the ring a face sweeps.
    source = sources.build(
        cell_power=jnp.zeros(mesh.topology.num_cells),
        face_power=emissive_power * geom.faces.area[faces],
        face=faces,
    )
    problem = Problem(
        scene,
        path=path,
        interactions=table,
        absorptivity=jnp.full(mesh.topology.num_cells, kappa),
        sources=source,
    )
    return Case(mesh, geom, axisymmetric, problem)
