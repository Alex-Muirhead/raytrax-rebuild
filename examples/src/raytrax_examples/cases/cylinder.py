"""The tangent slab in a 3D tetrahedral cylinder, axis along x.

A black hot end cap at low x radiates into the gas, a black wall cap at high x
absorbs what gets through, and the curved side is a mirror.
"""

from pathlib import Path

import gmsh
import jax.numpy as jnp
import numpy as np

from raytrax import interactions, sources
from raytrax import scene as scene_module
from raytrax.interactions import Kind
from raytrax.io import vtk
from raytrax.paths import LinearPath
from raytrax.problem import Problem
from raytrax_examples.cases import MESHES, Case

RADIUS = 1.0
LENGTH = 1.0


def mesh_file(mesh_size: float = 0.25) -> Path:
    """Give the mesh file, and build it with gmsh first if it does not exist.

    Parameters
    ----------
    mesh_size
        The cell size, as a factor on the gmsh default.

    Returns
    -------
    path
        The `.msh` file.
    """
    output = MESHES / f"cylinder_{mesh_size}.msh"
    if output.exists():
        return output

    gmsh.initialize(interruptible=False)
    gmsh.model.add("cylinder")

    volume = gmsh.model.occ.addCylinder(-LENGTH / 2, 0, 0, LENGTH, 0, 0, RADIUS)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(dim=3, tags=[volume], name="cylinder")

    # End caps by centre of mass; everything else is the lateral surface.
    for dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False):
        x, _, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        name = "hot_end" if abs(x + LENGTH / 2) < 1e-9 else "wall" if abs(x - LENGTH / 2) < 1e-9 else "lateral"
        gmsh.model.addPhysicalGroup(dim=dim, tags=[tag], name=name)

    gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size)
    gmsh.model.mesh.generate(dim=3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output))
    gmsh.finalize()
    return output


def case(kappa: float = 1.0, emissive_power: float = 1.0, mesh_size: float = 0.25) -> Case:
    """Read the cylinder into the problem to simulate.

    Parameters
    ----------
    kappa
        The absorption coefficient, uniform over the cells.
    emissive_power
        The power per unit area of the hot end.
    mesh_size
        The cell size of the mesh, as a factor on the gmsh default.

    Returns
    -------
    case
        The problem and the mesh it was built from.
    """
    mesh = vtk.read(mesh_file(mesh_size))
    scene = scene_module.build(mesh.topology, mesh.geometry, mesh.groups)
    scene_module.validate(mesh.topology, mesh.geometry)

    table = interactions.build(
        mesh.groups,
        {"hot_end": Kind.ABSORB, "wall": Kind.ABSORB, "lateral": Kind.SPECULAR},
    )

    hot_end = np.flatnonzero(mesh.groups.face_group == mesh.groups.index("hot_end"))
    source = sources.build(
        cell_power=jnp.zeros(mesh.topology.num_cells),
        face_power=emissive_power * mesh.geometry.faces.area[hot_end],
        face=hot_end,
    )
    problem = Problem(
        scene,
        path=LinearPath(),
        interactions=table,
        absorptivity=jnp.full(mesh.topology.num_cells, kappa),
        sources=source,
    )
    return Case(mesh, mesh.geometry, False, problem)
