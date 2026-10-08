"""A hot isothermal gas on a structured 2D square, between two cold black walls.

The walls at low and high x absorb, and the gas is the only source. The top
side is either a mirror, which makes the square an infinite slab, or another
cold black wall. Read flat, the square is extruded out of the plane; revolved
about its y = 0 edge, it is the `(x, r)` half-plane of a cylinder.
"""

from pathlib import Path

import gmsh
import jax.numpy as jnp
from scipy.constants import Stefan_Boltzmann

from raytrax import interactions, sources
from raytrax import scene as scene_module
from raytrax.interactions import Kind
from raytrax.io import vtk
from raytrax.mesh import geometry
from raytrax.paths import AxisymmetricPath, LinearPath
from raytrax.problem import Problem
from raytrax_examples.cases import MESHES, Case

SIDE = 15.0
TEMPERATURE = 2000.0


def mesh_file(cells: int = 15) -> Path:
    """Give the mesh file, and build it with gmsh first if it does not exist.

    Parameters
    ----------
    cells
        The number of quads along each side.

    Returns
    -------
    path
        The `.msh` file.
    """
    output = MESHES / f"square_{cells}.msh"
    if output.exists():
        return output

    gmsh.initialize(interruptible=False)
    gmsh.model.add("square")

    surface = gmsh.model.occ.addRectangle(0, 0, 0, SIDE, SIDE)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(dim=2, tags=[surface], name="square")

    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        x, y, _ = gmsh.model.occ.getCenterOfMass(dim, abs(tag))
        if abs(x) < 1e-9:
            name = "left"
        elif abs(x - SIDE) < 1e-9:
            name = "right"
        elif abs(y) < 1e-9:
            name = "axis"
        else:
            name = "lateral"
        gmsh.model.addPhysicalGroup(dim=1, tags=[abs(tag)], name=name)
        gmsh.model.mesh.setTransfiniteCurve(abs(tag), cells + 1)

    gmsh.model.mesh.setTransfiniteSurface(surface)
    gmsh.model.mesh.setRecombine(2, surface)

    gmsh.model.mesh.generate(dim=2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output))
    gmsh.finalize()
    return output


def case(
    axisymmetric: bool = False,
    lateral: Kind = Kind.SPECULAR,
    kappa: float = 1.0,
    temperature: float = TEMPERATURE,
    cells: int = 15,
) -> Case:
    """Read the square into the problem to simulate.

    Parameters
    ----------
    axisymmetric
        Whether the square is read as the `(x, r)` half-plane, revolved about y = 0.
    lateral
        The interaction at the side y = SIDE, and also at y = 0 when the square is read flat.
    kappa
        The absorption coefficient, uniform over the cells.
    temperature
        The temperature of the gas.
    cells
        The number of quads along each side of the mesh.

    Returns
    -------
    case
        The problem and the mesh it was built from.
    """
    mesh = vtk.read(mesh_file(cells))
    geom = geometry.revolve(mesh.geometry) if axisymmetric else mesh.geometry
    scene = scene_module.build(mesh.topology, geom, mesh.groups)
    scene_module.validate(mesh.topology, geom)

    # Revolved, the y = 0 edge is the axis and nothing should reach it.
    # Extruded it is the bottom side, and has to match the top.
    table = interactions.build(
        mesh.groups,
        {
            "left": Kind.ABSORB,
            "right": Kind.ABSORB,
            "lateral": lateral,
            "axis": Kind.ABSORB if axisymmetric else lateral,
        },
    )

    # A grey gas emits 4 kappa sigma T^4 per unit volume.
    emissive_power = Stefan_Boltzmann * temperature**4
    source = sources.build(cell_power=4 * kappa * emissive_power * jnp.asarray(geom.cells.volume))
    problem = Problem(
        scene,
        path=AxisymmetricPath() if axisymmetric else LinearPath(),
        interactions=table,
        absorptivity=jnp.full(mesh.topology.num_cells, kappa),
        sources=source,
    )
    return Case(mesh, geom, axisymmetric, problem)
