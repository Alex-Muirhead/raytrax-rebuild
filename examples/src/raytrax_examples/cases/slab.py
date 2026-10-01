"""The tangent slab on a 2D rectangle, with triangles scattered among quads.

A black hot end at low x radiates into the gas, a black wall at high x absorbs
what gets through, and the lateral side is a mirror. Read flat, the rectangle is
a slab extruded out of the plane; revolved about its y = 0 edge, it is the
`(x, r)` half-plane of a cylinder.
"""

from pathlib import Path

import gmsh
import jax.numpy as jnp
import numpy as np

from raytrax import interactions, sources
from raytrax import scene as scene_module
from raytrax.interactions import Kind
from raytrax.io import vtk
from raytrax.mesh import geometry
from raytrax.paths import AxisymmetricPath, LinearPath
from raytrax.problem import Problem
from raytrax_examples.cases import MESHES, Case

WIDTH = 1.0
HEIGHT = 1.0


def mesh_file(mesh_size: float = 0.07) -> Path:
    """Give the mesh file, and build it with gmsh first if it does not exist.

    Parameters
    ----------
    mesh_size
        The scale of the cell size, which varies from 0.25 to 0.7 of it across the domain.

    Returns
    -------
    path
        The `.msh` file.
    """
    output = MESHES / f"slab_{mesh_size}.msh"
    if output.exists():
        return output

    gmsh.initialize(interruptible=False)
    gmsh.model.add("slab")

    surface = gmsh.model.occ.addRectangle(-WIDTH / 2, 0, 0, WIDTH, HEIGHT)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(dim=2, tags=[surface], name="slab")

    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        x, y, _ = gmsh.model.occ.getCenterOfMass(dim, abs(tag))
        if abs(x + WIDTH / 2) < 1e-9:
            name = "hot_end"
        elif abs(x - WIDTH / 2) < 1e-9:
            name = "wall"
        elif abs(y) < 1e-9:
            name = "axis"
        else:
            name = "lateral"
        gmsh.model.addPhysicalGroup(dim=1, tags=[abs(tag)], name=name)

    # Two sine products of unrelated wavelength, so the size pattern never
    # repeats over the domain. The bracket stays in [0, 1], which floors the
    # size: without a floor the two waves cancel somewhere and gmsh meshes that
    # spot far too fine.
    wobble = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(
        wobble,
        "F",
        f"{0.25 * mesh_size} + {0.45 * mesh_size}"
        "*(0.5 + 0.25*Sin(6.0*x)*Cos(5.0*y + 1.0) + 0.25*Sin(11.0*y - 2.0)*Cos(9.0*x + 0.5))",
    )
    gmsh.model.mesh.field.setAsBackgroundMesh(wobble)
    # A background field only wins if the other size sources are switched off.
    for source in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber(f"Mesh.{source}", 0)

    # Algorithm 0 is the only recombination that leaves triangles behind; 1, 2
    # and 3 all reach pure quads. The quality floor sets the mix, and every quad
    # it accepts is convex, which `scene.validate` insists on.
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
    gmsh.option.setNumber("Mesh.RecombineMinimumQuality", 0.7)

    gmsh.model.mesh.generate(dim=2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output))
    gmsh.finalize()
    return output


def case(axisymmetric: bool = False, kappa: float = 1.0, emissive_power: float = 1.0, mesh_size: float = 0.07) -> Case:
    """Read the slab into the problem to simulate.

    Parameters
    ----------
    axisymmetric
        Whether the rectangle is read as the `(x, r)` half-plane, revolved about y = 0.
    kappa
        The absorption coefficient, uniform over the cells.
    emissive_power
        The power per unit area of the hot end.
    mesh_size
        The scale of the cell size of the mesh.

    Returns
    -------
    case
        The problem and the mesh it was built from.
    """
    mesh = vtk.read(mesh_file(mesh_size))
    geom = geometry.revolve(mesh.geometry) if axisymmetric else mesh.geometry
    scene = scene_module.build(mesh.topology, geom, mesh.groups)
    scene_module.validate(mesh.topology, geom)

    # Revolved, the y = 0 edge is the axis and nothing should reach it.
    # Extruded it is just another side, and has to match `lateral` or the slab
    # stops being symmetric.
    table = interactions.build(
        mesh.groups,
        {
            "hot_end": Kind.ABSORB,
            "wall": Kind.ABSORB,
            "lateral": Kind.SPECULAR,
            "axis": Kind.ABSORB if axisymmetric else Kind.SPECULAR,
        },
    )

    hot_end = np.flatnonzero(mesh.groups.face_group == mesh.groups.index("hot_end"))
    # Revolved, `faces.area` is already the area of the ring a face sweeps.
    source = sources.build(
        cell_power=jnp.zeros(mesh.topology.num_cells),
        face_power=emissive_power * geom.faces.area[hot_end],
        face=hot_end,
    )
    problem = Problem(
        scene,
        path=AxisymmetricPath() if axisymmetric else LinearPath(),
        interactions=table,
        absorptivity=jnp.full(mesh.topology.num_cells, kappa),
        sources=source,
    )
    return Case(mesh, geom, axisymmetric, problem)
