"""The tangent slab on a 2D rectangle, quads on the lower half and triangles on the upper.

The two element types share the interior faces along y = HEIGHT / 2. A black hot end at low x radiates into the gas,
a black wall at high x absorbs what gets through, and the lateral side is a
mirror. Read flat, the rectangle is a slab extruded out of the plane; revolved
about its y = 0 edge, it is the `(x, r)` half-plane of a cylinder.
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


def mesh_file(mesh_size: float = 0.12) -> Path:
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
    output = MESHES / f"plane_{mesh_size}.msh"
    if output.exists():
        return output

    gmsh.initialize(interruptible=False)
    gmsh.model.add("plane")

    lower = gmsh.model.occ.addRectangle(-WIDTH / 2, 0, 0, WIDTH, HEIGHT / 2)
    upper = gmsh.model.occ.addRectangle(-WIDTH / 2, HEIGHT / 2, 0, WIDTH, HEIGHT / 2)
    # Fragment so the two halves share vertices along y = HEIGHT / 2.
    parts, _ = gmsh.model.occ.fragment([(2, lower)], [(2, upper)])
    gmsh.model.occ.synchronize()

    surfaces = [tag for dim, tag in parts if dim == 2]
    quad_half = min(surfaces, key=lambda t: gmsh.model.occ.getCenterOfMass(2, t)[1])
    gmsh.model.addPhysicalGroup(dim=2, tags=surfaces, name="plane")

    # Fragmenting splits each outer edge in two, so gather curves by name and
    # register one physical group each -- one group per call would leave the
    # duplicates untagged.
    edges: dict[str, list[int]] = {}
    for dim, tag in gmsh.model.getBoundary(parts, oriented=False, combined=True):
        x, y, _ = gmsh.model.occ.getCenterOfMass(abs(dim), abs(tag))
        if abs(x + WIDTH / 2) < 1e-9:
            name = "hot_end"
        elif abs(x - WIDTH / 2) < 1e-9:
            name = "wall"
        elif abs(y) < 1e-9:
            name = "axis"
        else:
            name = "lateral"
        edges.setdefault(name, []).append(abs(tag))
    for name, tags in edges.items():
        gmsh.model.addPhysicalGroup(dim=1, tags=tags, name=name)

    gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size)
    gmsh.model.mesh.setRecombine(2, quad_half)
    gmsh.model.mesh.generate(dim=2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output))
    gmsh.finalize()
    return output


def case(axisymmetric: bool = False, kappa: float = 1.0, emissive_power: float = 1.0, mesh_size: float = 0.12) -> Case:
    """Read the plane into the problem to simulate.

    Parameters
    ----------
    axisymmetric
        Whether the rectangle is read as the `(x, r)` half-plane, revolved about y = 0.
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
