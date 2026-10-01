"""The shock layer around a Stardust-like capsule, solved axisymmetrically.

The mesh is the `(x, r)` half-plane between the capsule and the shock. The
capsule is solid and is not meshed: it is the `wall` boundary. Lengths are
metres, the stagnation point is the origin, and the freestream runs along +x.
The shock radiates into the gas and absorbs what comes back, the wall and the
axis absorb, and rays escape through the outflow.
"""

import math
from pathlib import Path

import gmsh
import jax.numpy as jnp
import numpy as np

from raytrax import interactions, sources
from raytrax import scene as scene_module
from raytrax.interactions import Kind
from raytrax.io import vtk
from raytrax.mesh import geometry
from raytrax.paths import AxisymmetricPath
from raytrax.problem import Problem
from raytrax_examples.cases import MESHES, Case

# The capsule: a sphere-cone forebody, a shoulder arc and a conical backshell.
NOSE_RADIUS = 0.2202
CONE_ANGLE = 60.0
BASE_RADIUS = 0.4064
SHOULDER_RADIUS = 0.0202
AFT_ANGLE = 30.0
# Where the domain ends downstream of the stagnation point.
LENGTH = 0.5
# The shock: its distance ahead of the stagnation point, and the angle of its
# asymptote to the axis.
STANDOFF = 0.02
SHOCK_ANGLE = 60.0


def mesh_file(mesh_size: float = 1.0) -> Path:
    """Give the mesh file, and build it with gmsh first if it does not exist.

    Parameters
    ----------
    mesh_size
        The cell size, as a factor on the grading from the stagnation point.

    Returns
    -------
    path
        The `.msh` file.
    """
    output = MESHES / f"reentry_{mesh_size}.msh"
    if output.exists():
        return output

    gmsh.initialize(interruptible=False)
    gmsh.model.add("reentry")

    cone, aft = math.radians(CONE_ANGLE), math.radians(AFT_ANGLE)

    # Body profile, downstream from the stagnation point. The nose sphere runs
    # to where its normal has turned through 90 - CONE_ANGLE, which is where it
    # meets the cone. The shoulder arc is tangent to the cone and to the
    # backshell, so its centre sits one shoulder radius inside each.
    nose_end = (NOSE_RADIUS * (1 - math.sin(cone)), NOSE_RADIUS * math.cos(cone))
    shoulder_r = BASE_RADIUS - SHOULDER_RADIUS
    cone_end_r = shoulder_r + SHOULDER_RADIUS * math.cos(cone)
    cone_end = (nose_end[0] + (cone_end_r - nose_end[1]) / math.tan(cone), cone_end_r)
    shoulder = (cone_end[0] + SHOULDER_RADIUS * math.sin(cone), shoulder_r)
    aft_end = (shoulder[0] + SHOULDER_RADIUS * math.sin(aft), shoulder_r + SHOULDER_RADIUS * math.cos(aft))
    base = (LENGTH, aft_end[1] - (LENGTH - aft_end[0]) * math.tan(aft))
    if base[1] <= 0.0:
        raise ValueError(f"the backshell closes on the axis before x = {LENGTH}")

    # Billig's hyperbola: vertex one standoff ahead of the nose, an asymptote
    # at `SHOCK_ANGLE` to the axis, and 1.143 nose radii of curvature at the
    # vertex, which is his high-Mach limit. Left at the cone angle the far shock
    # runs parallel to the forebody, so the layer thickens steadily along it
    # rather than pinching shut or flaring away.
    cot = 1.0 / math.tan(math.radians(SHOCK_ANGLE))
    semi_x, semi_r = 1.143 * NOSE_RADIUS * cot**2, 1.143 * NOSE_RADIUS * cot
    # r = semi_r sinh(u) walks the curve at a near-constant arc step, so the
    # vertex is sampled finely without piling points onto the tail.
    u_end = math.acosh((LENGTH + STANDOFF) / semi_x + 1.0)
    shock_curve = [
        (semi_x * (math.cosh(u) - 1.0) - STANDOFF, semi_r * math.sinh(u)) for u in (u_end * i / 160 for i in range(161))
    ]

    def point(position):
        return gmsh.model.occ.addPoint(position[0], position[1], 0.0)

    vertex, stagnation = point(shock_curve[0]), point((0.0, 0.0))
    corners = [stagnation, point(nose_end), point(cone_end), point(aft_end), point(base)]
    spline = [vertex, *map(point, shock_curve[1:])]

    wall = [
        gmsh.model.occ.addCircleArc(corners[0], point((NOSE_RADIUS, 0.0)), corners[1]),
        gmsh.model.occ.addLine(corners[1], corners[2]),
        gmsh.model.occ.addCircleArc(corners[2], point(shoulder), corners[3]),
        gmsh.model.occ.addLine(corners[3], corners[4]),
    ]
    axis = gmsh.model.occ.addLine(vertex, corners[0])
    outflow = gmsh.model.occ.addLine(corners[4], spline[-1])
    shock = gmsh.model.occ.addSpline(spline[::-1])

    loop = gmsh.model.occ.addCurveLoop([axis, *wall, outflow, shock])
    surface = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(dim=2, tags=[surface], name="shock_layer")
    for name, tags in (("axis", [axis]), ("wall", wall), ("outflow", [outflow]), ("shock", [shock])):
        gmsh.model.addPhysicalGroup(dim=1, tags=tags, name=name)

    # Cells grow with distance from the stagnation point: several across the
    # STANDOFF at the nose, and an order of magnitude coarser out in the
    # expansion behind the shoulder, which is large and carries no gradient.
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "PointsList", [stagnation])
    graded = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(graded, "InField", distance)
    gmsh.model.mesh.field.setNumber(graded, "SizeMin", STANDOFF / 10)
    gmsh.model.mesh.field.setNumber(graded, "SizeMax", BASE_RADIUS / 24)
    gmsh.model.mesh.field.setNumber(graded, "DistMin", NOSE_RADIUS / 4)
    gmsh.model.mesh.field.setNumber(graded, "DistMax", 3 * BASE_RADIUS)
    gmsh.model.mesh.field.setAsBackgroundMesh(graded)
    # A background field only wins if the other size sources are switched off.
    for source in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber(f"Mesh.{source}", 0)

    # Quads suit a finite-volume solver; algorithm 0 with a quality floor
    # leaves triangles only where the layer is too narrow to pair them off.
    gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
    gmsh.option.setNumber("Mesh.RecombineMinimumQuality", 0.5)

    gmsh.model.mesh.generate(dim=2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output))
    gmsh.finalize()
    return output


def case(kappa: float = 1.0, emissive_power: float = 1.0, mesh_size: float = 1.0) -> Case:
    """Read the shock layer into the problem to simulate.

    Parameters
    ----------
    kappa
        The absorption coefficient, uniform over the cells.
    emissive_power
        The power per unit area of the shock.
    mesh_size
        The cell size of the mesh, as a factor on its grading.

    Returns
    -------
    case
        The problem and the mesh it was built from.
    """
    mesh = vtk.read(mesh_file(mesh_size))
    geom = geometry.revolve(mesh.geometry)
    scene = scene_module.build(mesh.topology, geom, mesh.groups)
    scene_module.validate(mesh.topology, geom)

    # ABSORB rather than ESCAPE on the shock and the wall, so that their
    # deposits can be read back as a flux.
    table = interactions.build(
        mesh.groups,
        {"shock": Kind.ABSORB, "wall": Kind.ABSORB, "outflow": Kind.ESCAPE, "axis": Kind.ABSORB},
    )

    shock = np.flatnonzero(mesh.groups.face_group == mesh.groups.index("shock"))
    # Revolved, `faces.area` is already the area of the ring a face sweeps.
    source = sources.build(
        cell_power=jnp.zeros(mesh.topology.num_cells),
        face_power=emissive_power * geom.faces.area[shock],
        face=shock,
    )
    problem = Problem(
        scene,
        path=AxisymmetricPath(),
        interactions=table,
        absorptivity=jnp.full(mesh.topology.num_cells, kappa),
        sources=source,
    )
    return Case(mesh, geom, True, problem)
