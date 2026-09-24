"""Generate the example meshes with gmsh.

    uv run python -m raytrax_examples.meshes [which]

The examples call `path` instead, which generates a mesh the first time it is
asked for and reads the cached file after that.

The plane is deliberately mixed tri/quad: one half is recombined into quads and
the other left as triangles, so the two element types share an interior face.
That is the case the rebuild exists to support. It doubles as the axisymmetric
domain, revolved about its y = 0 edge.

The slab is the same geometry and the same groups, but with the two element
types interleaved instead of split, and the cell size varying across the domain.

The reentry mesh is a different problem: the shock layer around a re-entry
capsule, meshed the way an inviscid finite-volume solve would want it. Graded
by distance from the stagnation point and recombined to quads, with no wall
clustering, because there is no boundary layer to resolve.
"""

import argparse
import functools
import math
from collections.abc import Callable
from pathlib import Path

import gmsh

# The editable install leaves this file at examples/src/raytrax_examples.
MESHES = Path(__file__).resolve().parents[2] / "meshes"


def cylinder(radius: float = 1.0, length: float = 1.0, mesh_size: float = 0.25, output: Path = MESHES / "cylinder.msh"):
    """A tetrahedral cylinder with tagged end caps, axis along x."""
    gmsh.initialize(interruptible=False)
    gmsh.model.add("cylinder")

    volume = gmsh.model.occ.addCylinder(-length / 2, 0, 0, length, 0, 0, radius)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(dim=3, tags=[volume], name="cylinder")

    # End caps by centre of mass; everything else is the lateral surface.
    for dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False):
        x, _, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        name = "hot_end" if abs(x + length / 2) < 1e-9 else "wall" if abs(x - length / 2) < 1e-9 else "lateral"
        gmsh.model.addPhysicalGroup(dim=dim, tags=[tag], name=name)

    gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size)
    gmsh.model.mesh.generate(dim=3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output))
    gmsh.finalize()
    print(f"wrote {output}")


def plane(width: float = 1.0, height: float = 1.0, mesh_size: float = 0.12, output: Path = MESHES / "plane.msh"):
    """A 2D rectangle, quads on the lower half and triangles on the upper.

    It sits at y >= 0 so the same mesh reads either as an extruded slab or,
    revolved about y = 0, as the `(x, r)` half-plane of `cylinder`.
    """
    gmsh.initialize(interruptible=False)
    gmsh.model.add("plane")

    lower = gmsh.model.occ.addRectangle(-width / 2, 0, 0, width, height / 2)
    upper = gmsh.model.occ.addRectangle(-width / 2, height / 2, 0, width, height / 2)
    # Fragment so the two halves share vertices along y = height / 2.
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
        if abs(x + width / 2) < 1e-9:
            name = "hot_end"
        elif abs(x - width / 2) < 1e-9:
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
    print(f"wrote {output}")


def slab(width: float = 1.0, height: float = 1.0, mesh_size: float = 0.07, output: Path = MESHES / "slab.msh"):
    """A 2D rectangle with triangles scattered among quads, at varying cell size.

    Same geometry and groups as `plane`, so it substitutes for it anywhere.
    """
    gmsh.initialize(interruptible=False)
    gmsh.model.add("slab")

    surface = gmsh.model.occ.addRectangle(-width / 2, 0, 0, width, height)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(dim=2, tags=[surface], name="slab")

    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        x, y, _ = gmsh.model.occ.getCenterOfMass(dim, abs(tag))
        if abs(x + width / 2) < 1e-9:
            name = "hot_end"
        elif abs(x - width / 2) < 1e-9:
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
    print(f"wrote {output}")


def reentry(
    nose_radius: float = 0.2202,
    cone_angle: float = 60.0,
    base_radius: float = 0.4064,
    shoulder_radius: float = 0.0202,
    aft_angle: float = 30.0,
    length: float = 0.5,
    standoff: float = 0.02,
    shock_angle: float = 60.0,
    mesh_size: float = 1.0,
    output: Path = MESHES / "reentry.msh",
):
    """The shock layer around a Stardust-like capsule, as the (x, r) half-plane.

    The capsule is solid and is not meshed: it is the `wall` boundary. Lengths
    are metres, the stagnation point is the origin, and the freestream runs
    along +x.
    """
    gmsh.initialize(interruptible=False)
    gmsh.model.add("reentry")

    cone, aft = math.radians(cone_angle), math.radians(aft_angle)

    # Body profile, downstream from the stagnation point. The nose sphere runs
    # to where its normal has turned through 90 - cone_angle, which is where it
    # meets the cone. The shoulder arc is tangent to the cone and to the
    # backshell, so its centre sits one shoulder radius inside each.
    nose_end = (nose_radius * (1 - math.sin(cone)), nose_radius * math.cos(cone))
    shoulder_r = base_radius - shoulder_radius
    cone_end_r = shoulder_r + shoulder_radius * math.cos(cone)
    cone_end = (nose_end[0] + (cone_end_r - nose_end[1]) / math.tan(cone), cone_end_r)
    shoulder = (cone_end[0] + shoulder_radius * math.sin(cone), shoulder_r)
    aft_end = (shoulder[0] + shoulder_radius * math.sin(aft), shoulder_r + shoulder_radius * math.cos(aft))
    base = (length, aft_end[1] - (length - aft_end[0]) * math.tan(aft))
    if base[1] <= 0.0:
        raise ValueError(f"the backshell closes on the axis before x = {length}")

    # Billig's hyperbola: vertex one standoff ahead of the nose, an asymptote
    # at `shock_angle` to the axis, and 1.143 nose radii of curvature at the
    # vertex, which is his high-Mach limit. Left at the cone angle the far shock
    # runs parallel to the forebody, so the layer thickens steadily along it
    # rather than pinching shut or flaring away.
    cot = 1.0 / math.tan(math.radians(shock_angle))
    semi_x, semi_r = 1.143 * nose_radius * cot**2, 1.143 * nose_radius * cot
    # r = semi_r sinh(u) walks the curve at a near-constant arc step, so the
    # vertex is sampled finely without piling points onto the tail.
    u_end = math.acosh((length + standoff) / semi_x + 1.0)
    shock_curve = [
        (semi_x * (math.cosh(u) - 1.0) - standoff, semi_r * math.sinh(u)) for u in (u_end * i / 160 for i in range(161))
    ]

    def point(position):
        return gmsh.model.occ.addPoint(position[0], position[1], 0.0)

    vertex, stagnation = point(shock_curve[0]), point((0.0, 0.0))
    corners = [stagnation, point(nose_end), point(cone_end), point(aft_end), point(base)]
    spline = [vertex, *map(point, shock_curve[1:])]

    wall = [
        gmsh.model.occ.addCircleArc(corners[0], point((nose_radius, 0.0)), corners[1]),
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
    # standoff at the nose, and an order of magnitude coarser out in the
    # expansion behind the shoulder, which is large and carries no gradient.
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "PointsList", [stagnation])
    graded = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(graded, "InField", distance)
    gmsh.model.mesh.field.setNumber(graded, "SizeMin", standoff / 10)
    gmsh.model.mesh.field.setNumber(graded, "SizeMax", base_radius / 24)
    gmsh.model.mesh.field.setNumber(graded, "DistMin", nose_radius / 4)
    gmsh.model.mesh.field.setNumber(graded, "DistMax", 3 * base_radius)
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
    print(f"wrote {output}")


BUILDERS: dict[str, Callable[..., None]] = {
    "cylinder": cylinder,
    # Coarse enough that a cut through it reads as individual cells.
    "cylinder_coarse": functools.partial(cylinder, mesh_size=0.6),
    "plane": plane,
    "slab": slab,
    "reentry": reentry,
}


def path(name: str) -> Path:
    """Give the file of one named mesh, and generate it first if it does not exist.

    Parameters
    ----------
    name
        The key of the mesh in `BUILDERS`.

    Returns
    -------
    path
        The `.msh` file.
    """
    output = MESHES / f"{name}.msh"
    if not output.exists():
        BUILDERS[name](output=output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", nargs="?", default="all", choices=["all", *BUILDERS])
    parser.add_argument("--mesh-size", type=float, default=None)
    args = parser.parse_args()

    size = {} if args.mesh_size is None else {"mesh_size": args.mesh_size}
    for name, builder in BUILDERS.items():
        if args.which in ("all", name):
            builder(output=MESHES / f"{name}.msh", **size)
