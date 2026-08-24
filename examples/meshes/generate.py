"""Generate the example meshes with gmsh.

    uv run python examples/meshes/generate.py

The plane is deliberately mixed tri/quad: one half is recombined into quads and
the other left as triangles, so the two element types share an interior face.
That is the case the rebuild exists to support. It doubles as the axisymmetric
domain, revolved about its y = 0 edge.
"""

import argparse
from pathlib import Path

import gmsh

HERE = Path(__file__).parent


def cylinder(radius: float = 1.0, length: float = 1.0, mesh_size: float = 0.25, output: Path = HERE / "cylinder.msh"):
    """A tetrahedral cylinder with tagged end caps, axis along x."""
    gmsh.initialize()
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
    gmsh.write(str(output))
    gmsh.finalize()
    print(f"wrote {output}")


def plane(width: float = 1.0, height: float = 1.0, mesh_size: float = 0.12, output: Path = HERE / "plane.msh"):
    """A 2D rectangle, quads on the lower half and triangles on the upper.

    It sits at y >= 0 so the same mesh reads either as an extruded slab or,
    revolved about y = 0, as the meridional half-plane of `cylinder`.
    """
    gmsh.initialize()
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
    gmsh.write(str(output))
    gmsh.finalize()
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", nargs="?", default="all", choices=["all", "cylinder", "plane"])
    parser.add_argument("--mesh-size", type=float, default=None)
    args = parser.parse_args()

    size = {} if args.mesh_size is None else {"mesh_size": args.mesh_size}
    if args.which in ("all", "cylinder"):
        cylinder(**size)
    if args.which in ("all", "plane"):
        plane(**size)
