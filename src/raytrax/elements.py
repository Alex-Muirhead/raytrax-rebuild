from dataclasses import dataclass

# Face windings are counterclockwise viewed from outside the element, so the
# right-hand normal points outward. Vertex orderings follow VTK/meshio.


@dataclass(frozen=True)
class ElementType:
    """A first-order VTK cell, described by the local vertex ids of its faces."""

    name: str
    dim: int
    num_vertices: int
    faces: tuple[tuple[int, ...], ...]

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def max_face_verts(self) -> int:
        return max(len(face) for face in self.faces)


# 2D elements: "faces" are edges, wound counterclockwise around the element.
TRIANGLE = ElementType("triangle", 2, 3, ((0, 1), (1, 2), (2, 0)))
QUAD = ElementType("quad", 2, 4, ((0, 1), (1, 2), (2, 3), (3, 0)))

# 3D elements.
TETRA = ElementType(
    "tetra",
    3,
    4,
    (
        (0, 2, 1),  # base, away from vertex 3
        (0, 1, 3),  # away from vertex 2
        (1, 2, 3),  # away from vertex 0
        (0, 3, 2),  # away from vertex 1
    ),
)
HEXAHEDRON = ElementType(
    "hexahedron",
    3,
    8,
    (
        (0, 3, 2, 1),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),  # front
        (1, 2, 6, 5),  # right
        (2, 3, 7, 6),  # back
        (3, 0, 4, 7),  # left
    ),
)
WEDGE = ElementType(
    "wedge",
    3,
    6,
    (
        (0, 2, 1),  # bottom triangle
        (3, 4, 5),  # top triangle
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (2, 0, 3, 5),
    ),
)
PYRAMID = ElementType(
    "pyramid",
    3,
    5,
    (
        (0, 3, 2, 1),  # base quad, away from apex 4
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ),
)

ELEMENT_TYPES: dict[str, ElementType] = {
    element.name: element for element in (TRIANGLE, QUAD, TETRA, HEXAHEDRON, WEDGE, PYRAMID)
}

FACE_ELEMENT_NAMES: dict[int, str] = {2: "line", 3: "triangle", 4: "quad"}


def lookup(name: str) -> ElementType:
    if name not in ELEMENT_TYPES:
        supported = ", ".join(sorted(ELEMENT_TYPES))
        raise ValueError(f"Unsupported element type {name!r}, expected one of: {supported}")
    return ELEMENT_TYPES[name]
