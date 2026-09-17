import math
from dataclasses import dataclass, replace

import numpy as np
from jaxtyping import Float, Int

from raytrax.mesh.topology import MeshTopology, cell_face_vertices


@dataclass(frozen=True)
class FaceGeometry:
    """The geometry of each face.

    Attributes
    ----------
    centroid
        The centroid of each face.
    area
        The area of each face. In 2D, the length of each edge.
    normal
        The unit normal of each face, out of the cell in column 0 of `MeshTopology.face_cells`.
    offset
        The plane constant of each face, such that `normal . x = offset` on the face.
    flatness
        The largest deviation of each face from its plane, relative to the face size.
    fan_vertices
        The vertex ids of the simplices that make each face.
    fan_weight
        The area fraction of each simplex. Each row sums to 1.
    """

    centroid: Float[np.ndarray, "nface dim"]
    area: Float[np.ndarray, " nface"]
    normal: Float[np.ndarray, "nface dim"]
    offset: Float[np.ndarray, " nface"]
    flatness: Float[np.ndarray, " nface"]
    fan_vertices: Int[np.ndarray, "nface nfan dim"]
    fan_weight: Float[np.ndarray, "nface nfan"]


@dataclass(frozen=True)
class CellGeometry:
    """The geometry of each cell.

    Attributes
    ----------
    volume
        The volume of each cell. In 2D, the area of each cell.
    centroid
        The centroid of each cell.
    seed
        An interior point of each cell, which is a corner of all its simplices.
    simplex_vertices
        The vertex ids of the other corners of each simplex, with -1 where the
        simplex is not used.
    simplex_weight
        The volume fraction of each simplex. Each row sums to 1. A negative
        weight shows a cell that is not convex or is inverted.
    """

    volume: Float[np.ndarray, " ncell"]
    centroid: Float[np.ndarray, "ncell dim"]
    seed: Float[np.ndarray, "ncell dim"]
    simplex_vertices: Int[np.ndarray, "ncell nsimplex dim"]
    simplex_weight: Float[np.ndarray, "ncell nsimplex"]


@dataclass(frozen=True)
class MeshGeometry:
    """The geometry of a mesh.

    Attributes
    ----------
    vertices
        The position of each vertex.
    faces
        The geometry of each face.
    cells
        The geometry of each cell.
    """

    vertices: Float[np.ndarray, "nvert dim"]
    faces: FaceGeometry
    cells: CellGeometry


def _fan(points: Float[np.ndarray, "n k dim"]) -> Float[np.ndarray, "n nfan 3 dim"]:
    """Divide each polygon into triangles that share its first vertex.

    Parameters
    ----------
    points
        The corner positions of each polygon, in winding order.

    Returns
    -------
    triangles
        The corner positions of each triangle of each polygon.
    """
    num_verts = points.shape[-2]
    return np.stack([points[:, [0, j, j + 1]] for j in range(1, num_verts - 1)], axis=1)


def _triangle_normals(points: Float[np.ndarray, "... 3 dim"]) -> Float[np.ndarray, "... dim"]:
    """Give the area vector of each triangle.

    Parameters
    ----------
    points
        The corner positions of each triangle, in 3D.

    Returns
    -------
    normal
        The normal of each triangle by the right-hand rule, with a magnitude equal to its area.
    """
    ref, u, v = np.unstack(points, axis=-2)
    return np.cross(u - ref, v - ref) / 2


def face_geometry(topology: MeshTopology, vertices: np.ndarray) -> FaceGeometry:
    """Calculate the plane, area, centroid and simplex decomposition of each face.

    Parameters
    ----------
    topology
        The connectivity of the mesh.
    vertices
        The position of each vertex, with `topology.dim` columns.

    Returns
    -------
    faces
        The geometry of each face.
    """
    num_faces, dim = topology.num_faces, topology.dim
    max_fan = 1 if dim == 2 else topology.face_vertices.shape[1] - 2

    centroid = np.zeros((num_faces, dim))
    normal = np.zeros((num_faces, dim))
    area = np.zeros(num_faces)
    flatness = np.zeros(num_faces)
    fan_vertices = np.zeros((num_faces, max_fan, dim), dtype=np.int64)
    fan_weight = np.zeros((num_faces, max_fan))

    face_num_verts = topology.face_num_verts
    for num_verts in np.unique(face_num_verts):
        rows = np.flatnonzero(face_num_verts == num_verts)
        points = vertices[topology.face_vertices[rows, :num_verts]]

        if num_verts == 2:
            # A 2D edge: rotate the tangent a quarter turn to face outward.
            ref, end = np.unstack(points, axis=-2)
            vector = (end - ref) @ np.array([[0.0, -1.0], [1.0, 0.0]])
            centre = points.mean(axis=-2)
            pieces = topology.face_vertices[rows, None, :2]
            piece_area = np.linalg.vector_norm(end - ref, axis=-1)[:, None]
        else:
            # Exact for a planar polygon, best-fit plane for a warped quad.
            fan = _fan(points)
            fan_normal = _triangle_normals(fan)
            fan_area = np.linalg.vector_norm(fan_normal, axis=-1)
            vector = fan_normal.sum(axis=1)
            centre = np.einsum("ft,ftd->fd", fan_area, fan.mean(axis=-2)) / fan_area.sum(axis=1)[:, None]
            corners = np.arange(1, num_verts - 1)
            pieces = topology.face_vertices[rows][:, np.stack([np.zeros_like(corners), corners, corners + 1], axis=-1)]
            piece_area = fan_area

        magnitude = np.linalg.vector_norm(vector, axis=-1)
        unit = vector / magnitude[:, None]

        centroid[rows] = centre
        normal[rows] = unit
        area[rows] = magnitude
        fan_vertices[np.ix_(rows, range(pieces.shape[1]))] = pieces
        fan_weight[np.ix_(rows, range(pieces.shape[1]))] = piece_area / piece_area.sum(axis=1)[:, None]
        if num_verts > 3:
            # Warped faces break the half-space cell model, so measure them.
            offsets = np.einsum("fvd,fd->fv", points - centre[:, None, :], unit)
            flatness[rows] = np.abs(offsets).max(axis=1) / np.sqrt(magnitude)

    # Anchored on the centroid, not a corner, so a slightly warped quad stays
    # centred.
    offset = np.einsum("fd,fd->f", centroid, normal)
    return FaceGeometry(
        centroid=centroid,
        area=area,
        normal=normal,
        offset=offset,
        flatness=flatness,
        fan_vertices=fan_vertices,
        fan_weight=fan_weight,
    )


def cell_geometry(topology: MeshTopology, vertices: np.ndarray) -> CellGeometry:
    """Calculate the volume, centroid and simplex decomposition of each cell.

    Parameters
    ----------
    topology
        The connectivity of the mesh.
    vertices
        The position of each vertex, with `topology.dim` columns.

    Returns
    -------
    cells
        The geometry of each cell.
    """
    dim = topology.dim
    valid = topology.cell_face_valid
    face_verts = cell_face_vertices(topology)
    max_face_verts = face_verts.shape[-1]
    fans_per_face = 1 if dim == 2 else max_face_verts - 2

    # The vertex average sits strictly inside a convex cell.
    present = topology.cell_vertices >= 0
    seed = np.einsum("cv,cvd->cd", present, vertices[np.maximum(topology.cell_vertices, 0)])
    seed /= present.sum(axis=1)[:, None]

    simplices = []
    for slot in range(topology.max_faces):
        for corner in range(fans_per_face):
            picks = face_verts[:, slot, [0, 1] if dim == 2 else [0, corner + 1, corner + 2]]
            usable = valid[:, slot] & np.all(picks >= 0, axis=-1)
            simplices.append(np.where(usable[:, None], picks, -1))

    simplex_vertices = np.stack(simplices, axis=1)
    usable = np.all(simplex_vertices >= 0, axis=-1)

    corners = vertices[np.maximum(simplex_vertices, 0)]  # [ncell, nsimplex, dim, dim]
    volume = np.linalg.det(corners - seed[:, None, None, :]) / math.factorial(dim)
    volume = np.where(usable, volume, 0.0)
    total = volume.sum(axis=1)

    simplex_centroid = (corners.sum(axis=-2) + seed[:, None, :]) / (dim + 1)
    centroid = np.einsum("cs,csd->cd", volume, simplex_centroid) / total[:, None]

    return CellGeometry(
        volume=total,
        centroid=centroid,
        seed=seed,
        simplex_vertices=simplex_vertices,
        simplex_weight=volume / total[:, None],
    )


def build(topology: MeshTopology, vertices: np.ndarray) -> MeshGeometry:
    """Calculate the geometry of a mesh.

    Parameters
    ----------
    topology
        The connectivity of the mesh.
    vertices
        The position of each vertex. Columns past `topology.dim` are ignored.

    Returns
    -------
    geometry
        The geometry of the vertices, faces and cells.
    """
    vertices = np.asarray(vertices, dtype=float)[:, : topology.dim]
    return MeshGeometry(
        vertices=vertices,
        faces=face_geometry(topology, vertices),
        cells=cell_geometry(topology, vertices),
    )


def revolve(geometry: MeshGeometry, radial: int = 1) -> MeshGeometry:
    """Change the geometry of an (x, r) mesh to that of its solid of revolution.

    Cell volumes and face areas become those of the rings and surfaces that
    they make when revolved about the axis. The simplex weights change with
    them, so that a sample is uniform in the revolved solid. The positions stay
    in (x, r).

    Parameters
    ----------
    geometry
        The planar geometry of a 2D mesh.
    radial
        The column of the vertex positions that holds the radius.

    Returns
    -------
    geometry
        The revolved geometry.

    Raises
    ------
    ValueError
        If the mesh is not 2D, a vertex has a negative radius, or a cell is on the axis.
    """
    dim = geometry.vertices.shape[1]
    if dim != 2:
        raise ValueError(f"Revolving needs a 2D mesh, got {dim}D")
    if np.any(geometry.vertices[:, radial] < 0):
        raise ValueError(f"{np.sum(geometry.vertices[:, radial] < 0)} vertices lie at negative radius")

    cells, faces = geometry.cells, geometry.faces

    # Pappus: a piece sweeps 2 pi rbar times its own measure, and the simplex
    # centroid the volumes already came from is exactly the rbar it needs.
    corners = geometry.vertices[np.maximum(cells.simplex_vertices, 0)]
    simplex_radius = (corners[..., radial].sum(axis=-1) + cells.seed[:, None, radial]) / (dim + 1)
    cell_ring = cells.simplex_weight * simplex_radius
    cell_scale = cell_ring.sum(axis=1)
    if np.any(cell_scale <= 0):
        raise ValueError(f"{np.sum(cell_scale <= 0)} cells sweep no volume; they lie on the axis")

    fan_radius = geometry.vertices[np.maximum(faces.fan_vertices, 0)][..., radial].mean(axis=-1)
    face_ring = faces.fan_weight * fan_radius
    face_scale = face_ring.sum(axis=1)
    # A face lying on the axis sweeps nothing. Its area is genuinely zero, so it
    # can never be drawn from; keep its original weights rather than divide.
    swept = face_scale > 0

    return MeshGeometry(
        vertices=geometry.vertices,
        faces=replace(
            faces,
            area=2 * math.pi * faces.area * face_scale,
            fan_weight=np.where(
                swept[:, None], face_ring / np.where(swept, face_scale, 1.0)[:, None], faces.fan_weight
            ),
        ),
        cells=replace(
            cells,
            volume=2 * math.pi * cells.volume * cell_scale,
            simplex_weight=cell_ring / cell_scale[:, None],
        ),
    )
