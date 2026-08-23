import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int

from raytrax.mesh.topology import MeshTopology, cell_face_vertices


@dataclass(frozen=True)
class FaceGeometry:
    centroid: Float[np.ndarray, "nface dim"]
    area: Float[np.ndarray, " nface"]
    normal: Float[np.ndarray, "nface dim"]  # unit, outward from face_cells[:, 0]
    offset: Float[np.ndarray, " nface"]  # plane constant, normal . x = offset
    flatness: Float[np.ndarray, " nface"]  # deviation relative to face size
    fan_vertices: Int[np.ndarray, "nface nfan dim"]  # vertex ids of a simplex fan
    fan_weight: Float[np.ndarray, "nface nfan"]  # area fractions, rows sum to 1


@dataclass(frozen=True)
class CellGeometry:
    volume: Float[np.ndarray, " ncell"]
    centroid: Float[np.ndarray, "ncell dim"]
    seed: Float[np.ndarray, "ncell dim"]  # interior point every simplex spans from
    simplex_vertices: Int[np.ndarray, "ncell nsimplex dim"]  # vertex ids, -1 where unused
    simplex_weight: Float[np.ndarray, "ncell nsimplex"]  # volume fractions, rows sum to 1


@dataclass(frozen=True)
class MeshGeometry:
    vertices: Float[np.ndarray, "nvert dim"]
    faces: FaceGeometry
    cells: CellGeometry


def _fan(points: Float[np.ndarray, "n k dim"]) -> Float[np.ndarray, "n nfan 3 dim"]:
    """Split each polygon into triangles about its first vertex."""
    num_verts = points.shape[-2]
    return np.stack([points[:, [0, j, j + 1]] for j in range(1, num_verts - 1)], axis=1)


def _triangle_normals(points: Float[np.ndarray, "... 3 dim"]) -> Float[np.ndarray, "... dim"]:
    """Half the cross product of each triangle's edges: normal with area magnitude."""
    ref, u, v = np.unstack(points, axis=-2)
    return np.cross(u - ref, v - ref) / 2


def face_geometry(topology: MeshTopology, vertices: np.ndarray) -> FaceGeometry:
    """Plane, area and centroid of every face, from its canonical winding."""
    num_faces, dim = topology.num_faces, topology.dim
    max_fan = 1 if dim == 2 else topology.face_vertices.shape[1] - 2

    centroid = np.zeros((num_faces, dim))
    normal = np.zeros((num_faces, dim))
    area = np.zeros(num_faces)
    flatness = np.zeros(num_faces)
    fan_vertices = np.zeros((num_faces, max_fan, dim), dtype=np.int64)
    fan_weight = np.zeros((num_faces, max_fan))

    for num_verts in np.unique(topology.face_num_verts):
        rows = np.flatnonzero(topology.face_num_verts == num_verts)
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
    """Volume, centroid and a simplex decomposition of every cell.

    A simplex has positive volume exactly when the cell is convex and correctly
    oriented, which `scene.validate` checks.
    """
    dim = topology.dim
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
            usable = topology.cell_face_valid[:, slot] & np.all(picks >= 0, axis=-1)
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
    vertices = np.asarray(vertices, dtype=float)[:, : topology.dim]
    return MeshGeometry(
        vertices=vertices,
        faces=face_geometry(topology, vertices),
        cells=cell_geometry(topology, vertices),
    )
