from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from raytrax.elements import ElementType, lookup
from raytrax.mesh.unique import canonical_cycle, lex_unique

# Untagged boundary. Tagged groups take -2, -3, ... (see boundaries.py).
EXTERIOR = -1

Block = tuple[str, np.ndarray]


@dataclass(frozen=True)
class MeshTopology:
    """Index-only description of a mixed-element mesh.

    Cells are padded to `max_faces` slots; unused slots are masked by
    `cell_face_valid` and carry an EXTERIOR neighbour so that a degenerate
    lookup escapes rather than walking somewhere wrong.
    """

    dim: int
    block_elements: tuple[ElementType, ...]
    cell_block: np.ndarray  # [ncell] which input block a cell came from
    cell_vertices: np.ndarray  # [ncell, max_cell_verts], -1 padded
    cell_face: np.ndarray  # [ncell, max_faces] global face id, -1 padded
    cell_face_sign: np.ndarray  # [ncell, max_faces] +1 outward, -1 inward, 0 padded
    cell_face_valid: np.ndarray  # [ncell, max_faces]
    cell_neighbour: np.ndarray  # [ncell, max_faces] cell id or negative sentinel
    face_vertices: np.ndarray  # [nface, max_face_verts] canonically wound, -1 padded
    face_num_verts: np.ndarray  # [nface]
    face_cells: np.ndarray  # [nface, 2]; column 0 sees the stored normal as outward

    @property
    def num_cells(self) -> int:
        return self.cell_face.shape[0]

    @property
    def num_faces(self) -> int:
        return self.face_cells.shape[0]

    @property
    def max_faces(self) -> int:
        return self.cell_face.shape[1]

    @property
    def is_boundary(self) -> np.ndarray:
        """[nface] mask of faces with only one adjacent cell."""
        return np.any(self.face_cells < 0, axis=-1)


def build(blocks: Sequence[Block]) -> MeshTopology:
    """Build topology from (element name, connectivity) blocks of one dimension."""
    if not blocks:
        raise ValueError("Cannot build a mesh from no cells")

    elements = tuple(lookup(name) for name, _ in blocks)
    dims = {element.dim for element in elements}
    if len(dims) != 1:
        raise ValueError(f"Mesh mixes {sorted(dims)}-dimensional elements; it must be purely 2D or 3D")
    (dim,) = dims

    max_faces = max(element.num_faces for element in elements)
    max_face_verts = max(element.max_face_verts for element in elements)
    max_cell_verts = max(element.num_vertices for element in elements)

    sizes = [len(connectivity) for _, connectivity in blocks]
    bounds = np.cumulative_sum(sizes, include_initial=True)
    num_cells = int(bounds[-1])

    cell_block = np.zeros(num_cells, dtype=np.int64)
    cell_vertices = np.full((num_cells, max_cell_verts), -1, dtype=np.int64)
    slot_verts = np.full((num_cells, max_faces, max_face_verts), -1, dtype=np.int64)
    slot_num_verts = np.zeros((num_cells, max_faces), dtype=np.int64)

    for index, ((_, connectivity), element) in enumerate(zip(blocks, elements, strict=True)):
        lo, hi = bounds[index], bounds[index + 1]
        if connectivity.shape[1] != element.num_vertices:
            raise ValueError(f"{element.name} block has {connectivity.shape[1]} vertices per cell")
        cell_block[lo:hi] = index
        cell_vertices[lo:hi, : element.num_vertices] = connectivity
        for slot, local in enumerate(element.faces):
            slot_verts[lo:hi, slot, : len(local)] = connectivity[:, local]
            slot_num_verts[lo:hi, slot] = len(local)

    cell_face_valid = slot_num_verts > 0
    cell_face = np.full((num_cells, max_faces), -1, dtype=np.int64)
    reversed_bit = np.zeros((num_cells, max_faces), dtype=np.int64)

    # Deduplicate faces separately per vertex count so the keys stay rectangular;
    # this is what lets tets, pyramids, wedges and hexes share one mesh.
    face_key_groups, face_count_groups = [], []
    for num_verts in np.unique(slot_num_verts[cell_face_valid]):
        selected = slot_num_verts == num_verts
        keys, bits = canonical_cycle(slot_verts[selected][:, :num_verts])
        unique_keys, inverse = lex_unique(keys, return_inverse=True)

        cell_face[selected] = inverse + sum(len(group) for group in face_key_groups)
        reversed_bit[selected] = bits

        padded = np.full((len(unique_keys), max_face_verts), -1, dtype=np.int64)
        padded[:, :num_verts] = unique_keys
        face_key_groups.append(padded)
        face_count_groups.append(np.full(len(unique_keys), num_verts, dtype=np.int64))

    face_vertices = np.concatenate(face_key_groups)
    face_num_verts = np.concatenate(face_count_groups)
    num_faces = len(face_vertices)

    # Knit the two directions together. A face's two cells wind it oppositely, so
    # the reversed bit picks the column, and the stored normal is outward from
    # whichever cell landed in column 0.
    cell_ids = np.broadcast_to(np.arange(num_cells)[:, None], cell_face.shape)
    face_cells = np.full((num_faces, 2), EXTERIOR, dtype=np.int64)
    face_cells[cell_face[cell_face_valid], reversed_bit[cell_face_valid]] = cell_ids[cell_face_valid]
    if not np.array_equal(
        face_cells[cell_face[cell_face_valid], reversed_bit[cell_face_valid]], cell_ids[cell_face_valid]
    ):
        raise ValueError("Invalid mesh: a face is shared by more than two cells, or two cells wind it identically")

    cell_neighbour = np.full((num_cells, max_faces), EXTERIOR, dtype=np.int64)
    cell_neighbour[cell_face_valid] = face_cells[cell_face[cell_face_valid], 1 - reversed_bit[cell_face_valid]]

    cell_face_sign = np.where(cell_face_valid, np.where(reversed_bit == 0, 1, -1), 0)

    return MeshTopology(
        dim=dim,
        block_elements=elements,
        cell_block=cell_block,
        cell_vertices=cell_vertices,
        cell_face=cell_face,
        cell_face_sign=cell_face_sign,
        cell_face_valid=cell_face_valid,
        cell_neighbour=cell_neighbour,
        face_vertices=face_vertices,
        face_num_verts=face_num_verts,
        face_cells=face_cells,
    )


def cell_face_vertices(topology: MeshTopology) -> np.ndarray:
    """[ncell, max_faces, max_face_verts] face vertices in each cell's own winding.

    Rebuilt on demand rather than stored: it is only needed while building
    geometry, and it is the largest array in the pipeline.
    """
    max_face_verts = topology.face_vertices.shape[1]
    verts = np.full((topology.num_cells, topology.max_faces, max_face_verts), -1, dtype=np.int64)

    for index, element in enumerate(topology.block_elements):
        rows = np.flatnonzero(topology.cell_block == index)
        connectivity = topology.cell_vertices[rows, : element.num_vertices]
        for slot, local in enumerate(element.faces):
            verts[rows, slot, : len(local)] = connectivity[:, local]
    return verts
