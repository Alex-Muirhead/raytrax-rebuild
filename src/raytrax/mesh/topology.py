from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Int

from raytrax.elements import ElementType, lookup
from raytrax.mesh.unique import canonical_cycle, lex_unique

# Untagged boundary. Tagged groups take -2, -3, ... (see boundaries.py).
EXTERIOR = -1

Block = tuple[str, np.ndarray]


@dataclass(frozen=True)
class MeshTopology:
    """The connectivity of a mesh with mixed cell types, as indices only.

    Each cell has `max_faces` face slots. `cell_face_valid` shows the slots that
    the cell uses.

    Attributes
    ----------
    dim
        The dimension of the cells.
    block_elements
        The cell type of each input block.
    cell_block
        The input block index of each cell.
    cell_vertices
        The vertex ids of each cell, padded with -1.
    cell_faces
        The face id of each face slot, padded with -1.
    cell_face_signs
        +1 where the face normal points out of the cell, -1 where it points in,
        and 0 for padding.
    cell_neighbours
        The cell on the other side of each face slot, or a negative value at the boundary.
    face_vertices
        The vertex ids of each face in canonical winding, padded with -1.
    face_cells
        The two cells on each side of each face. The face normal points out of
        the cell in column 0. `EXTERIOR` replaces a cell that does not exist.
    """

    dim: int
    block_elements: tuple[ElementType, ...]
    cell_block: Int[np.ndarray, " ncell"]
    cell_vertices: Int[np.ndarray, "ncell max_cell_verts"]
    cell_faces: Int[np.ndarray, "ncell max_faces"]
    cell_face_signs: Int[np.ndarray, "ncell max_faces"]
    cell_neighbours: Int[np.ndarray, "ncell max_faces"]
    face_vertices: Int[np.ndarray, "nface max_face_verts"]
    face_cells: Int[np.ndarray, "nface 2"]

    @property
    def num_cells(self) -> int:
        """The number of cells."""
        return self.cell_faces.shape[0]

    @property
    def num_faces(self) -> int:
        """The number of faces."""
        return self.face_cells.shape[0]

    @property
    def max_faces(self) -> int:
        """The number of face slots of each cell."""
        return self.cell_faces.shape[1]

    @property
    def cell_face_valid(self) -> Bool[np.ndarray, "ncell max_faces"]:
        """The mask of face slots that each cell uses."""
        return self.cell_faces >= 0

    @property
    def face_num_verts(self) -> Int[np.ndarray, " nface"]:
        """The number of vertices of each face."""
        return np.sum(self.face_vertices >= 0, axis=1)

    @property
    def is_boundary(self) -> Bool[np.ndarray, " nface"]:
        """The mask of faces that have only one adjacent cell."""
        return np.any(self.face_cells < 0, axis=-1)


def build(blocks: Sequence[Block]) -> MeshTopology:
    """Make the topology of a mesh from its cell blocks.

    Parameters
    ----------
    blocks
        The meshio cell type name and the vertex ids of each cell, for each
        block. All blocks have cells of the same dimension.

    Returns
    -------
    topology
        The connectivity of the mesh.

    Raises
    ------
    ValueError
        If there are no blocks, the blocks have different dimensions, a block
        has the wrong number of vertices, or a face has more than two cells or
        two cells with the same winding.
    """
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
    cell_faces = np.full((num_cells, max_faces), -1, dtype=np.int64)
    reversed_bit = np.zeros((num_cells, max_faces), dtype=np.int64)

    # Deduplicate per vertex count, so the keys stay rectangular across a mesh
    # mixing tets, pyramids, wedges and hexes.
    face_key_groups: list[np.ndarray] = []
    for num_verts in np.unique(slot_num_verts[cell_face_valid]):
        selected = slot_num_verts == num_verts
        keys, bits = canonical_cycle(slot_verts[selected][:, :num_verts])
        unique_keys, inverse = lex_unique(keys, return_inverse=True)

        cell_faces[selected] = inverse + sum(len(group) for group in face_key_groups)
        reversed_bit[selected] = bits

        padded = np.full((len(unique_keys), max_face_verts), -1, dtype=np.int64)
        padded[:, :num_verts] = unique_keys
        face_key_groups.append(padded)

    face_vertices = np.concatenate(face_key_groups)
    num_faces = len(face_vertices)

    # A face's two cells wind it oppositely, so the reversed bit picks the
    # column, and the stored normal is outward from whichever lands in column 0.
    cell_ids = np.broadcast_to(np.arange(num_cells)[:, None], cell_faces.shape)
    face_cells = np.full((num_faces, 2), EXTERIOR, dtype=np.int64)
    face_cells[cell_faces[cell_face_valid], reversed_bit[cell_face_valid]] = cell_ids[cell_face_valid]
    if not np.array_equal(
        face_cells[cell_faces[cell_face_valid], reversed_bit[cell_face_valid]], cell_ids[cell_face_valid]
    ):
        raise ValueError("Invalid mesh: a face is shared by more than two cells, or two cells wind it identically")

    cell_neighbours = np.full((num_cells, max_faces), EXTERIOR, dtype=np.int64)
    cell_neighbours[cell_face_valid] = face_cells[cell_faces[cell_face_valid], 1 - reversed_bit[cell_face_valid]]

    cell_face_signs = np.where(cell_face_valid, np.where(reversed_bit == 0, 1, -1), 0)

    return MeshTopology(
        dim=dim,
        block_elements=elements,
        cell_block=cell_block,
        cell_vertices=cell_vertices,
        cell_faces=cell_faces,
        cell_face_signs=cell_face_signs,
        cell_neighbours=cell_neighbours,
        face_vertices=face_vertices,
        face_cells=face_cells,
    )


def cell_face_vertices(topology: MeshTopology) -> np.ndarray:
    """Give the vertex ids of each face slot, in the winding of its cell.

    Parameters
    ----------
    topology
        The connectivity of the mesh.

    Returns
    -------
    verts
        The vertex ids of each face slot of each cell, padded with -1.
    """
    # The largest array in the pipeline, so it is rebuilt on demand rather than stored.
    max_face_verts = topology.face_vertices.shape[1]
    verts = np.full((topology.num_cells, topology.max_faces, max_face_verts), -1, dtype=np.int64)

    for index, element in enumerate(topology.block_elements):
        rows = np.flatnonzero(topology.cell_block == index)
        connectivity = topology.cell_vertices[rows, : element.num_vertices]
        for slot, local in enumerate(element.faces):
            verts[rows, slot, : len(local)] = connectivity[:, local]
    return verts
