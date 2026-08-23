import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np
from numpy.typing import ArrayLike

from raytrax.elements import ELEMENT_TYPES, FACE_ELEMENT_NAMES
from raytrax.mesh import boundaries, geometry, topology


@dataclass(frozen=True)
class Mesh:
    """A mesh read from file, with enough provenance to write results back."""

    topology: topology.MeshTopology
    geometry: geometry.MeshGeometry
    groups: boundaries.BoundaryGroups
    source: meshio.Mesh

    def cell_data(self, name: str) -> np.ndarray:
        """Read a cell field, concatenated in the same order as our cells."""
        if name not in self.source.cell_data_dict:
            raise ValueError(f"No cell field named {name!r}, have: {', '.join(self.source.cell_data_dict)}")
        blocks = self.source.cell_data_dict[name]
        elements = [element.name for element in self.topology.block_elements]
        return np.concatenate([np.asarray(blocks[name]) for name in elements])


def read(path: str | Path) -> Mesh:
    """Read a mesh, keeping only cells of its highest dimension."""
    # meshio chatters on stdout for some formats.
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        source = meshio.read(path)

    volume_blocks = [(block.type, block.data) for block in source.cells if block.type in ELEMENT_TYPES]
    if not volume_blocks:
        raise ValueError(f"{path} contains no supported cell types")

    dim = max(ELEMENT_TYPES[name].dim for name, _ in volume_blocks)
    blocks = [(name, data) for name, data in volume_blocks if ELEMENT_TYPES[name].dim == dim]

    mesh_topology = topology.build(blocks)
    mesh_geometry = geometry.build(mesh_topology, source.points)
    return Mesh(
        topology=mesh_topology,
        geometry=mesh_geometry,
        groups=boundaries.tag(mesh_topology, _face_groups(mesh_topology, source)),
        source=source,
    )


def _face_groups(mesh_topology: topology.MeshTopology, source: meshio.Mesh) -> dict[str, np.ndarray]:
    """Map tagged face blocks of a meshio mesh onto our face ids."""
    groups: dict[str, np.ndarray] = {}
    face_names = {FACE_ELEMENT_NAMES[count] for count in np.unique(mesh_topology.face_num_verts)}

    for name, sets in source.cell_sets_dict.items():
        if name == "gmsh:bounding_entities":
            continue
        for face_name in face_names & set(sets):
            faces = np.concatenate([block.data for block in source.cells if block.type == face_name])
            groups.setdefault(name, np.empty(0, dtype=np.int64))
            found = boundaries.match_faces(mesh_topology, faces[sets[face_name]])
            groups[name] = np.concatenate([groups[name], found])
    return groups


def write(path: str | Path, mesh: Mesh, cell_data: dict[str, np.ndarray]) -> None:
    """Write the mesh back out with extra cell fields, split by original block."""
    # Annotated because meshio's `cells` parameter is an invariant list, so a
    # list[tuple[str, ndarray]] is not assignable to it.
    blocks: list[meshio.CellBlock | tuple[str, ArrayLike]] = [
        (element.name, mesh.topology.cell_vertices[mesh.topology.cell_block == index, : element.num_vertices])
        for index, element in enumerate(mesh.topology.block_elements)
    ]
    split = {
        name: [np.asarray(values)[mesh.topology.cell_block == index] for index in range(len(blocks))]
        for name, values in cell_data.items()
    }
    meshio.write_points_cells(path, mesh.geometry.vertices, blocks, cell_data=split)
