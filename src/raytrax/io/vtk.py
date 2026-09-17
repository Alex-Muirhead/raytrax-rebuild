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
    """A mesh read from a file, with the data necessary to write results back.

    Attributes
    ----------
    topology
        The connectivity of the cells of highest dimension.
    geometry
        The geometry of those cells and their faces.
    groups
        The boundary groups from the tagged lower-dimensional blocks.
    source
        The mesh as meshio read it.
    """

    topology: topology.MeshTopology
    geometry: geometry.MeshGeometry
    groups: boundaries.BoundaryGroups
    source: meshio.Mesh

    def cell_data(self, name: str) -> np.ndarray:
        """Read a cell field from the file.

        Parameters
        ----------
        name
            The name of the cell field.

        Returns
        -------
        values
            The value of the field for each cell, in the cell order of `topology`.

        Raises
        ------
        ValueError
            If the file has no cell field with that name.
        """
        if name not in self.source.cell_data_dict:
            raise ValueError(f"No cell field named {name!r}, have: {', '.join(self.source.cell_data_dict)}")
        blocks = self.source.cell_data_dict[name]
        elements = [element.name for element in self.topology.block_elements]
        return np.concatenate([np.asarray(blocks[name]) for name in elements])


def read(path: str | Path) -> Mesh:
    """Read a mesh from a file.

    Only the cells of the highest dimension become cells. Tagged blocks of
    lower dimension become boundary groups.

    Parameters
    ----------
    path
        The path of a mesh file in a format that meshio reads.

    Returns
    -------
    mesh
        The mesh, with its topology, geometry and boundary groups.

    Raises
    ------
    ValueError
        If the file contains no supported cell type.
    """
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
    """Find the face ids of each tagged face set in a meshio mesh.

    Parameters
    ----------
    mesh_topology
        The mesh that contains the faces.
    source
        The meshio mesh that holds the face blocks and their cell sets.

    Returns
    -------
    groups
        The face ids of each named set.
    """
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
    """Write a mesh to a file, with added cell fields.

    Parameters
    ----------
    path
        The path of the output file. Its extension sets the format.
    mesh
        The mesh to write, in its original cell blocks.
    cell_data
        The value of each field for each cell, in the cell order of `mesh.topology`.
    """
    # Annotated because meshio's `cells` parameter is an invariant list.
    blocks: list[meshio.CellBlock | tuple[str, ArrayLike]] = [
        (element.name, mesh.topology.cell_vertices[mesh.topology.cell_block == index, : element.num_vertices])
        for index, element in enumerate(mesh.topology.block_elements)
    ]
    split = {
        name: [np.asarray(values)[mesh.topology.cell_block == index] for index in range(len(blocks))]
        for name, values in cell_data.items()
    }
    meshio.write_points_cells(path, mesh.geometry.vertices, blocks, cell_data=split)
