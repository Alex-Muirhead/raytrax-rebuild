from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from jaxtyping import Int

from raytrax.mesh.topology import MeshTopology
from raytrax.mesh.unique import canonical_cycle, lex_unique

# Group 0 is the implicit group of every untagged boundary face.
DEFAULT_GROUP = 0


@dataclass(frozen=True)
class BoundaryGroups:
    """The interaction group of each face.

    Attributes
    ----------
    names
        The name of each group, by group index. Group 0 has the empty name and
        holds every face that no group tags.
    face_group
        The group index of each face.
    """

    names: tuple[str, ...]
    face_group: Int[np.ndarray, " nface"]

    def index(self, name: str) -> int:
        """Find the group index of a named group.

        Parameters
        ----------
        name
            The name of the group.

        Returns
        -------
        index
            The group index.

        Raises
        ------
        ValueError
            If no group has that name.
        """
        if name not in self.names:
            raise ValueError(f"No boundary group named {name!r}, have: {', '.join(self.names[1:])}")
        return self.names.index(name)


def match_faces(topology: MeshTopology, face_vertices: Int[np.ndarray, "n k"]) -> Int[np.ndarray, " n"]:
    """Find the face ids of faces given by their vertex ids.

    Parameters
    ----------
    topology
        The mesh that contains the faces.
    face_vertices
        The vertex ids of each face, in any winding. All faces have the same
        number of vertices.

    Returns
    -------
    face_ids
        The face id of each queried face.

    Raises
    ------
    ValueError
        If a queried face is not in the mesh.
    """
    keys, _ = canonical_cycle(np.asarray(face_vertices))
    num_verts = keys.shape[1]

    rows = np.flatnonzero(topology.face_num_verts == num_verts)
    stored = topology.face_vertices[rows, :num_verts]

    # If every query already exists, the union has exactly the stored keys.
    unique, inverse = lex_unique(np.concatenate([stored, keys]), return_inverse=True)
    if len(unique) != len(stored):
        raise ValueError("Some queried faces do not exist in the mesh")

    to_face = np.empty(len(stored), dtype=np.int64)
    to_face[inverse[: len(stored)]] = rows
    return to_face[inverse[len(stored) :]]


def tag(topology: MeshTopology, groups: Mapping[str, np.ndarray]) -> BoundaryGroups:
    """Assign named groups of boundary faces to group indices.

    Parameters
    ----------
    topology
        The mesh that contains the faces.
    groups
        The face ids of each named group. The groups get indices from 1, in
        iteration order.

    Returns
    -------
    groups
        The group of each face, with untagged faces in group 0.

    Raises
    ------
    ValueError
        If a group contains an interior face, or a face is in more than one group.
    """
    face_group = np.full(topology.num_faces, DEFAULT_GROUP, dtype=np.int64)
    names = ("",)

    for name, face_ids in groups.items():
        face_ids = np.asarray(face_ids)
        if not np.all(topology.is_boundary[face_ids]):
            raise ValueError(f"Face group {name!r} contains interior faces")
        if np.any(face_group[face_ids] != DEFAULT_GROUP):
            raise ValueError(f"Face group {name!r} overlaps an earlier group")
        face_group[face_ids] = len(names)
        names += (name,)

    return BoundaryGroups(names=names, face_group=face_group)
