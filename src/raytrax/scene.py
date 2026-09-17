import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from raytrax.mesh.boundaries import DEFAULT_GROUP, BoundaryGroups
from raytrax.mesh.geometry import MeshGeometry
from raytrax.mesh.topology import EXTERIOR, MeshTopology

# Pads cells with few faces. `crossing` rejects it for free: a zero normal is
# never aligned with a tangent.
NULL_HALFSPACE = 0.0


class Simplices(eqx.Module):
    """A decomposition of each cell or face into simplices, for sampling.

    Attributes
    ----------
    vertices
        The mesh vertex ids of each simplex.
    weight
        The measure fraction of each simplex. Each row sums to 1.
    apex
        The position of a corner that all simplices of a row share, or None if
        `vertices` holds all corners.
    """

    vertices: Int[Array, "n nsimplex ndim"]
    weight: Float[Array, "n nsimplex"]
    apex: Float[Array, "n ndim"] | None = None

    @property
    def width(self) -> int:
        """The spatial dimension of the simplices."""
        return self.vertices.shape[-1]


class CellTable(eqx.Module):
    """The device arrays of each cell, padded to a fixed number of face slots.

    Attributes
    ----------
    normal
        The outward unit normal of each face slot. A padded slot has a zero normal.
    offset
        The plane constant of each face slot, such that `normal . x = offset` on the face.
    neighbour
        The cell on the other side of each face slot, or a negative value at the boundary.
    face
        The face id of each face slot.
    interior
        The simplex decomposition of each cell volume.
    """

    normal: Float[Array, "ncell nslot ndim"]
    offset: Float[Array, "ncell nslot"]
    neighbour: Int[Array, "ncell nslot"]
    face: Int[Array, "ncell nslot"]

    interior: Simplices


class FaceTable(eqx.Module):
    """The device arrays of each face.

    Attributes
    ----------
    normal
        The unit normal of each face, out of the cell in column 0 of `cells`.
    cells
        The two cells on each side of each face. `EXTERIOR` replaces a cell
        that does not exist.
    group
        The boundary group index of each face.
    surface
        The simplex decomposition of each face area.
    """

    normal: Float[Array, "nface ndim"]
    cells: Int[Array, "nface 2"]
    group: Int[Array, " nface"]

    surface: Simplices


class Scene(eqx.Module):
    """The device arrays that the ray march uses, and no other data.

    The arrays are not the mesh: the face slots have padding, and the normals
    point out of each cell.

    Attributes
    ----------
    cells
        The arrays of each cell.
    faces
        The arrays of each face.
    vertices
        The position of each mesh vertex.

    Raises
    ------
    ValueError
        If the tables do not all have the spatial dimension of `vertices`.
    """

    cells: CellTable
    faces: FaceTable
    vertices: Float[Array, "nvert ndim"]

    def __check_init__(self):
        # The shape hook checks ndim within each table but never between them,
        # and it only runs under test; this runs always.
        ndim = self.vertices.shape[-1]
        widths = {
            "cells.normal": self.cells.normal.shape[-1],
            "cells.interior": self.cells.interior.width,
            "faces.normal": self.faces.normal.shape[-1],
            "faces.surface": self.faces.surface.width,
        }
        wrong = {name: width for name, width in widths.items() if width != ndim}
        if wrong:
            raise ValueError(f"Vertices are {ndim}D, but {wrong} disagree")

    @property
    def num_cells(self) -> int:
        """The number of cells."""
        return self.cells.neighbour.shape[0]

    @property
    def num_faces(self) -> int:
        """The number of faces."""
        return self.faces.group.shape[0]


def build(topology: MeshTopology, geometry: MeshGeometry, groups: BoundaryGroups | None = None) -> Scene:
    """Put a host-side mesh into padded device arrays.

    Parameters
    ----------
    topology
        The connectivity of the mesh.
    geometry
        The geometry of the mesh. For an axisymmetric run, give the revolved geometry.
    groups
        The boundary groups of the mesh, or None to put all faces in group 0.

    Returns
    -------
    scene
        The device arrays.
    """
    slot = np.maximum(topology.cell_faces, 0)
    valid = topology.cell_face_valid
    sign = topology.cell_face_signs

    normal = np.where(valid[..., None], geometry.faces.normal[slot] * sign[..., None], NULL_HALFSPACE)
    offset = np.where(valid, geometry.faces.offset[slot] * sign, NULL_HALFSPACE)
    # An unreachable slot still escapes rather than walking somewhere wrong.
    neighbour = np.where(valid, topology.cell_neighbours, EXTERIOR)
    face = np.where(valid, topology.cell_faces, 0)

    face_group = groups.face_group if groups is not None else np.full(topology.num_faces, DEFAULT_GROUP)

    return Scene(
        cells=CellTable(
            normal=jnp.asarray(normal),
            offset=jnp.asarray(offset),
            neighbour=jnp.asarray(neighbour),
            face=jnp.asarray(face),
            interior=Simplices(
                vertices=jnp.asarray(np.maximum(geometry.cells.simplex_vertices, 0)),
                weight=jnp.asarray(geometry.cells.simplex_weight),
                apex=jnp.asarray(geometry.cells.seed),
            ),
        ),
        faces=FaceTable(
            normal=jnp.asarray(geometry.faces.normal),
            cells=jnp.asarray(topology.face_cells),
            group=jnp.asarray(face_group),
            surface=Simplices(
                vertices=jnp.asarray(geometry.faces.fan_vertices),
                weight=jnp.asarray(geometry.faces.fan_weight),
            ),
        ),
        vertices=jnp.asarray(geometry.vertices),
    )


def validate(topology: MeshTopology, geometry: MeshGeometry, *, flatness: float = 1e-8) -> None:
    """Make sure that the mesh agrees with the half-space model of the march.

    Parameters
    ----------
    topology
        The connectivity of the mesh.
    geometry
        The geometry of the mesh.
    flatness
        The largest permitted deviation of a face from its plane, relative to the face size.

    Raises
    ------
    ValueError
        If a cell has a volume that is not positive, a cell is not convex or is
        inverted, a face is warped, or a cell centroid is not inside its cell.
    """
    if np.any(geometry.cells.volume <= 0):
        raise ValueError(f"{np.sum(geometry.cells.volume <= 0)} cells have non-positive volume")
    if np.any(geometry.cells.simplex_weight < 0):
        raise ValueError(
            f"{np.sum(np.any(geometry.cells.simplex_weight < 0, axis=1))} cells are non-convex or inverted"
        )
    if np.any(geometry.faces.flatness > flatness):
        worst = geometry.faces.flatness.max()
        raise ValueError(f"{np.sum(geometry.faces.flatness > flatness)} faces are warped (worst {worst:.2e})")

    # Every cell must lie strictly inside its own half-spaces, or rays leak.
    slot = np.maximum(topology.cell_faces, 0)
    normal = geometry.faces.normal[slot] * topology.cell_face_signs[..., None]
    offset = geometry.faces.offset[slot] * topology.cell_face_signs
    distance = np.einsum("cfd,cd->cf", normal, geometry.cells.centroid) - offset
    outside = topology.cell_face_valid & (distance >= 0)
    if np.any(outside):
        raise ValueError(f"{np.sum(np.any(outside, axis=1))} cells do not contain their own centroid")


def contains(
    scene: Scene, cell: Int[Array, "..."], point: Float[Array, "... ndim"], tolerance: float = 0.0
) -> Bool[Array, "..."]:
    """Find if a point is inside the half-spaces of a cell.

    Parameters
    ----------
    scene
        The device arrays of the mesh.
    cell
        The cell index.
    point
        The position to examine.
    tolerance
        The largest distance outside a face that counts as inside. A point that
        a crossing puts on a face needs a positive tolerance.

    Returns
    -------
    inside
        True if the point is inside the cell.
    """
    distance = jnp.einsum("...sd,...d->...s", scene.cells.normal[cell], point) - scene.cells.offset[cell]
    return jnp.all(distance <= tolerance, axis=-1)
