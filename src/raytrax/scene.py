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
    """A decomposition into simplices with weights summing to 1, for sampling.

    `apex` is the extra corner every simplex spans from, and is absent where the
    stored vertices already span them.
    """

    vertices: Int[Array, "n nsimplex ndim"]
    weight: Float[Array, "n nsimplex"]
    apex: Float[Array, "n ndim"] | None = None

    @property
    def width(self) -> int:
        """Spatial dimension. A simplex spans it with `ndim` vertices plus the apex."""
        return self.vertices.shape[-1]


class CellTable(eqx.Module):
    """Per-cell arrays, padded to a fixed number of face slots."""

    # Half-space form of every cell, already signed outward.
    normal: Float[Array, "ncell nslot ndim"]
    offset: Float[Array, "ncell nslot"]
    neighbour: Int[Array, "ncell nslot"]
    face: Int[Array, "ncell nslot"]

    interior: Simplices


class FaceTable(eqx.Module):
    """Per-face arrays. `normal` is outward from column 0 of `cells`."""

    normal: Float[Array, "nface ndim"]
    cells: Int[Array, "nface 2"]
    group: Int[Array, " nface"]

    surface: Simplices


class Scene(eqx.Module):
    """Device-side arrays: exactly what the ray march touches, nothing else.

    Grouped by leading index, so `cells.normal` and `faces.normal` cannot be
    confused. They are not the mesh: slots are padded and the signs are baked in.
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
        return self.cells.neighbour.shape[0]

    @property
    def num_faces(self) -> int:
        return self.faces.group.shape[0]


def build(topology: MeshTopology, geometry: MeshGeometry, groups: BoundaryGroups | None = None) -> Scene:
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
    """Raise if the mesh breaks an assumption the half-space march relies on."""
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
    """Whether a point lies inside a cell's half-spaces.

    A point left on a shared face by a crossing needs a tolerance to count as
    inside.
    """
    distance = jnp.einsum("...sd,...d->...s", scene.cells.normal[cell], point) - scene.cells.offset[cell]
    return jnp.all(distance <= tolerance, axis=-1)
