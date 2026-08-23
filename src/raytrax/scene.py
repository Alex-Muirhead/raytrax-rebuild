import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from raytrax.mesh.boundaries import DEFAULT_GROUP, BoundaryGroups
from raytrax.mesh.geometry import MeshGeometry
from raytrax.mesh.topology import EXTERIOR, MeshTopology

# A face slot a ray can never leave through, used to pad cells with few faces.
# `crossing` rejects it for free: a zero normal is never aligned with a tangent.
NULL_HALFSPACE = 0.0


class Scene(eqx.Module):
    """Flat device-side arrays: exactly what the ray march touches, nothing else.

    Cells are padded to a fixed number of face slots. Padded slots hold a null
    half-space, so mixed element types cost one rectangular array rather than a
    ragged one.
    """

    dim: int = eqx.field(static=True)

    # Half-space form of every cell, already signed outward. Materialised once
    # because the march gathers from it for every ray on every step.
    normal: Float[Array, "ncell nslot ndim"]
    offset: Float[Array, "ncell nslot"]
    neighbour: Int[Array, "ncell nslot"]
    face: Int[Array, "ncell nslot"]

    volume: Float[Array, " ncell"]
    centroid: Float[Array, "ncell ndim"]
    seed: Float[Array, "ncell ndim"]
    simplex_vertices: Int[Array, "ncell nsimplex ndim"]
    simplex_weight: Float[Array, "ncell nsimplex"]

    face_area: Float[Array, " nface"]
    face_normal: Float[Array, "nface ndim"]
    face_fan_vertices: Int[Array, "nface nfan ndim"]
    face_fan_weight: Float[Array, "nface nfan"]
    face_cells: Int[Array, "nface 2"]
    face_group: Int[Array, " nface"]

    vertices: Float[Array, "nvert ndim"]

    @property
    def num_cells(self) -> int:
        return self.volume.shape[0]

    @property
    def num_faces(self) -> int:
        return self.face_area.shape[0]


def build(topology: MeshTopology, geometry: MeshGeometry, groups: BoundaryGroups | None = None) -> Scene:
    slot = np.maximum(topology.cell_face, 0)
    valid = topology.cell_face_valid
    sign = topology.cell_face_sign

    normal = np.where(valid[..., None], geometry.faces.normal[slot] * sign[..., None], NULL_HALFSPACE)
    offset = np.where(valid, geometry.faces.offset[slot] * sign, NULL_HALFSPACE)
    # An unreachable slot still escapes rather than walking somewhere wrong.
    neighbour = np.where(valid, topology.cell_neighbour, EXTERIOR)
    face = np.where(valid, topology.cell_face, 0)

    face_group = groups.face_group if groups is not None else np.full(topology.num_faces, DEFAULT_GROUP)

    return Scene(
        dim=topology.dim,
        normal=jnp.asarray(normal),
        offset=jnp.asarray(offset),
        neighbour=jnp.asarray(neighbour),
        face=jnp.asarray(face),
        volume=jnp.asarray(geometry.cells.volume),
        centroid=jnp.asarray(geometry.cells.centroid),
        seed=jnp.asarray(geometry.cells.seed),
        simplex_vertices=jnp.asarray(np.maximum(geometry.cells.simplex_vertices, 0)),
        simplex_weight=jnp.asarray(geometry.cells.simplex_weight),
        face_area=jnp.asarray(geometry.faces.area),
        face_normal=jnp.asarray(geometry.faces.normal),
        face_fan_vertices=jnp.asarray(geometry.faces.fan_vertices),
        face_fan_weight=jnp.asarray(geometry.faces.fan_weight),
        face_cells=jnp.asarray(topology.face_cells),
        face_group=jnp.asarray(face_group),
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
    slot = np.maximum(topology.cell_face, 0)
    normal = geometry.faces.normal[slot] * topology.cell_face_sign[..., None]
    offset = geometry.faces.offset[slot] * topology.cell_face_sign
    distance = np.einsum("cfd,cd->cf", normal, geometry.cells.centroid) - offset
    outside = topology.cell_face_valid & (distance >= 0)
    if np.any(outside):
        raise ValueError(f"{np.sum(np.any(outside, axis=1))} cells do not contain their own centroid")


def contains(
    scene: Scene, cell: Int[Array, "..."], point: Float[Array, "... ndim"], tolerance: float = 0.0
) -> Bool[Array, "..."]:
    """Whether a point lies inside a cell's half-spaces.

    A point sitting exactly on a shared face -- which is where every crossing
    leaves a ray -- needs a tolerance to count as inside.
    """
    distance = jnp.einsum("...sd,...d->...s", scene.normal[cell], point) - scene.offset[cell]
    return jnp.all(distance <= tolerance, axis=-1)
