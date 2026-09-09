import numpy as np
import pytest
from meshes import ALL_ELEMENTS, brick, make, mixed_3d
from pytest import approx

from raytrax import scene as scene_module
from raytrax.mesh import boundaries, geometry, topology
from raytrax.mesh.unique import canonical_cycle

CASES = {
    **{name: builder() for name, builder in ALL_ELEMENTS.items()},
    "mixed_3d": mixed_3d(),
    "brick_3d": brick((2, 2, 2)),
    "brick_2d": brick((3, 3), dim=2),
}


@pytest.fixture(params=CASES, ids=list(CASES))
def mesh(request):
    blocks, vertices = CASES[request.param]
    return make(blocks, vertices)


class TestCanonicalWinding:
    """A face must key identically from either side and record which side it
    came from. Permutation parity cannot do this for quads, which is why the
    canonical cycle exists."""

    @pytest.mark.parametrize("size", [2, 3, 4])
    def test_reversal_flips_the_bit(self, size):
        face = np.array([10, 4, 7, 2])[:size]
        keys, bits = canonical_cycle(np.stack([face, face[::-1]]))
        assert np.array_equal(keys[0], keys[1])
        assert bits[0] != bits[1]

    @pytest.mark.parametrize("size", [3, 4])
    def test_rotations_agree(self, size):
        # An edge is excluded: rotating a 2-cycle is the same as reversing it.
        face = np.array([10, 4, 7, 2])[:size]
        keys, bits = canonical_cycle(np.stack([np.roll(face, shift) for shift in range(size)]))
        assert len(np.unique(keys, axis=0)) == 1
        assert len(set(bits.tolist())) == 1


class TestTopology:
    def test_every_face_has_one_or_two_cells(self, mesh):
        mesh_topology, _, _ = mesh
        assert np.all(mesh_topology.face_cells.max(axis=1) >= 0)

    def test_neighbours_are_reciprocal(self, mesh):
        """Walking out of a cell and back in must return where it started."""
        mesh_topology, _, _ = mesh
        cell, slot = np.nonzero(mesh_topology.cell_neighbours >= 0)
        neighbour = mesh_topology.cell_neighbours[cell, slot]
        shared = mesh_topology.cell_faces[cell, slot]

        back = np.argmax(mesh_topology.cell_faces[neighbour] == shared[:, None], axis=1)
        assert np.all(mesh_topology.cell_neighbours[neighbour, back] == cell)

    def test_shared_faces_have_opposite_signs(self, mesh):
        """One cell sees the stored normal as outward, the other as inward."""
        mesh_topology, _, _ = mesh
        interior = np.flatnonzero(np.all(mesh_topology.face_cells >= 0, axis=1))
        left, right = mesh_topology.face_cells[interior].T

        def sign_of(cells, faces):
            slot = np.argmax(mesh_topology.cell_faces[cells] == faces[:, None], axis=1)
            return mesh_topology.cell_face_signs[cells, slot]

        assert np.all(sign_of(left, interior) == -sign_of(right, interior))

    def test_padded_slots_are_inert(self, mesh):
        mesh_topology, _, scene = mesh
        padded = ~mesh_topology.cell_face_valid
        assert np.all(np.asarray(scene.cells.normal)[padded] == 0)
        assert np.all(np.asarray(scene.cells.neighbour)[padded] == topology.EXTERIOR)

    def test_tables_must_agree_on_dimension(self, mesh):
        """The cell, face and vertex tables are built separately, so nothing but
        `__check_init__` stops a 2D table being paired with a 3D one."""
        _, _, scene = mesh
        with pytest.raises(ValueError, match="disagree"):
            scene_module.Scene(scene.cells, scene.faces, scene.vertices[:, :-1])


class TestGeometry:
    def test_face_normals_are_unit(self, mesh):
        _, mesh_geometry, _ = mesh
        assert np.linalg.norm(mesh_geometry.faces.normal, axis=-1) == approx(1.0)

    def test_simplices_partition_each_cell(self, mesh):
        _, mesh_geometry, _ = mesh
        assert mesh_geometry.cells.simplex_weight.sum(axis=1) == approx(1.0)
        assert np.all(mesh_geometry.cells.simplex_weight >= 0)

    def test_fans_partition_each_face(self, mesh):
        _, mesh_geometry, _ = mesh
        assert mesh_geometry.faces.fan_weight.sum(axis=1) == approx(1.0)

    def test_cells_contain_their_own_centroid(self, mesh):
        """The half-space form is only valid if every cell encloses itself."""
        mesh_topology, mesh_geometry, _ = mesh
        scene_module.validate(mesh_topology, mesh_geometry)

    def test_divergence_theorem_matches_volume(self, mesh):
        """Independent check: sum(outward offset * area) / dim is the volume."""
        mesh_topology, mesh_geometry, _ = mesh
        slot = np.maximum(mesh_topology.cell_faces, 0)
        signed = mesh_topology.cell_face_signs * mesh_geometry.faces.offset[slot] * mesh_geometry.faces.area[slot]
        assert signed.sum(axis=1) / mesh_topology.dim == approx(mesh_geometry.cells.volume)


class TestKnownShapes:
    """Exact values for shapes whose volume and centroid are known by hand."""

    @pytest.mark.parametrize(
        "name,volume,centroid",
        [
            ("hexahedron", 1.0, [0.5, 0.5, 0.5]),
            ("tetra", 1 / 6, [0.25, 0.25, 0.25]),
            ("wedge", 0.5, [1 / 3, 1 / 3, 0.5]),
            ("pyramid", 1 / 3, [0.5, 0.5, 0.25]),
            ("quad", 1.0, [0.5, 0.5]),
            ("triangle", 0.5, [2 / 3, 1 / 3]),
        ],
    )
    def test_volume_and_centroid(self, name, volume, centroid):
        blocks, vertices = ALL_ELEMENTS[name]()
        _, mesh_geometry, _ = make(blocks, vertices)
        assert float(mesh_geometry.cells.volume[0]) == approx(volume)
        assert mesh_geometry.cells.centroid[0] == approx(centroid)


class TestBoundaries:
    def test_match_faces_ignores_winding(self):
        blocks, vertices = brick((2, 1, 1))
        mesh_topology, _, _ = make(blocks, vertices)
        faces = mesh_topology.face_vertices[mesh_topology.face_num_verts == 4]
        forward = boundaries.match_faces(mesh_topology, faces)
        assert np.array_equal(forward, boundaries.match_faces(mesh_topology, faces[:, ::-1]))

    def test_interior_faces_cannot_be_tagged(self):
        blocks, _ = brick((2, 1, 1))
        mesh_topology = topology.build(blocks)
        interior = np.flatnonzero(~mesh_topology.is_boundary)
        with pytest.raises(ValueError, match="interior"):
            boundaries.tag(mesh_topology, {"bad": interior})


class TestWarpedFaces:
    def test_warped_quad_is_rejected(self):
        """A non-planar face breaks the half-space model, so it must not pass."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.5]], dtype=float
        )
        mesh_topology = topology.build([("hexahedron", np.arange(8)[None, :])])
        mesh_geometry = geometry.build(mesh_topology, vertices)
        with pytest.raises(ValueError, match="warped"):
            scene_module.validate(mesh_topology, mesh_geometry)


class TestRevolve:
    """A revolved 2D mesh must measure the ring, not the (x, r) slice."""

    def annulus(self, inner=1.0, outer=2.0, length=1.0):
        """One quad spanning `inner` to `outer` in radius: a hollow cylinder."""
        vertices = np.array([[0, inner], [length, inner], [length, outer], [0, outer]], dtype=float)
        blocks = [("quad", np.arange(4)[None, :])]
        mesh_topology = topology.build(blocks)
        return mesh_topology, geometry.build(mesh_topology, vertices)

    def test_cell_volume_is_the_ring_it_sweeps(self):
        """Exactly the hollow cylinder: pi (outer^2 - inner^2) length."""
        _, mesh_geometry = self.annulus()
        revolved = geometry.revolve(mesh_geometry)
        assert float(revolved.cells.volume[0]) == approx(np.pi * (2.0**2 - 1.0**2) * 1.0)

    def test_face_area_is_the_surface_it_sweeps(self):
        """An axial edge at radius R sweeps 2 pi R L; a radial one sweeps an annulus."""
        _, mesh_geometry = self.annulus()
        revolved = geometry.revolve(mesh_geometry)
        centroid = mesh_geometry.faces.centroid
        for radius in (1.0, 2.0):
            face = np.flatnonzero(np.isclose(centroid[:, 1], radius))[0]
            assert float(revolved.faces.area[face]) == approx(2 * np.pi * radius * 1.0)
        radial_face = np.flatnonzero(np.isclose(centroid[:, 1], 1.5))[0]
        assert float(revolved.faces.area[radial_face]) == approx(np.pi * (2.0**2 - 1.0**2))

    def test_pappus_holds_for_a_triangle(self):
        """Not just boxes: the weighting is exact for any polygon."""
        vertices = np.array([[0, 1], [2, 1], [0, 4]], dtype=float)
        mesh_topology = topology.build([("triangle", np.arange(3)[None, :])])
        mesh_geometry = geometry.build(mesh_topology, vertices)
        revolved = geometry.revolve(mesh_geometry)
        area = 0.5 * 2 * 3
        assert float(revolved.cells.volume[0]) == approx(2 * np.pi * (1 + 1 + 4) / 3 * area)

    def test_weights_stay_normalised(self):
        blocks, vertices = brick((3, 3), dim=2)
        mesh_topology = topology.build(blocks)
        mesh_geometry = geometry.revolve(geometry.build(mesh_topology, vertices))
        assert np.allclose(mesh_geometry.cells.simplex_weight.sum(axis=1), 1.0)
        assert np.allclose(mesh_geometry.faces.fan_weight.sum(axis=1), 1.0)

    def test_a_face_on_the_axis_sweeps_nothing(self):
        blocks, vertices = brick((2, 2), dim=2)
        mesh_topology = topology.build(blocks)
        mesh_geometry = geometry.revolve(geometry.build(mesh_topology, vertices))
        on_axis = np.isclose(geometry.build(mesh_topology, vertices).faces.centroid[:, 1], 0.0)
        assert np.all(mesh_geometry.faces.area[on_axis] == 0.0)
        assert np.all(mesh_geometry.faces.area[~on_axis] > 0.0)

    def test_negative_radius_is_rejected(self):
        blocks, vertices = brick((2, 2), dim=2)
        mesh_topology = topology.build(blocks)
        mesh_geometry = geometry.build(mesh_topology, vertices - np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="negative radius"):
            geometry.revolve(mesh_geometry)

    def test_a_three_dimensional_mesh_is_rejected(self):
        blocks, vertices = brick((2, 2, 2))
        mesh_topology = topology.build(blocks)
        with pytest.raises(ValueError, match="2D"):
            geometry.revolve(geometry.build(mesh_topology, vertices))
