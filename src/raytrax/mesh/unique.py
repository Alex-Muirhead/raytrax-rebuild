import numpy as np
from jaxtyping import Bool, Int


def _lex_less(a: Int[np.ndarray, "n k"], b: Int[np.ndarray, "n k"]) -> Bool[np.ndarray, " n"]:
    """Row-wise lexicographic comparison of two equally shaped 2D arrays."""
    differs = a != b
    first = np.argmax(differs, axis=1)[:, None]
    return differs.any(axis=1) & (
        np.take_along_axis(a, first, axis=1)[:, 0] < np.take_along_axis(b, first, axis=1)[:, 0]
    )


def canonical_cycle(verts: Int[np.ndarray, "n k"]) -> tuple[Int[np.ndarray, "n k"], Int[np.ndarray, " n"]]:
    """Canonically wind each face, returning (keys, reversed_bit).

    Two cells sharing a face present it with opposite winding, so they agree on
    the key and disagree on the bit. Permutation parity cannot express this: for
    a quad the two windings routinely sort with the *same* parity, which silently
    collapses both cells into one slot of the face->cell table.

    The canonical winding starts at the lowest vertex id and walks whichever
    direction is lexicographically smaller, so the key keeps a real winding and
    can be used directly for face geometry.
    """
    verts = np.asarray(verts)
    _, num_verts = verts.shape

    # An edge has no cycle to rotate; direction is the whole of its orientation.
    if num_verts == 2:
        reversed_bit = verts[:, 0] > verts[:, 1]
        return np.where(reversed_bit[:, None], verts[:, ::-1], verts), reversed_bit.astype(np.int64)

    start = np.argmin(verts, axis=1)[:, None]
    steps = np.arange(num_verts)[None, :]
    forward = np.take_along_axis(verts, (start + steps) % num_verts, axis=1)
    backward = np.take_along_axis(verts, (start - steps) % num_verts, axis=1)

    reversed_bit = _lex_less(backward, forward)
    return np.where(reversed_bit[:, None], backward, forward), reversed_bit.astype(np.int64)


def lex_unique(
    keys: Int[np.ndarray, "n k"],
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> tuple[np.ndarray, ...]:
    """A stripped-down version of numpy.unique that uses lexsort on keys.

    Returns the unique keys, followed by whichever of index, inverse and counts
    were asked for, in that order.
    """
    if len(keys.shape) != 2:
        raise ValueError("Keys must be 2 dimensional")

    sorting_order = np.lexsort(np.unstack(keys, axis=1))

    # If we represent "sorted_keys" as letters (for simplicity).
    #
    #          sorted_keys      :: a, b, b, c, c, d, e, f, f, g, ..., z, z
    #  np.roll(sorted_keys, +1) :: z, a, b, b, c, c, d, e, f, f, g, ..., z
    # ------------------------- :: ---------------------------------------
    #                 mask      :: T, T, F, T, F, T, T, T, F, T,    ..., F
    #     sorted_keys[mask]     :: a, b, c, d, e, f, g, ...
    #       sorted_key_ids      :: 0, 1, 1, 2, 2, 3, 4, 5, 5, 6, ..., n, n
    #
    # Ordering guarantees the first key is a first occurrence, so ids start at 0.
    sorted_keys = keys[sorting_order, :]
    is_first_instance = np.ones(sorting_order.size, dtype=bool)
    is_first_instance[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=-1)
    ret = (sorted_keys[is_first_instance],)

    if return_index:
        ret += (sorting_order[is_first_instance],)
    if return_inverse:
        sorted_key_ids = np.cumulative_sum(is_first_instance) - 1
        key_ids = np.empty_like(sorting_order)
        key_ids[sorting_order] = sorted_key_ids
        ret += (key_ids,)
    if return_counts:
        idx = np.concat(np.nonzero(is_first_instance) + ([is_first_instance.size],))
        ret += (np.diff(idx),)

    return ret
