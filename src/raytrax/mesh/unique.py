import numpy as np
from jaxtyping import Bool, Int


def _lex_less(a: Int[np.ndarray, "n k"], b: Int[np.ndarray, "n k"]) -> Bool[np.ndarray, " n"]:
    """Compare two arrays row by row in lexicographic order.

    Parameters
    ----------
    a
        The left operand.
    b
        The right operand, with the same shape as `a`.

    Returns
    -------
    less
        True for each row where `a` is less than `b`.
    """
    differs = a != b
    first = np.argmax(differs, axis=1)[:, None]
    return differs.any(axis=1) & (
        np.take_along_axis(a, first, axis=1)[:, 0] < np.take_along_axis(b, first, axis=1)[:, 0]
    )


def canonical_cycle(verts: Int[np.ndarray, "n k"]) -> tuple[Int[np.ndarray, "n k"], Int[np.ndarray, " n"]]:
    """Put the vertices of each face into a canonical winding.

    Two cells that share a face get the same key and opposite bits.

    Parameters
    ----------
    verts
        The vertex ids of each face, in the winding of one cell.

    Returns
    -------
    keys
        The vertex ids of each face in canonical winding. Each key is a valid
        winding of the face, so it is usable for face geometry.
    reversed_bit
        1 where the canonical winding is opposite to the input winding, else 0.
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
    """Find the unique rows of a 2D array, as `numpy.unique` with `axis=0` does.

    Parameters
    ----------
    keys
        The rows to make unique.
    return_index
        Whether to also return the index of the first instance of each unique row.
    return_inverse
        Whether to also return the unique row index of each input row.
    return_counts
        Whether to also return the number of instances of each unique row.

    Returns
    -------
    unique
        The unique rows, in lexicographic order.
    index
        The index in `keys` of each unique row. Only if `return_index`.
    inverse
        The index in `unique` of each row of `keys`. Only if `return_inverse`.
    counts
        The number of instances of each unique row. Only if `return_counts`.

    Raises
    ------
    ValueError
        If `keys` is not 2D.
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
