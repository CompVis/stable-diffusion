"""Analytical transformations from raw image moments to central moments.

The expressions for the 2D central moments of order <=2 are often given in
textbooks. Expressions for higher orders and dimensions were generated in SymPy
using ``tools/precompute/moments_sympy.py`` in the GitHub repository.

"""
import itertools
import math

import numpy as np


def _moments_raw_to_central_fast(moments_raw):
    """Analytical formulae for 2D and 3D central moments of order < 4.

    `moments_raw_to_central` will automatically call this function when
    ndim < 4 and order < 4.

    Parameters
    ----------
    moments_raw : ndarray
        The raw moments.

    Returns
    -------
    moments_central : ndarray
        The central moments.
    """
    ndim = moments_raw.ndim
    order = moments_raw.shape[0] - 1
    float_dtype = moments_raw.dtype
    # convert to float64 during the computation for better accuracy
    moments_raw = moments_raw.astype(np.float64, copy=False)
    moments_central = np.zeros_like(moments_raw)
    if order >= 4 or ndim not in [2, 3]:
        raise ValueError(
            "This function only supports 2D or 3D moments of order < 4."
        )
    m = moments_raw
    if ndim == 2:
        cx = m[1, 0] / m[0, 0]
        cy = m[0, 1] / m[0, 0]
        moments_central[0, 0] = m[0, 0]
        # Note: 1st order moments are both 0
        if order > 1:
            # 2nd order moments
            moments_central[1, 1] = m[1, 1] - cx*m[0, 1]
            moments_central[2, 0] = m[2, 0] - cx*m[1, 0]
            moments_central[0, 2] = m[0, 2] - cy*m[0, 1]
        if order > 2:
            # 3rd order moments
            moments_central[2, 1] = (m[2, 1] - 2*cx*m[1, 1] - cy*m[2, 0]
                                     + cx**2*m[0, 1] + cy*cx*m[1, 0])
            moments_central[1, 2] = (m[1, 2] - 2*cy*m[1, 1] - cx*m[0, 2]
                                     + 2*cy*cx*m[0, 1])
            moments_central[3, 0] = m[3, 0] - 3*cx*m[2, 0] + 2*cx**2*m[1, 0]
            moments_central[0, 3] = m[0, 3] - 3*cy*m[0, 2] + 2*cy**2*m[0, 1]
    else:
        # 3D case
        cx = m[1, 0, 0] / m[0, 0, 0]
        cy = m[0, 1, 0] / m[0, 0, 0]
        cz = m[0, 0, 1] / m[0, 0, 0]
        moments_central[0, 0, 0] = m[0, 0, 0]
        # Note: all first order moments are 0
        if order > 1:
            # 2nd order moments
            moments_central[0, 0, 2] = -cz*m[0, 0, 1] + m[0, 0, 2]
            moments_central[0, 1, 1] = -cy*m[0, 0, 1] + m[0, 1, 1]
            moments_central[0, 2, 0] = -cy*m[0, 1, 0] + m[0, 2, 0]
            moments_central[1, 0, 1] = -cx*m[0, 0, 1] + m[1, 0, 1]
            moments_central[1, 1, 0] = -cx*m[0, 1, 0] + m[1, 1, 0]
            moments_central[2, 0, 0] = -cx*m[1, 0, 0] + m[2, 0, 0]
        if order > 2:
            # 3rd order moments
            moments_central[0, 0, 3] = (2*cz**2*m[0, 0, 1]
                                        - 3*cz*m[0, 0, 2]
                                        + m[0, 0, 3])
            moments_central[0, 1, 2] = (-cy*m[0, 0, 2]
                                        + 2*cz*(cy*m[0, 0, 1] - m[0, 1, 1])
                                        + m[0, 1, 2])
            moments_central[0, 2, 1] = (cy**2*m[0, 0, 1] - 2*cy*m[0, 1, 1]
                                        + cz*(cy*m[0, 1, 0] - m[0, 2, 0])
                                        + m[0, 2, 1])
            moments_central[0, 3, 0] = (2*cy**2*m[0, 1, 0]
                                        - 3*cy*m[0, 2, 0]
                                        + m[0, 3, 0])
            moments_central[1, 0, 2] = (-cx*m[0, 0, 2]
                                        + 2*cz*(cx*m[0, 0, 1] - m[1, 0, 1])
                                        + m[1, 0, 2])
            moments_central[1, 1, 1] = (-cx*m[0, 1, 1]
                                        + cy*(cx*m[0, 0, 1] - m[1, 0, 1])
                                        + cz*(cx*m[0, 1, 0] - m[1, 1, 0])
                                        + m[1, 1, 1])
            moments_central[1, 2, 0] = (-cx*m[0, 2, 0]
                                        - 2*cy*(-cx*m[0, 1, 0] + m[1, 1, 0])
                                        + m[1, 2, 0])
            moments_central[2, 0, 1] = (cx**2*m[0, 0, 1]
                                        - 2*cx*m[1, 0, 1]
                                        + cz*(cx*m[1, 0, 0] - m[2, 0, 0])
                                        + m[2, 0, 1])
            moments_central[2, 1, 0] = (cx**2*m[0, 1, 0]
                                        - 2*cx*m[1, 1, 0]
                                        + cy*(cx*m[1, 0, 0] - m[2, 0, 0])
                                        + m[2, 1, 0])
            moments_central[3, 0, 0] = (2*cx**2*m[1, 0, 0]
                                        - 3*cx*m[2, 0, 0]
                                        + m[3, 0, 0])

    return moments_central.astype(float_dtype, copy=False)


def moments_raw_to_central(moments_raw):
    ndim = moments_raw.ndim
    order = moments_raw.shape[0] - 1
    if ndim in [2, 3] and order < 4:
        return _moments_raw_to_central_fast(moments_raw)

    moments_central = np.zeros_like(moments_raw)
    m = moments_raw
    # centers as computed in centroid above
    centers = tuple(m[tuple(np.eye(ndim, dtype=int))] / m[(0,)*ndim])

    if ndim == 2:
        # This is the general 2D formula from
        # https://en.wikipedia.org/wiki/Image_moment#Central_moments
        for p in range(order + 1):
            for q in range(order + 1):
                if p + q > order:
                    continue
                for i in range(p + 1):
                    term1 = math.comb(p, i)
                    term1 *= (-centers[0]) ** (p - i)
                    for j in range(q + 1):
                        term2 = math.comb(q, j)
                        term2 *= (-centers[1]) ** (q - j)
                        moments_central[p, q] += term1*term2*m[i, j]
        return moments_central

    # The nested loops below are an n-dimensional extension of the 2D formula
    # given at https://en.wikipedia.org/wiki/Image_moment#Central_moments

    # iterate over all [0, order] (inclusive) on each axis
    for orders in itertools.product(*((range(order + 1),) * ndim)):
        # `orders` here is the index into the `moments_central` output array
        if sum(orders) > order:
            # skip any moment that is higher than the requested order
            continue
        # loop over terms from `m` contributing to `moments_central[orders]`
        for idxs in itertools.product(*[range(o + 1) for o in orders]):
            val = m[idxs]
            for i_order, c, idx in zip(orders, centers, idxs):
                val *= math.comb(i_order, idx)
                val *= (-c) ** (i_order - idx)
            moments_central[orders] += val

    return moments_central
