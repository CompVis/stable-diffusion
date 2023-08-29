__all__ = ['slice_along_axes']


def slice_along_axes(image, slices, axes=None, copy=False):
    """Slice an image along given axes.

    Parameters
    ----------
    image : ndarray
        Input image.
    slices : list of 2-tuple (a, b) where a < b.
        For each axis in `axes`, a corresponding 2-tuple
        ``(min_val, max_val)`` to slice with (as with Python slices,
        ``max_val`` is non-inclusive).
    axes : int or tuple, optional
        Axes corresponding to the limits given in `slices`. If None,
        axes are in ascending order, up to the length of `slices`.
    copy : bool, optional
        If True, ensure that the output is not a view of `image`.

    Returns
    -------
    out : ndarray
        The region of `image` corresponding to the given slices and axes.

    Examples
    --------
    >>> from skimage import data
    >>> img = data.camera()
    >>> img.shape
    (512, 512)
    >>> cropped_img = slice_along_axes(img, [(0, 100)])
    >>> cropped_img.shape
    (100, 512)
    >>> cropped_img = slice_along_axes(img, [(0, 100), (0, 100)])
    >>> cropped_img.shape
    (100, 100)
    >>> cropped_img = slice_along_axes(img, [(0, 100), (0, 75)], axes=[1, 0])
    >>> cropped_img.shape
    (75, 100)
    """

    # empty length of bounding box detected on None
    if not slices:
        return image

    if axes is None:
        axes = list(range(image.ndim))
        if len(axes) < len(slices):
            raise ValueError("More `slices` than available axes")

    elif len(axes) != len(slices):
        raise ValueError("`axes` and `slices` must have equal length")

    if len(axes) != len(set(axes)):
        raise ValueError("`axes` must be unique")

    if not all(a >= 0 and a < image.ndim for a in axes):
        raise ValueError(f"axes {axes} out of range; image has only "
                         f"{image.ndim} dimensions")

    _slices = [slice(None),] * image.ndim
    for (a, b), ax in zip(slices, axes):
        if a < 0:
            a %= image.shape[ax]
        if b < 0:
            b %= image.shape[ax]
        if a > b:
            raise ValueError(
                f"Invalid slice ({a}, {b}): must be ordered `(min_val, max_val)`"
            )
        if a < 0 or b > image.shape[ax]:
            raise ValueError(f"Invalid slice ({a}, {b}) for image with dimensions {image.shape}")
        _slices[ax] = slice(a, b)

    image_slice = image[tuple(_slices)]

    if copy and image_slice.base is not None:
        image_slice = image_slice.copy()

    return image_slice
