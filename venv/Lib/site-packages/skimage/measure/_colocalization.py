import numpy as np
from scipy.stats import pearsonr

from .._shared.utils import check_shape_equality, as_binary_ndarray

__all__ = ['pearson_corr_coeff',
           'manders_coloc_coeff',
           'manders_overlap_coeff',
           'intersection_coeff',
           ]


def pearson_corr_coeff(image0, image1, mask=None):
    r"""Calculate Pearson's Correlation Coefficient between pixel intensities
    in channels.

    Parameters
    ----------
    image0 : (M, N) ndarray
        Image of channel A.
    image1 : (M, N) ndarray
        Image of channel 2 to be correlated with channel B.
        Must have same dimensions as `image0`.
    mask : (M, N) ndarray of dtype bool, optional
        Only `image0` and `image1` pixels within this region of interest mask
        are included in the calculation. Must have same dimensions as `image0`.

    Returns
    -------
    pcc : float
        Pearson's correlation coefficient of the pixel intensities between
        the two images, within the mask if provided.
    p-value : float
        Two-tailed p-value.

    Notes
    -----
    Pearson's Correlation Coefficient (PCC) measures the linear correlation
    between the pixel intensities of the two images. Its value ranges from -1
    for perfect linear anti-correlation to +1 for perfect linear correlation.
    The calculation of the p-value assumes that the intensities of pixels in
    each input image are normally distributed.

    Scipy's implementation of Pearson's correlation coefficient is used. Please
    refer to it for further information and caveats [1]_.

    .. math::
        r = \frac{\sum (A_i - m_A_i) (B_i - m_B_i)}
        {\sqrt{\sum (A_i - m_A_i)^2 \sum (B_i - m_B_i)^2}}

    where
        :math:`A_i` is the value of the :math:`i^{th}` pixel in `image0`
        :math:`B_i` is the value of the :math:`i^{th}` pixel in `image1`,
        :math:`m_A_i` is the mean of the pixel values in `image0`
        :math:`m_B_i` is the mean of the pixel values in `image1`

    A low PCC value does not necessarily mean that there is no correlation
    between the two channel intensities, just that there is no linear
    correlation. You may wish to plot the pixel intensities of each of the two
    channels in a 2D scatterplot and use Spearman's rank correlation if a
    non-linear correlation is visually identified [2]_. Also consider if you
    are interested in correlation or co-occurence, in which case a method
    involving segmentation masks (e.g. MCC or intersection coefficient) may be
    more suitable [3]_ [4]_.

    Providing the mask of only relevant sections of the image (e.g., cells, or
    particular cellular compartments) and removing noise is important as the
    PCC is sensitive to these measures [3]_ [4]_.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html  # noqa
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html  # noqa
    .. [3] Dunn, K. W., Kamocka, M. M., & McDonald, J. H. (2011). A practical
           guide to evaluating colocalization in biological microscopy.
           American journal of physiology. Cell physiology, 300(4), C723–C742.
           https://doi.org/10.1152/ajpcell.00462.2010
    .. [4] Bolte, S. and Cordelières, F.P. (2006), A guided tour into
           subcellular colocalization analysis in light microscopy. Journal of
           Microscopy, 224: 213-232.
           https://doi.org/10.1111/j.1365-2818.2006.01706.x
    """
    image0 = np.asarray(image0)
    image1 = np.asarray(image1)
    if mask is not None:
        mask = as_binary_ndarray(mask, variable_name="mask")
        check_shape_equality(image0, image1, mask)
        image0 = image0[mask]
        image1 = image1[mask]
    else:
        check_shape_equality(image0, image1)
        # scipy pearsonr function only takes flattened arrays
        image0 = image0.reshape(-1)
        image1 = image1.reshape(-1)

    return pearsonr(image0, image1)


def manders_coloc_coeff(image0, image1_mask, mask=None):
    r"""Manders' colocalization coefficient between two channels.

    Parameters
    ----------
    image0 : (M, N) ndarray
        Image of channel A. All pixel values should be non-negative.
    image1_mask : (M, N) ndarray of dtype bool
        Binary mask with segmented regions of interest in channel B.
        Must have same dimensions as `image0`.
    mask : (M, N) ndarray of dtype bool, optional
        Only `image0` pixel values within this region of interest mask are
        included in the calculation.
        Must have same dimensions as `image0`.

    Returns
    -------
    mcc : float
        Manders' colocalization coefficient.

    Notes
    -----
    Manders' Colocalization Coefficient (MCC) is the fraction of total
    intensity of a certain channel (channel A) that is within the segmented
    region of a second channel (channel B) [1]_. It ranges from 0 for no
    colocalisation to 1 for complete colocalization. It is also referred to
    as M1 and M2.

    MCC is commonly used to measure the colocalization of a particular protein
    in a subceullar compartment. Typically a segmentation mask for channel B
    is generated by setting a threshold that the pixel values must be above
    to be included in the MCC calculation. In this implementation,
    the channel B mask is provided as the argument `image1_mask`, allowing
    the exact segmentation method to be decided by the user beforehand.

    The implemented equation is:

    .. math::
        r = \frac{\sum A_{i,coloc}}{\sum A_i}

    where
        :math:`A_i` is the value of the :math:`i^{th}` pixel in `image0`
        :math:`A_{i,coloc} = A_i` if :math:`Bmask_i > 0`
        :math:`Bmask_i` is the value of the :math:`i^{th}` pixel in
        `mask`

    MCC is sensitive to noise, with diffuse signal in the first channel
    inflating its value. Images should be processed to remove out of focus and
    background light before the MCC is calculated [2]_.

    References
    ----------
    .. [1] Manders, E.M.M., Verbeek, F.J. and Aten, J.A. (1993), Measurement of
           co-localization of objects in dual-colour confocal images. Journal
           of Microscopy, 169: 375-382.
           https://doi.org/10.1111/j.1365-2818.1993.tb03313.x
           https://imagej.net/media/manders.pdf
    .. [2] Dunn, K. W., Kamocka, M. M., & McDonald, J. H. (2011). A practical
           guide to evaluating colocalization in biological microscopy.
           American journal of physiology. Cell physiology, 300(4), C723–C742.
           https://doi.org/10.1152/ajpcell.00462.2010

    """
    image0 = np.asarray(image0)
    image1_mask = as_binary_ndarray(image1_mask, variable_name="image1_mask")
    if mask is not None:
        mask = as_binary_ndarray(mask, variable_name="mask")
        check_shape_equality(image0, image1_mask, mask)
        image0 = image0[mask]
        image1_mask = image1_mask[mask]
    else:
        check_shape_equality(image0, image1_mask)
    # check non-negative image
    if image0.min() < 0:
        raise ValueError("image contains negative values")

    sum = np.sum(image0)
    if (sum == 0):
        return 0
    return np.sum(image0 * image1_mask) / sum


def manders_overlap_coeff(image0, image1, mask=None):
    r"""Manders' overlap coefficient

    Parameters
    ----------
    image0 : (M, N) ndarray
        Image of channel A. All pixel values should be non-negative.
    image1 : (M, N) ndarray
        Image of channel B. All pixel values should be non-negative.
        Must have same dimensions as `image0`
    mask : (M, N) ndarray of dtype bool, optional
        Only `image0` and `image1` pixel values within this region of interest
        mask are included in the calculation.
        Must have ♣same dimensions as `image0`.

    Returns
    -------
    moc: float
        Manders' Overlap Coefficient of pixel intensities between the two
        images.

    Notes
    -----
    Manders' Overlap Coefficient (MOC) is given by the equation [1]_:

    .. math::
        r = \frac{\sum A_i B_i}{\sqrt{\sum A_i^2 \sum B_i^2}}

    where
        :math:`A_i` is the value of the :math:`i^{th}` pixel in `image0`
        :math:`B_i` is the value of the :math:`i^{th}` pixel in `image1`

    It ranges between 0 for no colocalization and 1 for complete colocalization
    of all pixels.

    MOC does not take into account pixel intensities, just the fraction of
    pixels that have positive values for both channels[2]_ [3]_. Its usefulness
    has been criticized as it changes in response to differences in both
    co-occurence and correlation and so a particular MOC value could indicate
    a wide range of colocalization patterns [4]_ [5]_.

    References
    ----------
    .. [1] Manders, E.M.M., Verbeek, F.J. and Aten, J.A. (1993), Measurement of
           co-localization of objects in dual-colour confocal images. Journal
           of Microscopy, 169: 375-382.
           https://doi.org/10.1111/j.1365-2818.1993.tb03313.x
           https://imagej.net/media/manders.pdf
    .. [2] Dunn, K. W., Kamocka, M. M., & McDonald, J. H. (2011). A practical
           guide to evaluating colocalization in biological microscopy.
           American journal of physiology. Cell physiology, 300(4), C723–C742.
           https://doi.org/10.1152/ajpcell.00462.2010
    .. [3] Bolte, S. and Cordelières, F.P. (2006), A guided tour into
           subcellular colocalization analysis in light microscopy. Journal of
           Microscopy, 224: 213-232.
           https://doi.org/10.1111/j.1365-2818.2006.01
    .. [4] Adler J, Parmryd I. (2010), Quantifying colocalization by
           correlation: the Pearson correlation coefficient is
           superior to the Mander's overlap coefficient. Cytometry A.
           Aug;77(8):733-42.https://doi.org/10.1002/cyto.a.20896
    .. [5] Adler, J, Parmryd, I. Quantifying colocalization: The case for
           discarding the Manders overlap coefficient. Cytometry. 2021; 99:
           910– 920. https://doi.org/10.1002/cyto.a.24336

    """
    image0 = np.asarray(image0)
    image1 = np.asarray(image1)
    if mask is not None:
        mask = as_binary_ndarray(mask, variable_name="mask")
        check_shape_equality(image0, image1, mask)
        image0 = image0[mask]
        image1 = image1[mask]
    else:
        check_shape_equality(image0, image1)

    # check non-negative image
    if image0.min() < 0:
        raise ValueError("image0 contains negative values")
    if image1.min() < 0:
        raise ValueError("image1 contains negative values")

    denom = (np.sum(np.square(image0)) * (np.sum(np.square(image1)))) ** 0.5
    return np.sum(np.multiply(image0, image1)) / denom


def intersection_coeff(image0_mask, image1_mask, mask=None):
    r"""Fraction of a channel's segmented binary mask that overlaps with a
    second channel's segmented binary mask.

    Parameters
    ----------
    image0_mask : (M, N) ndarray of dtype bool
        Image mask of channel A.
    image1_mask : (M, N) ndarray of dtype bool
        Image mask of channel B.
        Must have same dimensions as `image0_mask`.
    mask : (M, N) ndarray of dtype bool, optional
        Only `image0_mask` and `image1_mask` pixels within this region of
        interest
        mask are included in the calculation.
        Must have same dimensions as `image0_mask`.

    Returns
    -------
    Intersection coefficient, float
        Fraction of `image0_mask` that overlaps with `image1_mask`.

    """
    image0_mask = as_binary_ndarray(image0_mask, variable_name="image0_mask")
    image1_mask = as_binary_ndarray(image1_mask, variable_name="image1_mask")
    if mask is not None:
        mask = as_binary_ndarray(mask, variable_name="mask")
        check_shape_equality(image0_mask, image1_mask, mask)
        image0_mask = image0_mask[mask]
        image1_mask = image1_mask[mask]
    else:
        check_shape_equality(image0_mask, image1_mask)

    nonzero_image0 = np.count_nonzero(image0_mask)
    if nonzero_image0 == 0:
        return 0
    nonzero_joint = np.count_nonzero(np.logical_and(image0_mask, image1_mask))
    return nonzero_joint / nonzero_image0
