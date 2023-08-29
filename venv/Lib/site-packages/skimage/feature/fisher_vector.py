"""
fisher_vector.py - Implementation of the Fisher vector encoding algorithm

This module contains the source code for Fisher vector computation. The
computation is separated into two distinct steps, which are called separately
by the user, namely:

learn_gmm: Used to estimate the GMM for all vectors/descriptors computed for
           all examples in the dataset (e.g. estimated using all the SIFT
           vectors computed for all images in the dataset, or at least a subset
           of this).

fisher_vector: Used to compute the Fisher vector representation for a
               single set of descriptors/vector (e.g. the SIFT
               descriptors for a single image in your dataset, or
               perhaps a test image).

Reference: Perronnin, F. and Dance, C. Fisher kernels on Visual Vocabularies
           for Image Categorization, IEEE Conference on Computer Vision and
           Pattern Recognition, 2007

Origin Author: Dan Oneata (Author of the original implementation for the Fisher
vector computation using scikit-learn and NumPy. Subsequently ported to
scikit-image (here) by other authors.)
"""
import numpy as np


class FisherVectorException(Exception):
    pass


class DescriptorException(FisherVectorException):
    pass


def learn_gmm(descriptors, *, n_modes=32, gm_args=None):
    """Estimate a Gaussian mixture model (GMM) given a set of descriptors and
    number of modes (i.e. Gaussians). This function is essentially a wrapper
    around the scikit-learn implementation of GMM, namely the
    :func:`sklearn.mixture.GaussianMixture` class.

    Due to the nature of the Fisher vector, the only enforced parameter of the
    underlying scikit-learn class is the covariance_type, which must be 'diag'.

    There is no simple way to know what value to use for `n_modes` a-priori.
    Typically, the value is usually one of ``{16, 32, 64, 128}``. One may train
    a few GMMs and choose the one that maximises the log probability of the
    GMM, or choose `n_modes` such that the downstream classifier trained on
    the resultant Fisher vectors has maximal performance.

    Parameters
    ----------
    descriptors : np.ndarray (N, M) or list [(N1, M), (N2, M), ...]
        List of NumPy arrays, or a single NumPy array, of the descriptors
        used to estimate the GMM. The reason a list of NumPy arrays is
        permissible is because often when using a Fisher vector encoding,
        descriptors/vectors are computed separately for each sample/image in
        the dataset, such as SIFT vectors for each image. If a list if passed
        in, then each element must be a NumPy array in which the number of
        rows may differ (e.g. different number of SIFT vector for each image),
        but the number of columns for each must be the same (i.e. the
        dimensionality must be the same).
    n_modes : int
        The number of modes/Gaussians to estimate during the GMM estimate.
    gm_args : dict
        Keyword arguments that can be passed into the underlying scikit-learn
        :func:`sklearn.mixture.GaussianMixture` class.

    Returns
    -------
    gmm : :func:`sklearn.mixture.GaussianMixture`
        The estimated GMM object, which contains the necessary parameters
        needed to compute the Fisher vector.

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

    Examples
    --------
    >>> import pytest
    >>> _ = pytest.importorskip('sklearn')
    >>> from skimage.feature import fisher_vector
    >>> rng = np.random.Generator(np.random.PCG64())
    >>> sift_for_images = [rng.standard_normal((10, 128)) for _ in range(10)]
    >>> num_modes = 16
    >>> # Estimate 16-mode GMM with these synthetic SIFT vectors
    >>> gmm = learn_gmm(sift_for_images, n_modes=num_modes)
    """

    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError(
            'scikit-learn is not installed. Please ensure it is installed in '
            'order to use the Fisher vector functionality.'
        )

    if not isinstance(descriptors, (list, np.ndarray)):
        raise DescriptorException(
            'Please ensure descriptors are either a NumPY array, '
            'or a list of NumPy arrays.'
        )

    d_mat_1 = descriptors[0]
    if isinstance(descriptors, list) and not isinstance(d_mat_1, np.ndarray):
        raise DescriptorException(
            'Please ensure descriptors are a list of NumPy arrays.'
        )

    if isinstance(descriptors, list):
        expected_shape = descriptors[0].shape
        ranks = [len(e.shape) == len(expected_shape) for e in descriptors]
        if not all(ranks):
            raise DescriptorException(
                'Please ensure all elements of your descriptor list '
                'are of rank 2.'
            )
        dims = [e.shape[1] == descriptors[0].shape[1] for e in descriptors]
        if not all(dims):
            raise DescriptorException(
                'Please ensure all descriptors are of the same dimensionality.'
            )

    if not isinstance(n_modes, int) or n_modes <= 0:
        raise FisherVectorException(
            'Please ensure n_modes is a positive integer.'
        )

    if gm_args:
        has_cov_type = 'covariance_type' in gm_args
        cov_type_not_diag = gm_args['covariance_type'] != 'diag'
        if has_cov_type and cov_type_not_diag:
            raise FisherVectorException('Covariance type must be "diag".')

    if isinstance(descriptors, list):
        descriptors = np.vstack(descriptors)

    if gm_args:
        has_cov_type = 'covariance_type' in gm_args
        if has_cov_type:
            gmm = GaussianMixture(
                n_components=n_modes, **gm_args
            )
        else:
            gmm = GaussianMixture(
                n_components=n_modes, covariance_type='diag', **gm_args
            )
    else:
        gmm = GaussianMixture(n_components=n_modes, covariance_type='diag')

    gmm.fit(descriptors)

    return gmm


def fisher_vector(descriptors, gmm, *, improved=False, alpha=0.5):
    """Compute the Fisher vector given some descriptors/vectors,
    and an associated estimated GMM.

    Parameters
    ----------
    descriptors : np.ndarray, shape=(n_descriptors, descriptor_length)
        NumPy array of the descriptors for which the Fisher vector
        representation is to be computed.
    gmm : sklearn.mixture.GaussianMixture
        An estimated GMM object, which contains the necessary parameters needed
        to compute the Fisher vector.
    improved : bool, default=False
        Flag denoting whether to compute improved Fisher vectors or not.
        Improved Fisher vectors are L2 and power normalized. Power
        normalization is simply f(z) = sign(z) pow(abs(z), alpha) for some
        0 <= alpha <= 1.
    alpha : float, default=0.5
        The parameter for the power normalization step. Ignored if
        improved=False.

    Returns
    -------
    fisher_vector : np.ndarray
        The computation Fisher vector, which is given by a concatenation of the
        gradients of a GMM with respect to its parameters (mixture weights,
        means, and covariance matrices). For D-dimensional input descriptors or
        vectors, and a K-mode GMM, the Fisher vector dimensionality will be
        2KD + K. Thus, its dimensionality is invariant to the number of
        descriptors/vectors.

    References
    ----------
    .. [1] Perronnin, F. and Dance, C. Fisher kernels on Visual Vocabularies
           for Image Categorization, IEEE Conference on Computer Vision and
           Pattern Recognition, 2007
    .. [2] Perronnin, F. and Sanchez, J. and Mensink T. Improving the Fisher
           Kernel for Large-Scale Image Classification, ECCV, 2010

    Examples
    --------
    >>> import pytest
    >>> _ = pytest.importorskip('sklearn')
    >>> from skimage.feature import fisher_vector, learn_gmm
    >>> sift_for_images = [np.random.random((10, 128)) for _ in range(10)]
    >>> num_modes = 16
    >>> # Estimate 16-mode GMM with these synthetic SIFT vectors
    >>> gmm = learn_gmm(sift_for_images, n_modes=num_modes)
    >>> test_image_descriptors = np.random.random((25, 128))
    >>> # Compute the Fisher vector
    >>> fv = fisher_vector(test_image_descriptors, gmm)
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError(
            'scikit-learn is not installed. Please ensure it is installed in '
            'order to use the Fisher vector functionality.'
        )

    if not isinstance(descriptors, np.ndarray):
        raise DescriptorException(
            'Please ensure descriptors is a NumPy array.'
        )

    if not isinstance(gmm, GaussianMixture):
        raise FisherVectorException(
            'Please ensure gmm is a sklearn.mixture.GaussianMixture object.'
        )

    if improved and not isinstance(alpha, float):
        raise FisherVectorException(
            'Please ensure that the alpha parameter is a float.'
        )

    num_descriptors = len(descriptors)

    mixture_weights = gmm.weights_
    means = gmm.means_
    covariances = gmm.covariances_

    posterior_probabilities = gmm.predict_proba(descriptors)

    # Statistics necessary to compute GMM gradients wrt its parameters
    pp_sum = posterior_probabilities.mean(axis=0, keepdims=True).T
    pp_x = posterior_probabilities.T.dot(descriptors) / num_descriptors
    pp_x_2 = posterior_probabilities.T.dot(
        np.power(descriptors, 2)
    ) / num_descriptors

    # Compute GMM gradients wrt its parameters
    d_pi = pp_sum.squeeze() - mixture_weights

    d_mu = pp_x - pp_sum * means

    d_sigma_t1 = pp_sum * np.power(means, 2)
    d_sigma_t2 = pp_sum * covariances
    d_sigma_t3 = 2 * pp_x * means
    d_sigma = -pp_x_2 - d_sigma_t1 + d_sigma_t2 + d_sigma_t3

    # Apply analytical diagonal normalization
    sqrt_mixture_weights = np.sqrt(mixture_weights)
    d_pi /= sqrt_mixture_weights
    d_mu /= sqrt_mixture_weights[:, np.newaxis] * np.sqrt(covariances)
    d_sigma /= np.sqrt(2) * sqrt_mixture_weights[:, np.newaxis] * covariances

    # Concatenate GMM gradients to form Fisher vector representation
    fisher_vector = np.hstack((d_pi, d_mu.ravel(), d_sigma.ravel()))

    if improved:
        fisher_vector = \
            np.sign(fisher_vector) * np.power(np.abs(fisher_vector), alpha)
        fisher_vector = fisher_vector / np.linalg.norm(fisher_vector)

    return fisher_vector
