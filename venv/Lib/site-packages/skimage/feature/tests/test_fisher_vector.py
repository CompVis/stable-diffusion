import pytest

import numpy as np

pytest.importorskip('sklearn')

from skimage.feature.fisher_vector import (   # noqa: E402
    learn_gmm, fisher_vector, FisherVectorException,
    DescriptorException
)


def test_gmm_wrong_descriptor_format_1():
    """Test that DescriptorException is raised when wrong type for descriptions
    is passed.
    """

    with pytest.raises(DescriptorException):
        learn_gmm('completely wrong test', n_modes=1)


def test_gmm_wrong_descriptor_format_2():
    """Test that DescriptorException is raised when descriptors are of
    different dimensionality.
    """

    with pytest.raises(DescriptorException):
        learn_gmm([np.zeros((5, 11)), np.zeros((4, 10))], n_modes=1)


def test_gmm_wrong_descriptor_format_3():
    """Test that DescriptorException is raised when not all descriptors are of
    rank 2.
    """

    with pytest.raises(DescriptorException):
        learn_gmm([np.zeros((5, 10)), np.zeros((4, 10, 1))], n_modes=1)


def test_gmm_wrong_descriptor_format_4():
    """Test that DescriptorException is raised when elements of descriptor list
    are of the incorrect type (i.e. not a NumPy ndarray).
    """

    with pytest.raises(DescriptorException):
        learn_gmm([[1, 2, 3], [1, 2, 3]], n_modes=1)


def test_gmm_wrong_num_modes_format_1():
    """Test that FisherVectorException is raised when incorrect type for
    n_modes is passed into the learn_gmm function.
    """

    with pytest.raises(FisherVectorException):
        learn_gmm([np.zeros((5, 10)), np.zeros((4, 10))], n_modes='not_valid')


def test_gmm_wrong_num_modes_format_2():
    """Test that FisherVectorException is raised when a number that is not a
    positive integer is passed into the n_modes argument of learn_gmm.
    """

    with pytest.raises(FisherVectorException):
        learn_gmm([np.zeros((5, 10)), np.zeros((4, 10))], n_modes=-1)


def test_gmm_wrong_covariance_type():
    """Test that FisherVectorException is raised when wrong covariance type is
    passed in as a keyword argument.
    """

    with pytest.raises(FisherVectorException):
        learn_gmm(
            np.random.random((10, 10)), n_modes=2,
            gm_args={'covariance_type': 'full'}
        )


def test_gmm_correct_covariance_type():
    """Test that GMM estimation is successful when the correct covariance type
    is passed in as a keyword argument.
    """

    gmm = learn_gmm(
        np.random.random((10, 10)), n_modes=2,
        gm_args={'covariance_type': 'diag'}
    )

    assert gmm.means_ is not None
    assert gmm.covariances_ is not None
    assert gmm.weights_ is not None


def test_gmm_e2e():
    """
    Test the GMM estimation. Since this is essentially a wrapper for the
    scikit-learn GaussianMixture class, the testing of the actual inner
    workings of the GMM estimation is left to scikit-learn and its
    dependencies.

    We instead simply assert that the estimation was successful based on the
    fact that the GMM object will have associated mixture weights, means, and
    variances after estimation is successful/complete.
    """

    gmm = learn_gmm(np.random.random((100, 64)), n_modes=5)

    assert gmm.means_ is not None
    assert gmm.covariances_ is not None
    assert gmm.weights_ is not None


def test_fv_wrong_descriptor_types():
    """
    Test that DescriptorException is raised when the incorrect type for the
    descriptors is passed into the fisher_vector function.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print(
            'scikit-learn is not installed. Please ensure it is installed in '
            'order to use the Fisher vector functionality.'
        )

    with pytest.raises(DescriptorException):
        fisher_vector([[1, 2, 3, 4]], GaussianMixture())


def test_fv_wrong_gmm_type():
    """
    Test that FisherVectorException is raised when a GMM not of type
    sklearn.mixture.GaussianMixture is passed into the fisher_vector
    function.
    """

    class MyDifferentGaussianMixture:
        pass

    with pytest.raises(FisherVectorException):
        fisher_vector(np.zeros((10, 10)), MyDifferentGaussianMixture())


def test_fv_e2e():
    """
    Test the Fisher vector computation given a GMM returned from the learn_gmm
    function. We simply assert that the dimensionality of the resulting Fisher
    vector is correct.

    The dimensionality of a Fisher vector is given by 2KD + K, where K is the
    number of Gaussians specified in the associated GMM, and D is the
    dimensionality of the descriptors using to estimate the GMM.
    """

    dim = 128
    num_modes = 8

    expected_dim = 2 * num_modes * dim + num_modes

    descriptors = [
        np.random.random((np.random.randint(5, 30), dim))
        for _ in range(10)
    ]

    gmm = learn_gmm(descriptors, n_modes=num_modes)

    fisher_vec = fisher_vector(descriptors[0], gmm)

    assert len(fisher_vec) == expected_dim


def test_fv_e2e_improved():
    """
    Test the improved Fisher vector computation given a GMM returned from the
    learn_gmm function. We simply assert that the dimensionality of the
    resulting Fisher vector is correct.

    The dimensionality of a Fisher vector is given by 2KD + K, where K is the
    number of Gaussians specified in the associated GMM, and D is the
    dimensionality of the descriptors using to estimate the GMM.
    """

    dim = 128
    num_modes = 8

    expected_dim = 2 * num_modes * dim + num_modes

    descriptors = [
        np.random.random((np.random.randint(5, 30), dim))
        for _ in range(10)
    ]

    gmm = learn_gmm(descriptors, n_modes=num_modes)

    fisher_vec = fisher_vector(descriptors[0], gmm, improved=True)

    assert len(fisher_vec) == expected_dim
