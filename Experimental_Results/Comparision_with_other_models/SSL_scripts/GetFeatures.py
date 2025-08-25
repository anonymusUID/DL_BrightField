# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:47:04 2024

@author: Van
"""

import numpy as np
from skimage.filters import rank, sobel
from skimage.morphology import disk

def get_entropy(test_image, train_entropy_BWimg, S_nhood):
    """
    Calculate entropy feature vectors.

    Parameters:
    test_image (numpy.ndarray): Input image.
    train_entropy_BWimg (numpy.ndarray): Binary mask for entropy calculation.
    S_nhood (numpy.ndarray): Neighborhood for entropy calculation.

    Returns:
    fv_entropy (numpy.ndarray): Entropy feature vector.
    test_img_entropy (numpy.ndarray): Entropy image.
    """
    # Calculate entropy image
    #test_img_entropy = rank.entropy(test_image, disk(S_nhood.shape[0] // 2))
    test_img_entropy = rank.entropy(test_image, S_nhood)
    bin_img_entropy = (test_img_entropy == 0)
    test_img_entropy = test_img_entropy + 0.001 * bin_img_entropy

    # Calculate background entropy feature vector
    train_entropy = train_entropy_BWimg.astype(float) * test_img_entropy
    ind_entropy = np.nonzero(train_entropy)  # Indices of non-zero elements
    fv_entropy = train_entropy[ind_entropy]  # Entropy feature vector of background

    return fv_entropy, test_img_entropy


def get_gradient(test_image, train_entropy_BWimg):
    """
    Calculate gradient feature vectors.

    Parameters:
    test_image (numpy.ndarray): Input image.
    train_entropy_BWimg (numpy.ndarray): Binary mask for gradient calculation.

    Returns:
    fv_gradient (numpy.ndarray): Gradient feature vector.
    test_img_gradient (numpy.ndarray): Gradient image.
    """
    # Calculate gradient image using the Sobel operator
    test_img_gradient = sobel(test_image)
    
    # Protect against gradient = 0 elements
    bin_img_gradient = (test_img_gradient == 0)
    test_img_gradient = test_img_gradient + 0.001 * bin_img_gradient

    # Calculate background gradient feature vector
    train_gradient = train_entropy_BWimg.astype(float) * test_img_gradient
    ind_gradient = np.nonzero(train_gradient)  # Indices of non-zero elements
    fv_gradient = train_gradient[ind_gradient]  # Gradient feature vector of background

    return fv_gradient, test_img_gradient

def get_intensity(test_image, train_entropy_BWimg):
    """
    Calculate intensity feature vectors.

    Parameters:
    test_image (numpy.ndarray): Input image.
    train_entropy_BWimg (numpy.ndarray): Binary mask for intensity calculation.

    Returns:
    fv_intensity (numpy.ndarray): Intensity feature vector.
    test_img_inten (numpy.ndarray): Intensity image.
    """
    # Intensity image is the original test image
    test_img_inten = test_image.copy()

    # Protect against intensity = 0 elements
    bin_img_inten = (test_img_inten == 0)
    test_img_inten = test_img_inten + bin_img_inten.astype(np.uint8)

    # Calculate background intensity feature vector
    train_intensity = train_entropy_BWimg.astype(np.uint8) * test_img_inten
    ind_intensity = np.nonzero(train_intensity)  # Indices of non-zero elements
    fv_intensity = train_intensity[ind_intensity]  # Intensity feature vector of background

    return fv_intensity, test_img_inten


def get_imgentropy(test_image, S_nhood):
    """
    Get entropy image.

    Parameters:
    test_image (numpy.ndarray): Input image.
    S_nhood (numpy.ndarray): Neighborhood for entropy calculation.

    Returns:
    test_img_entropy (numpy.ndarray): Entropy image.
    """
    # Calculate entropy image using a disk-shaped structuring element
    #test_img_entropy = rank.entropy(test_image, disk(S_nhood.shape[0] // 2))
    test_img_entropy = rank.entropy(test_image, S_nhood)

    # Protect against entropy = 0 elements
    bin_img_entropy = (test_img_entropy == 0)
    test_img_entropy = test_img_entropy + 0.001 * bin_img_entropy

    return test_img_entropy


def get_imgintensity(test_image):
    """
    Get intensity image.

    Parameters:
    test_image (numpy.ndarray): Input image.

    Returns:
    test_img_inten (numpy.ndarray): Intensity image.
    """
    # Intensity image is the original test image
    test_img_inten = test_image.copy()
    bin_img_inten = (test_img_inten == 0)
    test_img_inten = test_img_inten + bin_img_inten.astype(np.uint8)

    return test_img_inten


def get_imggradient(test_image):
    """
    Get gradient image.

    Parameters:
    test_image (numpy.ndarray): Input image.

    Returns:
    test_img_gradient (numpy.ndarray): Gradient image.
    """
    # Calculate gradient image using the Sobel operator
    test_img_gradient = sobel(test_image)

    # Protect against gradient = 0 elements
    bin_img_gradient = (test_img_gradient == 0)
    test_img_gradient = test_img_gradient + 0.001 * bin_img_gradient

    return test_img_gradient

