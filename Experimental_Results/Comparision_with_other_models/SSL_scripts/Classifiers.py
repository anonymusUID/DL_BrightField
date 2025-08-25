# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:52:40 2024

@author: Van
"""

import numpy as np
import pandas as pd
from skimage import filters
from skimage.filters import rank
from skimage.morphology import disk
from GetFeatures import get_entropy, get_gradient, get_intensity, get_imgentropy, get_imggradient, get_imgintensity
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def GenFeatureVectorsTrain(img, BW_bg_train, BW_cell_train, S_nhood, label):
    """
    Generate feature vectors for training based on entropy, gradient, and intensity.

    Parameters:
    img (numpy.ndarray): Input image.
    BW_bg_train (numpy.ndarray): Binary mask for the background training data.
    BW_cell_train (numpy.ndarray): Binary mask for the cell training data.
    S_nhood (numpy.ndarray): Neighborhood for entropy calculation.
    label (list): List containing labels for background and cell.

    Returns:
    FV_table (pandas.DataFrame): Table containing all feature vectors and labels.
    """

    # Generate feature vectors for entropy, gradient, and intensity
    bg_fv_entropy = get_entropy(img, BW_bg_train, S_nhood)[0]
    #bg_fv_entropy2=bg_fv_entropy[0]
    cell_fv_entropy = get_entropy(img, BW_cell_train, S_nhood)[0]

    bg_fv_gradient = get_gradient(img, BW_bg_train)[0]
    cell_fv_gradient = get_gradient(img, BW_cell_train)[0]

    bg_fv_intensity = get_intensity(img, BW_bg_train)[0]
    cell_fv_intensity = get_intensity(img, BW_cell_train)[0]

    # Create background and cell labels
    bg_fv_class = [label[0]] * bg_fv_entropy.size
    cell_fv_class = [label[1]] * cell_fv_entropy.size

    # Create tables containing all feature vectors and labels
    train_bg_fv_table = pd.DataFrame({
        'entropy': bg_fv_entropy,
        'gradient': bg_fv_gradient,
        'intensity': bg_fv_intensity,
        'Label': bg_fv_class
    })

    train_cell_fv_table = pd.DataFrame({
        'entropy': cell_fv_entropy,
        'gradient': cell_fv_gradient,
        'intensity': cell_fv_intensity,
        'Label': cell_fv_class
    })

    # Concatenate both tables
    FV_table = pd.concat([train_bg_fv_table, train_cell_fv_table], ignore_index=True)

    return FV_table


def GenFeatureVectorsTest(img, S_nhood):
    """
    Generate feature vectors for testing based on entropy, gradient, and intensity.

    Parameters:
    img (numpy.ndarray): Input image.
    S_nhood (numpy.ndarray): Neighborhood for entropy calculation.

    Returns:
    FV_table (pandas.DataFrame): Table containing feature vectors.
    """
    # Get the dimensions of the image
    rows, cols = img.shape

    # Generate feature vectors for entropy, gradient, and intensity
    img_entropy = get_imgentropy(img, S_nhood)
    test_fv_entropy = img_entropy.flatten()  # Feature vector of all pixels

    img_gradient = get_imggradient(img)
    test_fv_gradient = img_gradient.flatten()

    img_intensity = get_imgintensity(img)
    test_fv_intensity = img_intensity.flatten()

    # Create a table containing all feature vectors
    FV_table = pd.DataFrame({
        'entropy': test_fv_entropy,
        'gradient': test_fv_gradient,
        'intensity': test_fv_intensity
    })

    return FV_table

def Bayes_S_G_I_trainClassifier(training_data):
    """
    Train a Naive Bayes classifier and compute its validation accuracy.

    Parameters:
    training_data (pandas.DataFrame): A table containing the predictor and response columns.

    Returns:
    trained_classifier (dict): A dictionary containing the trained classifier and additional information.
    validation_accuracy (float): The accuracy of the classifier in percent.
    """
    # Extract predictors and response
    predictor_names = ['entropy', 'gradient', 'intensity']
    predictors = training_data[predictor_names]
    response = training_data['Label']
    
    # Encode the response variable
    le = LabelEncoder()
    response_encoded = le.fit_transform(response)

    # Check for categorical predictors
    is_categorical_predictor = [False, False, False]

    # Train a Naive Bayes classifier
    if any(is_categorical_predictor):
        # Use Categorical Naive Bayes if any predictor is categorical
        classifier = CategoricalNB()
    else:
        # Use Gaussian Naive Bayes for all numeric predictors
        classifier = GaussianNB()

    classifier.fit(predictors, response_encoded)

    # Create the result dictionary with predict function
    def predict_fn(new_data):
        return le.inverse_transform(classifier.predict(new_data[predictor_names]))

    trained_classifier = {
        'predictFcn': predict_fn,
        'RequiredVariables': predictor_names,
        'ClassificationNaiveBayes': classifier,
        'About': 'This dictionary contains a trained model exported from a Python script.',
        'HowToPredict': ('To make predictions on a new dataframe, T, use: \n'
                         '  yfit = trained_classifier["predictFcn"](T) \n'
                         'The dataframe, T, must contain the variables listed in trained_classifier["RequiredVariables"].')
    }

    # Perform cross-validation
    validation_accuracy = cross_val_score(classifier, predictors, response_encoded, cv=5, scoring='accuracy').mean() * 100

    return trained_classifier, validation_accuracy

