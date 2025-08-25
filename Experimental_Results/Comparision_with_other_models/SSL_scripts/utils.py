# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:35:59 2024

@author: Van
"""

import os
import numpy as np
import cv2
from skimage import io, color, img_as_ubyte, filters, morphology, measure
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt


## Function to create a folder if not exists
def create_folder(folder_name):
    """
    Check if the folder already exists. If not, create it.

    Parameters:
    folder_name (str): Name or path of the folder to create.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f'Folder "{folder_name}" was created successfully.')
    else:
        print(f'Folder "{folder_name}" already exists.')
        




## Fucntion to read input images
def ReadDataCube(dir_str):
    """
    Read in a data cube for video microscopy data.

    Parameters:
    dir_str (str): The directory containing the microscopy data files.

    Returns:
    out (numpy.ndarray): The data cube containing the processed images.
    info (list): List of file information.
    """
    files = [f for f in os.listdir(dir_str) if os.path.isfile(os.path.join(dir_str, f))]
    num_of_files = len(files)
    S_th = 1.5e5
    S_nhood = np.ones((7, 7), dtype=bool)  # Neighborhood for entropy filter
    out = []  # List to store output images

    # Read in data from files.
    for i in range(num_of_files):
        # Form string for filename.
        file_path = os.path.join(dir_str, files[i])
        
        # Read file to image, map to grayscale and rescale image.
        #A = io.imread(file_path)
        #A = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
        A = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
        
        # Check original image type, if not uint16 and intensity is less than 1000, multiply by a factor of 16
        if A.dtype != np.uint8:
            if np.mean(A) < 1000:
                A = A * 16
        
        # If RGB then make Grayscale.
        if len(A.shape) == 3 and A.shape[2] == 3:
            A = color.rgb2gray(A)
            A = img_as_ubyte(A)  # Convert to uint8 after grayscale
            
            
        # If not uint8 then recast.
        if A.dtype != np.uint8:
            A = img_as_ubyte(A)
            
            
        # Calculate entropy image
        entro_img = filters.rank.entropy(A, np.ones((7, 7)))
        S_total = np.sum(entro_img)
        
        # If sum entropy of all pixels is less than a threshold number, compose a 2nd order image
        if S_total < S_th:
            A = A ** 2
        
        # Store the processed image in the list.
        out.append(A)
    
    # Convert list to numpy array
    out = np.stack(out, axis=-1)
    info = files
    return out, info


## Function to calculate optical flow at every single pixel
def OpticalFlowSeg(I_pair, th, MinPixelsInConnectedRegion, SizeOfSmoothingDisk):
    """
    Optical flow segmentation using two images.

    Parameters:
    I_pair (numpy.ndarray): Pair of images for optical flow calculation.
    th (float): Threshold for flow vector magnitude.
    MinPixelsInConnectedRegion (int): Minimum size of connected regions.
    SizeOfSmoothingDisk (int): Size of the smoothing disk.

    Returns:
    fillBW (numpy.ndarray): Binary mask after segmentation.
    mag (numpy.ndarray): Magnitude of the optical flow.
    """
    #th = 0.04
    #MinPixelsInConnectedRegion = 600
    #SizeOfSmoothingDisk = 5
    
    # # Initialize Farneback optical flow parameters
    # flow = cv2.calcOpticalFlowFarneback(
    #     I_pair[:, :, 0], 
    #     I_pair[:, :, 1], 
    #     None, 
    #     0.5,  # Pyr scale
    #     3,    # Levels
    #     25,   # Winsize
    #     3,    # Iterations
    #     5,    # PolyN
    #     1.2, # PolySigma
    #     0     # Flags
    # )
    
    height, width = I_pair[:,:,0].shape[:2]
    prev_u8 = np.zeros((height, width), dtype=np.uint8)
    next_u8 = I_pair[:,:,1]
    
    # fixed_u8  = img_as_ubyte((fixed - fixed.min())/(fixed.max()-fixed.min()))
    # moving_u8 = img_as_ubyte((moving - moving.min())/(moving.max()-moving.min()))
    
    mean_fixed = np.mean(I_pair[:, :, 0])
    if -1.0 < mean_fixed < 1.0:
        next_u8_convert = img_as_ubyte((next_u8 - next_u8.min())/(next_u8.max()-next_u8.min()))
    else:
        next_u8_convert = next_u8
        
    #next_u8_convert = img_as_ubyte((next_u8 - next_u8.min())/(next_u8.max()-next_u8.min()))
    

    
    
    flow = cv2.calcOpticalFlowFarneback(
    prev_u8, next_u8_convert, None,
    pyr_scale=0.5,
    levels=3,
    winsize=25,
    iterations=3,
    poly_n=7,
    poly_sigma=1.5,
    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    

    # Threshold the magnitude to create initial binary mask
    BW = mag > th
    
    # Remove regions with low connectivity (size filtering)
    BW2 = morphology.remove_small_objects(BW, MinPixelsInConnectedRegion,connectivity =2)

    # Close image: Smooth mask to remove "over-articulation"
    se = morphology.disk(SizeOfSmoothingDisk)
    closeBW = morphology.binary_closing(BW2, se)

    # Fill image holes
    fillBW = binary_fill_holes(closeBW)

    # Segment different regions
    L, NF = measure.label(fillBW, return_num=True)

    return fillBW, mag
    

## Function to find the minimum connected area in the input image using OF
def findMinConnectedRegion(imgdata):
    """
    Apply Optical Flow (OF) to determine a reasonable MinPixelsConnectedRegion and segment the image.

    Parameters:
    imgdata (numpy.ndarray): The input image data.

    Returns:
    MinArea (int): Minimum area of connected regions.
    MeanArea (int): Mean area of connected regions.
    B_OF (list): List of boundary coordinates for segmented regions.
    img_pair (numpy.ndarray): Pair of fixed and simulated images.
    mag (numpy.ndarray): Magnitude of optical flow.
    """

    # Threshold and morphological processing parameters
    th = 0.04
    MinPixelsInConnectedRegion = 600
    SizeOfSmoothingDisk = 5
    sigma = 0.1
     # Use Gaussian filter to simulate moving frame
    fixed = imgdata
   
    
    sim = filters.gaussian(fixed,
                   sigma=sigma,           # same σ
                   truncate=4.0,        # controls filter size (~4σ each side)
                   preserve_range=True, # don’t normalize the output
                   mode='reflect')  
    moving = sim.astype(np.uint8)
    # Create image pair
    img_pair = np.stack((fixed, moving), axis=-1)
    # Compute optical flow segmentation (not self-supervised)
    fillBW, mag = OpticalFlowSeg(img_pair, th, MinPixelsInConnectedRegion, SizeOfSmoothingDisk)

    # Calculate the statistics of segmented regions

    L, NF = measure.label(fillBW,connectivity = 2, return_num=True)
    stats_OF = measure.regionprops_table(L, properties=['area'])
    Area_OF = stats_OF['area']

    if np.isnan(np.mean(Area_OF)):
        MinArea = 550
    else:
        MinArea = round(np.mean(Area_OF) / 4)
        MeanArea = round(np.mean(Area_OF))

    # Find boundaries of segmented regions
    B_OF = measure.find_contours(fillBW, level=0.5)

    return MinArea, B_OF, img_pair, mag

def normalize_to_uint8(image):
    """
    Normalize a float64 image to the range [0, 255] and convert to uint8.
    
    Parameters:
    image (numpy.ndarray): Input image of type float64.

    Returns:
    numpy.ndarray: Normalized image of type uint8.
    """
    # Normalize the image to the range [0, 255]
    image_normalize = np.zeros(image.shape)
    image_uint8 = np.zeros(image.shape)
    
    image_normalize = 255 * (image- np.min(image)) / (np.max(image) - np.min(image))
    # Convert to uint8
    image_uint8= image_normalize.astype(np.uint8)
    
    return image_uint8


def OF_CellvsBackgroundTraining(I_pair, MinPixelsInConnectedRegion, S_nhood, extra_S):
    """
    Use a pair of images for optical flow inputs to isolate background and cells using entropy measurements for self-tuning the threshold.

    Parameters:
    I_pair (numpy.ndarray): Pair of images for optical flow calculation.
    MinPixelsInConnectedRegion (int): Minimum size of connected regions.
    S_nhood (numpy.ndarray): Neighborhood for entropy filter.
    extra_S (float): Extra entropy value determined by camera.

    Returns:
    bg_train_entr (numpy.ndarray): Background training entropy.
    cell_train_entr (numpy.ndarray): Cell training entropy.
    """

    # Initialize parameters
    SizeOfSmoothingDisk = 5
    dilate = False
    color_img = False
    
    # norm_fixed = normalize_to_uint8(I_pair[:,:,0])
    # train_img_entropy = filters.rank.entropy(norm_fixed, S_nhood)
    
    norm_img_pair = normalize_to_uint8(I_pair)
    train_img_entropy = filters.rank.entropy(I_pair[:,:,0], S_nhood)

    # Initial threshold for background
    bg_th_init = 1e-4
    bg_train_S_total = 1
    i=1

    # Find background using optical flow segmentation
    while bg_train_S_total <= 1e5:
        BW_Low_th, mag = OpticalFlowSeg(I_pair, bg_th_init, MinPixelsInConnectedRegion, SizeOfSmoothingDisk)
        #fillBW, mag = OpticalFlowSeg(img_pair, th, MinPixelsInConnectedRegion, SizeOfSmoothingDisk)
        bg_mag = (~BW_Low_th) * mag
        ####
        bg_mask = bg_mag.astype(bool)
        bg_mask_double = bg_mask.astype(float)
        bg_train_entr = bg_mask_double * train_img_entropy
    ####
        #bg_train_entr = (bg_mag.astype(float)) * train_img_entropy
        bg_train_S_total = np.sum(bg_train_entr)
        if bg_th_init > 0.002:
            break
        bg_th_init += 1e-4

    OF_Labelled_Background = ~BW_Low_th
    bg_th = bg_th_init - 1e-4

    # Generate Cell Training Data
    cell_train_S_total = 1
    i = 1
    S_total_th = 1e5
    cell_th_init = 0.04

    # Find cells using optical flow segmentation
    while cell_train_S_total <= S_total_th:
        if cell_th_init <= 0:
            cell_th_init = 0.00002
            BW_high_th, mag = OpticalFlowSeg(I_pair, cell_th_init, MinPixelsInConnectedRegion, SizeOfSmoothingDisk)
            cell_mag = BW_high_th * mag
            ###
            cell_mask = (cell_mag != 0).astype(np.float64)   # or np.float32 if you prefer
            cell_train_entr = cell_mask * train_img_entropy
            ###

            break

        BW_high_th, mag = OpticalFlowSeg(I_pair, cell_th_init, MinPixelsInConnectedRegion, SizeOfSmoothingDisk)
        cell_mag = BW_high_th * mag

        if i == 1:
            ##
            cell_mask = (cell_mag != 0).astype(np.float64)   # or np.float32 if you prefer
            cell_train_entr = cell_mask * train_img_entropy
            
            cell_train_S_total = np.sum(cell_train_entr)
            S_total_th = cell_train_S_total + extra_S
        else:
            ##
            cell_mask = (cell_mag != 0).astype(np.float64)   # or np.float32 if you prefer
            cell_train_entr = cell_mask * train_img_entropy
            ##
            #cell_train_entr = (cell_mag.astype(float)) * train_img_entropy
            cell_train_S_total = np.sum(cell_train_entr)

        cell_th_init -= 2e-3
        i += 1

    OF_Labelled_Cells = BW_high_th
    cell_th = cell_th_init + 2e-3

    return bg_train_entr, cell_train_entr



def im2uint8_matlab_style(A):
    A = np.asarray(A, dtype=np.float64)
    A = np.clip(A, 0.0, 1.0)
    return img_as_ubyte(A)
