#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:53:20 2023

@author: vech
"""
import numpy as np
import matplotlib.pyplot as plt

# Import packages for image processing
import skimage
import imageio.v2 as imageio
from skimage import morphology #for measuring things in the masks
from skimage.transform import rotate
from skimage.color import rgb2gray, rgba2rgb  # Import rgba2rgb function
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square, opening, disk
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage import segmentation
from skimage.segmentation import slic



#-------------------
# Help functions
#------------------



#Main function to extract features from an image, that calls other functions    
def extract_features(image, mask):
    



    # Measuring asymmetry
    asymmetry_score = computeAsymmetry(mask)
    

    # Measuring colour
    colour_score = how_many_colours_are_there(image, mask, hue_range = 60)
   

    # Measuring dots
    dots_score = computeDotsScore(image,mask)
    


    # Output: 1D array of all the scores
    return np.array([asymmetry_score,colour_score,dots_score], dtype=np.float16)

    #### TK - create output as list, if we want to have image_id as a column in features.csv.
    #list_with_scores = [asymmetry_score,colour_score,dots_score]
    #return list_with_scores




#########################################################################
############################### ASYMMETRY ###############################
#########################################################################

def rotateImage(mask, angle, center):

    """
    Rotates the given image by given angle in clockwise direction. Center is set to center of image by default. 
    See skimage documentation: https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rotate
    :image: 2D numpy array
    :angle: degrees
    :return: new image - 2D numpy array
    """

    rotated_image = rotate(mask, -angle, center = center)
    return rotated_image


def findCentroid(mask):

    """
    Finds centroid using the mean of x and y coordinates.
    If confused regards to axis in numpy, see this answer (scroll to the bottom for visual):
    https://stackoverflow.com/questions/17079279/how-is-axis-indexed-in-numpys-array
    :mask: 2D array containing binary values where 1s signify the selected region
    :return: xCoord, yCoord
    """

    # X-coordinates
    region_column_indices = np.where(np.sum(mask, axis = 0) > 0)[0]
    left, right = region_column_indices[0], region_column_indices[-1]
    xCoord = (left + right)//2

    # Y-coordinates
    region_row_indices = np.where(np.sum(mask, axis = 1) > 0)[0]
    top, bottom = region_row_indices[0], region_row_indices[-1]
    yCoord = (top + bottom)//2

    return xCoord, yCoord


def halveTheRegionHorizontally(yCoord, mask):

    """
    Splits the image into two halves horizontally. The horizontal "line" is set to go through the y-th Coordinate.
    :yCoord: index
    :mask: 2D binary numpy array
    :return: 2x 2D numpy array with exact same dimensions representing the two halves.
    """

    # Get the halves
    upper = mask[:yCoord]
    lower = mask[yCoord:]

    # Make sure both halves have the same amount of rows
    n_rows_upper = upper.shape[0]
    n_rows_lower = lower.shape[0]

    # Lower half needs more rows
    if  n_rows_upper > n_rows_lower:

        # Get inputs for transformation
        row_difference = n_rows_upper - n_rows_lower
        n_columns = lower.shape[1]
        additional_rows = [[0]*n_columns for _ in range(row_difference)]

        # Stacks row-wise lower and then additional rows --> notice the order is important since we want to add new rows to the bottom
        lower = np.vstack((lower, additional_rows))

    # Upper half needs more rows
    elif n_rows_upper < n_rows_lower:

        # Get inputs for transformation
        row_difference = n_rows_lower - n_rows_upper
        n_columns = upper.shape[1]
        additional_rows = [[0]*n_columns for _ in range(row_difference)]

        # Same logic as above, notice here that we are choosing first additional rows and then upper
        upper = np.vstack((additional_rows, upper))
    
    # Flip the lower along the x-axis, so it can be then compared directly without any further transformation
    lower = np.flip(lower, axis = 0)

    return lower, upper



def computeAsymmetry(mask):

    """
    Computes the asymmetry of the region by following procedure:
    1. Finds the midpoint of lesion using the mean of x and y coordinates
    Then rotates the images by specified angles and for each rotation:
    2. Splits the region in half using the above coordinates (horizontally)
    3. Subtracts the two halves from each other, sums the differences and computes 
    this difference relative to the size of the lesion.
    Finally, out of all computations, we take the minimum value and return it as the asymmetry.
    :mask: 2D binary numpy array
    :return: horizontal_asymmetry (normalized by division by lesion area)
    """

    # Total area of lesion
    lesion_area = np.sum(mask, dtype=np.float64)
    
    # Get center
    xCoord, yCoord = findCentroid(mask)
    center = [xCoord, yCoord]

    # Specify the angles for rotation
    angles = [i for i in range(30, 181, 30)]

    # Get the asymmetry results for each rotation
    asymmetry_results = []
    for angle in angles:

        # Rotation
        rotated_mask = rotateImage(mask, angle, center)

        # Horizontal split
        bottom, top = halveTheRegionHorizontally(yCoord, rotated_mask)
        horizontal_asymmetry = abs(np.sum((bottom - top)))/lesion_area

        # Save the result
        asymmetry_results.append(horizontal_asymmetry)
    
    return min(asymmetry_results)



#########################################################################
############################# COLOUR ####################################
#########################################################################

def find_topbottom(mask):
    '''
    Function to get top / bottom boundaries of lesion using a binary mask.
    :mask: Binary image mask as numpy.array
    :return: top, bottom as int
    '''
    region_row_indices = np.where(np.sum(mask, axis = 1) > 0)[0]
    top, bottom = region_row_indices[0], region_row_indices[-1]
    return top, bottom

def find_leftright(mask):
    '''
    Function to get left / right boundaries of lesion using a binary mask.
    :mask: Binary image mask as numpy.array
    :return: left, right as int
    '''

    region_column_indices = np.where(np.sum(mask, axis = 0) > 0)[0]
    left, right = region_column_indices[0], region_column_indices[-1]
    return left, right

def lesionMaskCrop(image, mask):
    '''
    This function masks and crops an area of a color image corresponding to a binary mask of same dimension.

    :image: RGB image read as numpy.array
    :mask: Corresponding binary mask as numpy.array
    '''
    # Getting top/bottom and left/right boundries of lesion
    top, bottom = find_topbottom(mask)
    left, right = find_leftright(mask)

    # Masking out lesion in color image
    im_masked = image.copy()
    im_masked[mask==0] = 0 # color 0 = black

    # Cropping image using lesion boundaries
    im_crop = im_masked[top:bottom+1,left:right+1]

    return(im_crop)

def rgb_to_hsv(r, g, b):

    """
    Converts RGB colour value to HSV value

    Credit for the entire function goes to: 
    https://www.w3resource.com/python-exercises/math/python-math-exercise-77.php
    """
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

def getColorFeatures(image, mask):

    """
    This function returns the hues within an image. 
    This method uses SLIC segmentation to select superpixels for gathering average regional colour intensities.
    Once colour intense regions are segmented, the mean colour in each region is calculated.
    These averages are converted to hue for each regions.

    :image: RGB image read as numpy.array
    :mask: Corresponding binary mask as numpy.array
    :return: list with hues
    """

    # Mask and crop image to only contain lesion
    im_lesion = lesionMaskCrop(image, mask)

    # Get SLIC boundaries
    segments = slic(im_lesion, n_segments=60, compactness=5, sigma=1, start_label=1)

    # Fetch RegionProps - this includes min/mean/max values for color intensity
    regions = regionprops(segments, intensity_image=im_lesion)

    # Access mean color intensity for each region
    mean_intensity = [r.mean_intensity for r in regions]

    # Get only segments with color in them
    color_intensity = []
    for mean in mean_intensity:
        if sum(mean) != 0:
            color_intensity.append(mean)

    # Convert RGB color means to HSV
    color_mean_hsv = [rgb_to_hsv(col_int[0], col_int[1], col_int[2]) for col_int in color_intensity]

    # Extract values for hue
    color_mean_hue = [hsv[0] for hsv in color_mean_hsv]

    return color_mean_hue


    #This function is the "ultimate" one that should be used
def how_many_colours_are_there(image, mask, hue_range = 60):

    '''
    Counts the number of diifferent colours are in an image.
    Specifically assesses how much the hues of an image vary.
    
    :image: RGB image read as numpy.array
    :mask: Corresponding binary mask as numpy.array
    :hue_range: adjustable, but set at 60 since this is generally given as the value difference between the 6 main colours of the hue spectrum
    :return: int value from 1-5 to fit our annotation scale
    '''
    #Get a list of all the hues from an image, and sort them
    hues_for_all_regions = getColorFeatures(image=image, mask=mask)
    hues_for_all_regions = sorted(hues_for_all_regions)


    #assign variables for measuring colour count
    first_colour = hues_for_all_regions[0]
    number_of_colours =1
    lower_bound_for_next_colour = first_colour + hue_range
    
    #Using the hue range, sees how many distinct colours are in the image
    for hue in hues_for_all_regions:
        if hue > lower_bound_for_next_colour:
            lower_bound_for_next_colour = hue + hue_range
            if lower_bound_for_next_colour > 360:
                break
            number_of_colours +=1
    
    #Checks if the last hue is close to the first hue (since hue is a circular scale)
    if (hue - 360 + hue_range) > first_colour:
        number_of_colours -= 1

    #adherence to our scale (1-4,5+ colours)
    if number_of_colours > 5:
        number_of_colours = 5
        return(number_of_colours)
    else:              
        return(number_of_colours)

#########################################################################
############################## DOTS #####################################
#########################################################################

def calculate_compactness(region):
    perimeter = region.perimeter
    area = region.area
    if perimeter == 0:
        return 0
    return (perimeter ** 2) / (4 * np.pi * area)

def check_for_dots(label_image):
    # Count the number of regions detected
    if np.max(label_image) > 0:  # Checks if there are any labeled regions
        return 2  # Dots are present
    else:
        return 1  # No dots

def load_and_process_image(image, mask):

    # Convert RGBA image to RGB if it has an alpha channel
    if image.shape[2] == 4:
        image = rgba2rgb(image)

    # Convert the image to grayscale
    gray_image = rgb2gray(image)

    # Adjust the threshold to focus on dark brown spots
    thresh = threshold_otsu(gray_image)
    binary = gray_image < thresh * 0.6  # Adjust the threshold value as needed for better separation

    # Apply the mask to the binary image
    binary_masked = np.logical_and(binary, mask)

    # Morphological operations to clean up the image
    binary_closed = closing(binary_masked, disk(1))  # Adjust the disk size as needed for better closing
    binary_opened = opening(binary_closed, disk(1))  # Adjust the disk size as needed for better opening

    # Remove artifacts connected to image border
    binary_cleared = clear_border(binary_opened)

    # Label and identify regions in the image
    label_image = label(binary_cleared)

    # Calculate compactness for each region
    compactness_threshold = 2  # Adjust as needed for better detection of small circular shapes
    regions = regionprops(label_image)
    image_label_overlay = np.zeros_like(label_image)
    for region in regions:
        compactness = calculate_compactness(region)
        if compactness > compactness_threshold:
            image_label_overlay[label_image == region.label] = 1

    return label_image


def computeDotsScore(image,mask):

    label_image = load_and_process_image(image,mask)

    dots_score = check_for_dots(label_image)

    return dots_score