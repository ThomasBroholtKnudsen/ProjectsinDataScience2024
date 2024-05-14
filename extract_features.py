# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import skimage
import imageio.v2 as imageio

# Import functions for image processing
from skimage import morphology
from skimage.transform import rotate
from skimage.color import rgb2gray, rgba2rgb  
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square, opening, disk
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage import segmentation
from skimage.segmentation import slic

#---------------------------------------------------
# Sources:
# 
# Ludek???
# https://www.w3resource.com/python-exercises/math/python-math-exercise-77.php
# https://bioimagebook.github.io/chapters/2-processing/1-processing_and_analysis/processing_and_analysis.html
#---------------------------------------------------

#---------------------------------------------------
# Help functions
#---------------------------------------------------

#Main function to extract features from an image and mask, that calls other functions    
def extract_features(image, mask):

    # Measuring asymmetry
    asymmetry_score = computeAsymmetry(mask)

    # Measuring colour
    colour_score = how_many_colours_are_there(image, mask, hue_range = 60)
   
    # Measuring dots
    dots_score = computeDotsScore(image,mask)

    # Output: 1D array of all the scores
    return np.array([asymmetry_score,colour_score,dots_score], dtype=np.float16)


#########################################################################
############################### ASYMMETRY ###############################
#########################################################################

def rotateImage(mask, angle, center):

    """
    This function rotates the given image (a 2D numpy array) by the number of given degrees
    and returns the rotated image as a 2D numpy array.
    """

    rotated_image = rotate(mask, -angle, center = center)
    return rotated_image


def findCentroid(mask):

    """
    This function computes the centroid of a given mask (a 2D binary numpy array),
    by computing the mean of the x and y-coordinates of the mask region.
    The pair of coordinates for the centroid are returned.
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
    This function takes a y-coordiante and an image (2D numpy array) as input.
    It splits the image horizontally through the given y-coordinate into two halves.
    It returns two 2D numpy arrays representing the two halves.
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

        # Stacks row-wise lower and then additional rows
        lower = np.vstack((lower, additional_rows))

    # Upper half needs more rows
    elif n_rows_upper < n_rows_lower:

        # Get inputs for transformation
        row_difference = n_rows_lower - n_rows_upper
        n_columns = upper.shape[1]
        additional_rows = [[0]*n_columns for _ in range(row_difference)]

        # Stacks row-wise additional rows and then upper
        upper = np.vstack((additional_rows, upper))
    
    # Flip the lower along the x-axis, so it can be then compared directly without any further transformation
    lower = np.flip(lower, axis = 0)

    return lower, upper

def halveTheRegionVertically(xCoord, mask):
    """
    This function takes an x-coordiante and an image (2D numpy array) as input.
    It splits the image vertically through the given x-coordinate into two halves.
    It returns two 2D numpy arrays representing the two halves.
    """

    # Get the halves
    left = mask[:, :xCoord]
    right = mask[:, xCoord:]

    # Make sure both halves have the same amount of columns
    n_cols_left = left.shape[1]
    n_cols_right = right.shape[1]

    # Right half needs more columns
    if n_cols_left > n_cols_right:

        # Get inputs for transformation
        col_difference = n_cols_left - n_cols_right
        n_rows = right.shape[0]
        additional_cols = np.zeros((n_rows, col_difference))

        # Stacks column-wise right and then additional columns
        right = np.hstack((right, additional_cols))

    # Left half needs more columns
    elif n_cols_left < n_cols_right:

        # Get inputs for transformation
        col_difference = n_cols_right - n_cols_left
        n_rows = left.shape[0]
        additional_cols = np.zeros((n_rows, col_difference))

        # Stacks column-wise additional columns and then left
        left = np.hstack((additional_cols, left))

    # Flip the right along the y-axis, so it can be then compared directly without any further transformation
    right = np.flip(right, axis=1)

    return left, right


def computeAsymmetry(mask):

    """
    This function takes a mask (2D binary numpy array) as input.
    It computes the asymmetry of the masked region as follows:
    1. Finds centroid x and y-coordinate pair of the mask. 
    2. The mask is split into two halves (both horizontally and vertically) using the computed centroid
    3. The two regions are subtracted from each other. The differences are summed and then made relative to the size of the mask.
       This is done both horizontally and vertically. 
    4. The horizontal and vertical asymmetry scores are aggregated. 
    
    This procedure is done for a selected number of rotations of the mask. 
    The minimum of the aggregated asymmetry scores is returned.
    """

    # Make sure the mask is binary and not consisting of floats.
    mask[mask > 0] = 1 


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
        
        # Vertical split
        left, right = halveTheRegionVertically(xCoord, rotated_mask)
        vertical_asymmetry = np.sum(np.abs(left - right)) / lesion_area

        # Save the result
        asymmetry_results.append(horizontal_asymmetry + vertical_asymmetry)
    
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
    segments = slic(im_lesion, n_segments=6000, compactness=0.00001, sigma=1, start_label=1)

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
def how_many_colours_are_there(image, mask, hue_range = 7):

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
    elif number_of_colours < 1:
        number_of_colours = 1
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

def check_for_dots(regions):
    compactness_threshold = 2  # Threshold for shape circularity
    dot_count = 0
    for region in regions:
        compactness = calculate_compactness(region)
        if compactness > compactness_threshold:
            dot_count += 1
    return 1 if dot_count >= 10 else 0

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
    # Remove artifacts connected to the image border
    binary_cleared = clear_border(binary_opened)
    # Label and identify regions in the image
    label_image = label(binary_cleared)
    regions = regionprops(label_image)
    return regions, label_image  # Return regions for compactness check and label_image if needed

def computeDotsScore(image, mask):
    regions, label_image = load_and_process_image(image, mask)  # Get regions directly from image processing function
    dots_score = check_for_dots(regions)  # Pass regions to check for dots
    return dots_score
