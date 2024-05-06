#imports
import skimage
from skimage import segmentation
from skimage.segmentation import slic
import imageio.v2 as imageio
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.measure import regionprops

# Paths to the images
#This will most likely change to iteration, but keeping it here for now
image_path = '..\\dots_and_globues\\PAT_166_257_586.png'
mask_path = '..\\dots_and_globues\\PAT_166_257_586_mask.png'

# Load the image and the mask
im_rgb = imread(image_path)
mask = imread(mask_path)

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

def lesionMaskCrop(im_rgb, mask):
    '''
    This function masks and crops an area of a color image corresponding to a binary mask of same dimension.

    :image: RGB image read as numpy.array
    :mask: Corresponding binary mask as numpy.array
    '''
    # Getting top/bottom and left/right boundries of lesion
    top, bottom = find_topbottom(mask)
    left, right = find_leftright(mask)

    # Masking out lesion in color image
    im_masked = im_rgb.copy()
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

def getColorFeatures(im_rgb, mask):

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
    im_lesion = lesionMaskCrop(im_rgb, mask)

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
def how_many_colours_are_there(im_rgb, mask, hue_range = 60):

    '''
    Counts the number of diifferent colours are in an image.
    Specifically assesses how much the hues of an image vary.
    
    :image: RGB image read as numpy.array
    :mask: Corresponding binary mask as numpy.array
    :hue_range: adjustable, but set at 60 since this is generally given as the value difference between the 6 main colours of the hue spectrum
    :return: int value from 1-5 to fit our annotation scale
    '''
    #Get a list of all the hues from an image, and sort them
    hues_for_all_regions = getColorFeatures(im_rgb=im_rgb, mask=mask)
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