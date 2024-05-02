#Everything between the lines of # is copied code
######################################################################################
################################### COLOR VARIATION ##################################
######################################################################################


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
    TODO: Add rest of the description

    This function computes the color brightness variations of an image, quantified as the IQR. This method 
    uses SLIC segmentation to select superpixels for grathering average regional color intensities. 
    These averages are converted to HSV to measure the spread of brightness ('Value') across all regions.

    :image: RGB image read as numpy.array
    :mask: Corresponding binary mask as numpy.array
    :return: list with extracted features
    """

    # Mask and crop image to only contain lesion
    im_lesion = lesionMaskCrop(image, mask)

    # Get SLIC boundaries
    segments = slic(im_lesion, n_segments=250, compactness=50, sigma=1, start_label=1)

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

    # Extract values for each channel
    color_mean_hue = [hsv[0] for hsv in color_mean_hsv]
    color_mean_satur = [hsv[1] for hsv in color_mean_hsv]
    color_mean_value = [hsv[2] for hsv in color_mean_hsv]

    # Compute different features based on the above values
    # * Compute SD for hue
    hue_sd = np.std(np.array(color_mean_hue))

    # * Compute SD for satur
    satur_sd = np.std(np.array(color_mean_satur))

    # * Compute SD for value
    value_sd =np.std(np.array(color_mean_value))

    # * Computing IQR range for color values
    q1 = np.quantile(color_mean_value, 0.25, interpolation='midpoint')
    q3 = np.quantile(color_mean_value, 0.75, interpolation='midpoint')
    iqr_val = q3 - q1
    
    return [hue_sd, satur_sd, value_sd, iqr_val]

#############################################################
#Here begins ours

#What i think we wanna do is count the number of regions that have a colour that is far enough away from another colour
#So could decide how far away a colour needs to be to be considered different? (~60 degress on hue score?) can test this

#no regions, and their colour
#check each region is far enough away from another region
#if not combine the two, new colours have to be far enough away from one of these two colours

#colour_mean_hue is a list with all the region's hues


def how_many_colours_are_there(colour_mean_hue, hue_range = 60):

    regions_that_are_distinct = []

    for region_a in colour_mean_hue:
        for region_b in colour_mean_hue:
            if region_a <= (region_b + hue_range) and region_a > (region_b - hue_range):
                regions_that_are_distinct.append(region_a)
                break

            elif (359 + region_a) <= (region_b + hue_range) and (359 + region_a) > (region_b - hue_range):
                regions_that_are_distinct.append(region_a)
                break

            else:

    number_of_colours = len(regions_that_are_distinct) +1
    return(number_of_colours)
    