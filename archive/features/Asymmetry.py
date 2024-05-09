import numpy as np
from skimage.transform import rotate
import matplotlib.pyplot as plt

def readImage(filepath):
    image = plt.imread(filepath)
    return image

def rotateImage(image, angle, center):

    """
    Rotates the given image by given angle in clockwise direction. Center is set to center of image by default. 
    See skimage documentation: https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rotate
    :image: 2D numpy array
    :angle: degrees
    :return: new image - 2D numpy array
    """

    rotated_image = rotate(image, -angle, center = center)
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
    lesion_area = np.sum(mask)
    
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
    

