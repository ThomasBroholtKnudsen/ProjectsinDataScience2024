import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square, opening, disk
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

def load_and_process_image(image_path, mask_path):
    # Load the image and the mask
    image = imread(image_path)
    mask = imread(mask_path)

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
    dots_detected = 0  # Initialize the flag
    for region in regions:
        compactness = calculate_compactness(region)
        if compactness > compactness_threshold:
            dots_detected = 1  # Set the flag if dots are detected
            break  # Exit loop once dots are detected

    return image, gray_image, binary_cleared, label_image, dots_detected

# Paths to the image and the mask
image_path = 'PAT_1379_1300_924.png' #test images
mask_path = 'PAT_1379_1300_924_mask.png'

# Load, process, and display
image, gray_image, binary_cleared, label_image, dots_detected = load_and_process_image(image_path, mask_path)

# Output 0 if no dots are detected and 1 if dots are detected
print(dots_detected)
