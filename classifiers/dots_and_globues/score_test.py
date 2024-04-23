import numpy as np
from scipy import ndimage
from skimage.filters import threshold_triangle
from skimage.morphology import disk, extrema
from skimage.segmentation import watershed
from skimage.color import label2rgb
from matplotlib import pyplot as plt
import imageio

# Function to load an image
def load_image(filepath):
    image = imageio.imread(filepath)
    print("Image loaded successfully")
    return image

# Function to process and count spots
def process_and_count_spots(filepath):
    # Load image and convert to float
    im_rgb = load_image(filepath).astype(np.float32)
    print("Image converted to float")

    # Normalize image
    min_val = np.min(im_rgb.reshape((-1, 3)), axis=0)
    percentile_val = np.percentile(im_rgb.reshape((-1, 3)), 99.5, axis=0)
    im_rgb = (im_rgb - min_val) / (percentile_val - min_val)
    print("Image normalized")

    # Extract red channel
    im = im_rgb[:, :, 0]

    # Apply Difference of Gaussians
    sigma = 2
    sigma2 = sigma * 5
    im_dog = ndimage.gaussian_filter(im, sigma) - ndimage.gaussian_filter(im, sigma2)
    print("Difference of Gaussians applied")

    # Apply threshold
    threshold_value = threshold_triangle(im_dog)
    bw_spots = im_dog > threshold_value
    print(f"Threshold applied with value: {threshold_value}")

    # Morphological opening
    strel = disk(3)
    bw_spots_opened = ndimage.binary_opening(bw_spots, structure=strel)
    print("Morphological opening applied")

    # Label spots
    bw_spots_max = ndimage.distance_transform_edt(bw_spots_opened) > 0.5
    bw_spots_max = ndimage.binary_dilation(bw_spots_max, structure=np.ones((3, 3)))
    lab_spots, num_spots = ndimage.label(bw_spots_max)
    print(f"Spots labeled, number of spots: {num_spots}")

    return num_spots

# File path to the image
filepath = 'PAT_1018_107_68.png'

# Process the image and print the number of detected spots
num_spots = process_and_count_spots(filepath)
print(f'Number of detected spots: {num_spots}')
