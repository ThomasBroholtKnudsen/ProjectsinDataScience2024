import numpy as np
from scipy import ndimage
from skimage.filters import threshold_triangle
from skimage.morphology import disk, extrema
from skimage.segmentation import watershed, mark_boundaries
from skimage.color import label2rgb
from matplotlib import pyplot as plt
import imageio

# Function to load an image
def load_image(filepath):
    return imageio.imread(filepath)

# Function to create a figure
def create_figure(size=(8, 8)):
    return plt.figure(figsize=size)

# Function to display an image in a subplot
def show_image(image, title='', pos=None):
    if pos is not None:
        plt.subplot(pos)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Load image
im_rgb = load_image('PAT_1018_107_68.png').astype(np.float32)

# Convert to RGB - rescale each channel between 0 and 1
im_rgb = (im_rgb - np.min(im_rgb.reshape((-1, 3)), axis=0)) / (np.percentile(im_rgb.reshape((-1, 3)), 99.5, axis=0) - np.min(im_rgb.reshape((-1, 3)), axis=0))

# Extract channel of interest
im = im_rgb[:, :, 0]

# Apply difference of Gaussians filter
sigma = 3
sigma2 = sigma * 1.6
im_dog = ndimage.gaussian_filter(im, sigma) - ndimage.gaussian_filter(im, sigma2)

# Threshold for spot detection
bw_spots = im_dog > threshold_triangle(im_dog)

# Clean up with a morphological opening
strel = disk(3)
bw_spots_opened = ndimage.binary_opening(bw_spots, structure=strel)

# Perform distance-and-watershed-based split
bw_spots_dist = ndimage.distance_transform_edt(bw_spots_opened)
bw_spots_max = extrema.h_maxima(bw_spots_dist, 0.5)
bw_spots_max = ndimage.binary_dilation(bw_spots_max, structure=np.ones((3, 3)))
bw_spots_max = np.bitwise_and(bw_spots_opened, bw_spots_max)
lab_spots, n = ndimage.label(bw_spots_max)
lab_spots = watershed(-bw_spots_dist, markers=lab_spots, mask=bw_spots_opened, watershed_line=True)
bw_spots_cleaned = lab_spots > 0

# Show images
create_figure((12, 6))
show_image(np.clip(im_rgb, 0, 1), title='Original image', pos=241)
show_image(im, title='Extract channel', pos=242)
show_image(im_dog, title='Apply filters', pos=243)
show_image(bw_spots, title='Apply threshold', pos=244)
show_image(bw_spots_cleaned, title='Refine detection', pos=245)
show_image(label2rgb(lab_spots, bg_label=0, bg_color='black'), title='Distinguish objects', pos=246)
show_image(mark_boundaries(np.clip(im_rgb, 0, 1), lab_spots, mode='thick'), title='Relate objects to image', pos=247)
plt.show()
