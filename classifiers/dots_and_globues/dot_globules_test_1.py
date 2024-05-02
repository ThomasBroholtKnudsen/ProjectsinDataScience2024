import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.morphology import disk, extrema
from skimage.segmentation import watershed, mark_boundaries
from skimage.color import label2rgb
from matplotlib import pyplot as plt
import imageio.v2 as imageio  # Updated import statement

# Function to load an image
def load_image(filepath):
    return imageio.imread(filepath)

# Function to load and apply a mask image
def apply_mask(image, mask):
    mask = mask > 0  # Assuming the mask is a binary image where white areas are True
    return image * mask  # Element-wise multiplication to apply the mask

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

# Paths to the images
image_path = 'PAT_1379_1300_924.png'
mask_path = 'PAT_1379_1300_924_mask.png'

# Load image and mask
im_rgb = load_image(image_path).astype(np.float32)
mask = load_image(mask_path)

# Convert RGB image to grayscale
im_gray = np.mean(im_rgb, axis=2)

# Apply the mask to the grayscale image
im_masked = apply_mask(im_gray, mask)

# Apply difference of Gaussians filter to the masked image
sigma = 3
sigma2 = sigma * 1.6
im_dog = ndimage.gaussian_filter(im_masked, sigma) - ndimage.gaussian_filter(im_masked, sigma2)

# Threshold using Otsu's method
bw_spots = np.zeros_like(im_dog)
threshold_value = threshold_otsu(im_dog[mask > 0])
bw_spots[mask > 0] = im_dog[mask > 0] < threshold_value

# Morphological opening with a larger disk
strel = disk(5)
bw_spots_opened = ndimage.binary_opening(bw_spots, structure=strel)

# Perform distance-and-watershed-based split
bw_spots_dist = ndimage.distance_transform_edt(bw_spots_opened)
bw_spots_max = extrema.h_maxima(bw_spots_dist, 1.5)
bw_spots_max = ndimage.binary_dilation(bw_spots_max, structure=np.ones((5, 5)))
bw_spots_max = np.bitwise_and(bw_spots_opened, bw_spots_max)
lab_spots, _ = ndimage.label(bw_spots_max)
lab_spots = watershed(-bw_spots_dist, markers=lab_spots, mask=bw_spots_opened, watershed_line=True)

# Calculate compactness and filter spots
properties = measure.regionprops(lab_spots)
compactness_scores = [4 * np.pi * prop.area / (prop.perimeter ** 2) for prop in properties]
filtered_labels = [prop.label for prop, score in zip(properties, compactness_scores) if score > 0.7]  # Adjust threshold as needed

# Create a mask for the filtered labels
bw_spots_filtered = np.isin(lab_spots, filtered_labels)

# Show images
create_figure((12, 6))
show_image(np.clip(im_rgb, 0, 1), title='Original image', pos=241)
show_image(im_masked, title='Masked image', pos=242)
show_image(im_dog, title='Apply filters', pos=243)
show_image(bw_spots, title='Apply threshold', pos=244)
show_image(bw_spots_filtered, title='Refine detection', pos=245)
show_image(label2rgb(lab_spots, bg_label=0, bg_color='black'), title='Distinguish objects', pos=246)
plt.show()  # This ensures that the plot is displayed
