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
    image = imread(PAT_1379_1300_924.png)
    mask = imread(PAT_1379_1300_924_mask.png)

    # Convert the image to grayscale
    gray_image = rgb2gray(image)

    # Apply Otsu's threshold to convert to binary image
    thresh = threshold_otsu(gray_image)
    binary = gray_image > thresh

    # Morphological operations to clean up the image
    binary_closed = closing(binary, square(3))
    binary_opened = opening(binary_closed, disk(2))

    # Remove artifacts connected to image border
    binary_cleared = clear_border(binary_opened)

    # Label and identify regions in the image
    label_image = label(binary_cleared)
    image_label_overlay = label_image > 0

    return image, gray_image, binary_cleared, image_label_overlay, label_image

def display_results(image, gray_image, binary_cleared, image_label_overlay, label_image):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(gray_image, cmap='gray')
    ax[1].set_title('Grayscale Image')
    ax[1].axis('off')

    ax[2].imshow(binary_cleared, cmap='gray')
    ax[2].set_title('Binary Image')
    ax[2].axis('off')

    ax[3].imshow(image)
    ax[3].set_title('Detected Features on Original')
    ax[3].axis('off')
    for region in regionprops(label_image):
        # Draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax[3].add_patch(rect)

    plt.tight_layout()
    plt.show()

# Paths to the image and the mask
image_path = 'path_to_your_image.png'
mask_path = 'path_to_your_mask.png'

# Load, process, and display
image, gray_image, binary_cleared, image_label_overlay, label_image = load_and_process_image(image_path, mask_path)
display_results(image, gray_image, binary_cleared, image_label_overlay, label_image)
