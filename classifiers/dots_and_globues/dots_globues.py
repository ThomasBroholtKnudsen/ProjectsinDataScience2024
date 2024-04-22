"""
Quick image analysis workflow to demonstrate the main steps involved.
"""

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_triangle
from skimage.morphology import disk, extrema
from skimage.segmentation import watershed, mark_boundaries
from skimage.color import label2rgb
from matplotlib import pyplot as plt

# Read & crop image
x = 10
y = 0
s = 450
im_rgb = load_image('hela-cells.zip')[x:x+s, y:y+s, :].astype(np.float32)

# Convert to RGB - need to rescale each channel between 0 and 1
im_rgb_rgb = im_rgb - np.min(im_rgb.reshape((-1, 3)), axis=0).reshape((1, 1, 3))
im_rgb = im_rgb / np.percentile(im_rgb.reshape((-1, 3)), 99.5, axis=0).reshape((1, 1, 3))

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
fig = create_figure(figsize=(8, 4))

show_image(np.clip(im_rgb, 0, 1), title='(A) Original image', pos=241)
show_image(im, title='(B) Extract channel', clip_percentile=0.5, pos=242)
show_image(im_dog, title='(C) Apply filters', clip_percentile=0.5, pos=243)
show_image(bw_spots, title='(D) Apply threshold', pos=244)
show_image(bw_spots_cleaned, title='(E) Refine detection', pos=245)
show_image(label2rgb(lab_spots, bg_label=0, bg_color='black'), title='(F) Distinguish objects', pos=246)
show_image(mark_boundaries(np.clip(im_rgb, 0, 1), lab_spots, mode='thick'), title='(G) Relate objects to image', pos=247)
show_image('images/workflow_results.png', title='(H) Make measurements', pos=248)
glue_fig('fig_overview_workflow', fig)