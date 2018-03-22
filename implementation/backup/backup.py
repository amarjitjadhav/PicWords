import sys

from skimage.segmentation import slic
from skimage.data import astronaut
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse

# 2) Patch Generation

# segmentation.slic(im2, n_segments=100, compactness=0.1, enforce_connectivity=True)
# cv2.imshow('image',im2)
# cv2.waitKey(0)


# numSegments = 100
# segments = slic(im2, n_segments=100, compactness=10)
# # cv2.imshow(segments)
# # cv2.waitKey(0)
# fig = plt.figure("Superpixels -- %d segments" % (numSegments))
# ax = fig.add_subplot(1, 1, 1)
# ax.imshow(mark_boundaries(im2, segments))
# plt.axis("off")
 
# # show the plots
# plt.show()




# extrct the patches from images 
patches = image.extract_patches_2d(segments_slic, (2, 2), max_patches=2,random_state=0)
print(patches[1])

import textwrap
texto = "Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
novo = textwrap.wrap(texto, width=20)
print(novo)