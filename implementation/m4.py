from __future__ import print_function

import numpy as np
import cv2

# segmentation headers
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

from sklearn.feature_extraction import image
import textwrap

# Step #1
img = cv2.imread('croped_img.png', 0)
img_bw = img <= 128
img_bw = 255*img_bw.astype('uint8')

# Step #2
im2, contours, hierarchy = cv2.findContours(img_bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# Step #3
out = img.copy()

# Step #4
ref = np.zeros_like(img_bw)
cv2.drawContours(ref, contours, 0, 255, 1)

# Step #5
M = cv2.moments(contours[0])
centroid_x = int(M['m10']/M['m00'])
centroid_y = int(M['m01']/M['m00'])

# Get dimensions of the image
width = img.shape[1]
height = img.shape[0]

# Define total number of angles we want
N = 20

# Step #6
for i in range(N):
  # Step #6a
  tmp = np.zeros_like(img_bw)

  # Step #6b
  theta = i*(360/N)
  theta *= np.pi/180.0

  # Step #6c
  cv2.line(tmp, (centroid_x, centroid_y),
           (int(centroid_x+np.cos(theta)*width),
            int(centroid_y-np.sin(theta)*height)), 255, 5)

  # Step #6d
  (row,col) = np.nonzero(np.logical_and(tmp, ref))

  # Step #6e
  cv2.line(out, (centroid_x, centroid_y), (col[0],row[0]), 0, 1)

# Show the image
# Step #7
cv2.imshow('Output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()