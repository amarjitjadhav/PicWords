from __future__ import print_function

import numpy as np
import cv2

# segmentation headers
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

from sklearn.feature_extraction import image
import textwrap
# read Image 

im = cv2.imread('2.png')

# 1) Silhouette Image Generation

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imwrite(filename='./silhouette_image.png',img=im2) 
cv2.imshow('image',im2)
cv2.waitKey(0)


# 2) Patch Generation
img = cv2.imread('silhouette_image.png')
segments_slic = slic(img, n_segments=100, compactness=10, sigma=1)
cv2.imshow('segmented_image',mark_boundaries(img, segments_slic))
cv2.waitKey(0)

# segments_slic1 = mark_boundaries(img, segments_slic)
# cv2.imwrite(filename='./segment_image.png',img=segments_slic1) 
#print("Slic number of segments: %d" % len(np.unique(segments_slic)))


# import pdb
# pdb.set_trace()

# calculate area of each pixel

# num_pixels = img.shape[0] * img.shape[1]
# num_segs = np.max(segments_slic)
# areas = []
# for i in range(num_segs + 1):
#     area = np.sum(segments_slic == i) * 1.0 / num_pixels
#     areas.append(area)

# extrct the patches from images 
# loop over the unique segment values
# for (i, segVal) in enumerate(np.unique(segments_slic)):
# 	mask = np.zeros(img.shape[:2], dtype = "uint8")
# 	mask[segments_slic == segVal] = 255
# 	cv2.imshow("Mask", mask)
# 	cv2.imshow("Applied", cv2.bitwise_and(img, img, mask = mask))
# 	cv2.waitKey(0)

# 3) Keword module
text = "aishwarya miss bachan world mumbai actress year moved abhishek modelling married architecture winning studies"
Keword = text.split(" ")
print(Keword[0])
# novo = textwrap.wrap(text, width=20)
# print(novo[0])
