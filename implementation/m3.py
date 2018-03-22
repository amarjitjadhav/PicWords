from __future__ import print_function

import numpy as np
import cv2

# segmentation headers
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

from sklearn.feature_extraction import image
import textwrap


# 3) Keword module
text = "aishwarya miss bachan world mumbai actress year moved abhishek modelling married architecture winning studies"
Keword = text.split(" ")
#print(Keword)


# 4) convert text to image
blank_image = cv2.imread('download.png')
fontface = cv2.FONT_HERSHEY_TRIPLEX
fontscale = 1
fontcolor = (0, 0, 0)
cv2.putText(blank_image, Keword[0],(10,20), fontface, fontscale, fontcolor)
cv2.imshow('image',blank_image)
cv2.waitKey(0)

crop_img = blank_image[:28, 10:184]
cv2.imwrite(filename='./croped_img.png',img=crop_img)
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

#extract the countour of the keword image
imgray1 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
ret1, thresh1 = cv2.threshold(imgray1, 127, 255, 0)
kernel = np.ones((10,15), np.uint8)
img_dilation = cv2.dilate(thresh1, kernel, iterations=1)

#find contours
im3,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step #3
out = thresh1.copy()
# Step #4
ref = np.zeros_like(img_dilation)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    # Getting ROI
    roi = crop_img[y:y+h, x:x+w]
    # compute the center of the contour
    M = cv2.moments(ctr)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(crop_img, (cX, cY), 4, (0, 0, 0), -1)

    # show ROI
    #cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(crop_img,(x,y),( x + w, y + h ),(0,255,0),2)
    #cv2.waitKey(0)
cv2.imwrite(filename='./contour.png',img=crop_img)
cv2.imshow('marked areas',crop_img)
cv2.waitKey(0)

# Get dimensions of the image
width = thresh1.shape[1]
height = thresh1.shape[0]

# Define total number of angles we want
N = 20

# Step #6
for i in range(N):
  # Step #6a
  tmp = np.zeros_like(img_dilation)

  # Step #6b
  theta = i*(360/N)
  theta *= np.pi/180.0

  # Step #6c
  cv2.line(tmp, (cX, cY),
           (int(cX+np.cos(theta)*width),
            int(cY-np.sin(theta)*height)), 255, 5)

  # Step #6d
  (row,col) = np.nonzero(np.logical_and(tmp, ref))

  # Step #6e
  cv2.line(out, (cX, cY), (col[0],row[0]), 0, 1)

# Show the image
# Step #7
cv2.imshow('Output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # read Image 
# # Step #1
# img = cv2.imread('contour.png', 0)
# img_bw = img <= 128
# img_bw = 255*img_bw.astype('uint8')

# # Step #2
# im3,contours,hier = cv2.findContours(img_bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# # Step #3
# out = img.copy()

# # Step #4
# ref = np.zeros_like(img_bw)
# cv2.drawContours(ref, contours, 0, 255, 1)

# # Step #5
# M = cv2.moments(contours[0])
# centroid_x = int(M['m10']/M['m00'])
# centroid_y = int(M['m01']/M['m00'])

# # Get dimensions of the image
# width = img.shape[1]
# height = img.shape[0]

# # Define total number of angles we want
# N = 20

# # Step #6
# for i in range(N):
#   # Step #6a
#   tmp = np.zeros_like(img_bw)

#   # Step #6b
#   theta = i*(360/N)
#   theta *= np.pi/180.0

#   # Step #6c
#   cv2.line(tmp, (centroid_x, centroid_y),
#            (int(centroid_x+np.cos(theta)*width),
#             int(centroid_y-np.sin(theta)*height)), 255, 5)

#   # Step #6d
#   (row,col) = np.nonzero(np.logical_and(tmp, ref))

#   # Step #6e
#   cv2.line(out, (centroid_x, centroid_y), (col[0],row[0]), 0, 1)

# # Show the image
# # Step #7
# cv2.imshow('Output', out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

