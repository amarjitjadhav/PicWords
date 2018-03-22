import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from itertools import cycle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
#%matplotlib inline
pylab.rcParams['figure.figsize'] = 16, 12

#image = Image.open('bread.jpg')
image = Image.open('1.png')
#Image is (687 x 1025, RGB channels)
image = np.array(image)
original_shape = image.shape
print(original_shape)
# Flatten image.
X = np.reshape(image, [-1, 4])
#plt.imshow(image)

#Estimate the kernel bandwidth to use from our image (the datapoints).
# bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)
# print(bandwidth)

#Now run Meanshift on the image to do the image segmentation, which is stored in X.
#ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms = MeanShift()
ms.fit(X)

# no. of clusters is equal to no of colors.	
labels = ms.labels_
print(labels.shape)
cluster_centers = ms.cluster_centers_
print(cluster_centers.shape)
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

#Just take size, ignore RGB channels.
segmented_image = np.reshape(labels, original_shape[:2])

plt.figure(2)
plt.subplot(1, 2, 1)
#plt.imshow(image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.show()
plt.axis('off')

# convert to grayscale image
gray_image = (segmented_image/6.0 * 255.0).clip(0.0, 255.0).astype(np.uint8)
cv2.imwrite(filename='./gray_image.png',img=gray_image) 

# covert it to binary image
ret,thresh_img = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
# plt.imshow(thresh_img)
# plt.show()

#applying a Gaussian filtering to remove tiny holes and blurs
blur = cv2.GaussianBlur(thresh_img,(5,5),0)
plt.imshow(blur)
plt.show()
