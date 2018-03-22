from __future__ import print_function

#import matplotlib.pyplot as plt
import numpy as np
import cv2
#from skimage.data import astronaut
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
#from skimage.util import img_as_float

#img = img_as_float(astronaut()[::2, ::2])
img = cv2.imread('silhouette_image.png')
#segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
#segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)

#print("Felzenszwalb's number of segments: %d" % len(np.unique(segments_fz)))
print("Slic number of segments: %d" % len(np.unique(segments_slic)))
#print("Quickshift number of segments: %d" % len(np.unique(segments_quick)))

# fig, ax = plt.subplots(1, 3, sharex=True, sharey=True,
#                         subplot_kw={'adjustable': 'box-forced'})
# fig.set_size_inches(8, 3, forward=True)
# fig.tight_layout()

#ax[0].imshow(mark_boundaries(img, segments_fz))
#ax[0].set_title("Felzenszwalbs's method")

#ax[1].imshow(mark_boundaries(img, segments_slic))
cv2.imshow('segmented_image',mark_boundaries(img, segments_slic))
cv2.waitKey(0)
#ax[1].set_title("SLIC")

#ax[2].imshow(mark_boundaries(img, segments_quick))
#ax[2].set_title("Quickshift")
# for a in ax:
#     a.set_xticks(())
#     a.set_yticks(())
#plt.show()