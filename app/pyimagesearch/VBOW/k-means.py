# import the necessary packages
from descriptor import Descriptor
import argparse
import glob
import cv2
import progressbar
import numpy as np
import pickle

# initialize the color descriptor
d = Descriptor()
descriptors = {}
desc = []
# open the output index file for writing
#output = open('app/index2.csv', "w")

bar = progressbar.ProgressBar(max_value=11).start()

i =0
# use glob to grab the image paths and loop over them
for imagePath in glob.glob('app/static/mini_data/' + "*.jpg"):
    # extract the image ID (i.e. the unique filename) from the image
    # path and load the image itself
    imageID = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath, cv2.COLOR_BGR2HSV)

    # describe the image
    features = d.describe(image)
    #add the list of descriptors to
    descriptors[imageID] = features.flatten()
    desc.append(features.flatten())
    i += 1
    bar.update(i)


d1= desc[0]
for descriptor in desc[1:]:
    d1 = np.hstack((d1, descriptor))  # Stacking the descriptors
d1 = d1.flatten()

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply KMeans
k = 100
compactness, labels, centers = cv2.kmeans(d1, k, None, criteria, 10, flags)

dico_label = {}
for id, descriptor in descriptors.items():
    dico_label[id] = labels[:len(descriptor)]

with open('app/index_test.txt', 'wb') as handle:
  pickle.dump(dico_label, handle)


# close the index file
bar.finish()


