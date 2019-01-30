import numpy as np
import cv2
from cv2 import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


class Descriptor:

    def __init__(self):
        dico = []

    def describe(self, image):

        sift = cv2.xfeatures2d.SIFT_create()

        keypoints, descriptors = sift.detectAndCompute(image, None)

        return descriptors

    def hist(self, image, labels):
        # Perform Tf-Idf vectorization
        nbr_occurences = np.sum((labels > 0) * 1, axis=0)
        # Calculating the number of occurrences
        idf = np.array(np.log((1.0*len(image)+1) / (1.0*nbr_occurences + 1)), 'float32')
        # Giving weight to one that occurs more frequently
        hist = list(labels.flatten())
        compte = {k: hist.count(k) for k in set(hist)}

        #plt.bar(list(compte.keys()), compte.values(), color='r')
        #plt.show()
        #/idf

        ##hist = cv2.normalize(hist, hist).flatten()

        return compte

    def invert_dict_nonunique(d):
        newdict = {}
        for k, v in d.iteritems():
            newdict.setdefault(v, []).append(k)
        return newdict

