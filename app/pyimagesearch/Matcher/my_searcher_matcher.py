# import the necessary packages
from descriptor import Descriptor
from matcher import Matcher
import argparse
import glob
import cv2
import progressbar
import numpy as np
import pickle


d = Descriptor()

# load the query image and describe it
query = cv2.imread('app/static/queries/103100.jpg', cv2.COLOR_BGR2HSV)
features = d.describe(query)


# perform the search
matcher = Matcher()
results = matcher.match(features)

# display the query
cv2.imshow("Query", query)


# loop over the results
for (score, resultID) in results:
    # load the result image and display it
    print('result')
    result = cv2.imread('app/static/mini_data' + "/" + resultID)
    print(resultID)
    cv2.imshow("Result", result)
    cv2.waitKey(0)

