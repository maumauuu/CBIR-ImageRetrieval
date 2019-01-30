# import the necessary packages
from descriptor import Descriptor
from my_searcher import Searcher
import argparse
import cv2
import pickle

d = Descriptor()

with open('app/index_test.txt', 'rb') as handle:
  b = pickle.loads(handle.read())

print(b)

# load the query image and describe it
query = cv2.imread('app/static/queries/103100.jpg',cv2.COLOR_BGR2HSV)
hist = d.hist(query, b['103100.jpg'])


# perform the search
searcher = Searcher(b, 'app/index_test.txt')
results = searcher.search(hist)

# display the query
cv2.imshow("Query", query)


# loop over the results
for (score, resultID) in results:
    # load the result image and display it
    result = cv2.imread('app/static/mini_data' + "/" + resultID)
    print(resultID)
    cv2.imshow("Result", result)
    cv2.waitKey(0)

