# import the necessary packages
import cv2
import glob
from pyimagesearch.Matcher.descriptor import Descriptor


class Matcher:

    def match(self, features, limit=5):

        d = Descriptor()

        results = {}
        for test in glob.glob('app/static/mini_data/' + "*.jpg"):
            bf = cv2.BFMatcher()
            des = cv2.imread(test,  cv2.COLOR_BGR2HSV)
            des = d.describe(des)
            matches = bf.knnMatch(features, des, k=2)
            # Apply ratio test
            good = []
            imageID = test[test.rfind("/") + 1:]

            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            results[imageID] = len(good)
        print(results)

        # sort our results, so that the smaller distances (i.e. the
        # more relevant images are at the front of the list)
        results = sorted([(v, k) for (k, v) in results.items()])
        # return our (limited) results
        results= [(v,k) for (v,k) in results if v >=30]
        print(results)
        return results[:limit]
