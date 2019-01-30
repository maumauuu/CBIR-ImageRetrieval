# import the necessary packages
import numpy as np
from pyimagesearch.VBOW.descriptor import Descriptor

import progressbar

class Searcher:
    def __init__(self, index,indexPath):
        # store our index path
        self.index = index
        self.indexPath = indexPath

    def search(self, queryHist, limit=3):
        # initialize our dictionary of results
        results = {}
        d = Descriptor()
       # bar = progressbar.ProgressBar(max_value=812).start()

        i = 0

        # loop over the rows in the index
        for id,labels in self.index.items():
            # parse out the image ID and features, then compute the
            # chi-squared distance between the features in our index
            # and our query features

            hist = d.hist(id, labels)
            dist = self.chi2_distance(hist, queryHist)

            # now that we have the distance between the two histogramm
            # we can udpate the results dictionary -- the
            # key is the current image ID in the index and the
            # value is the distance we just computed, representing
            # how 'similar' the image in the index is to our query
            results[id] = dist
       #     i += 1
      #      bar.update(i)
        # close the reader
       # bar.finish()
        # sort our results, so that the smaller distances (i.e. the
        # more relevant images are at the front of the list)

        results = sorted([(v, k) for (k, v) in results.items()])
        # return our (limited) results
        return results[:limit]

    def chi2_distance(self, histA, histB, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
        # return the chi-squared distance
        return d
