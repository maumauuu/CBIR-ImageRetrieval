import os

from flask import Flask, render_template, request, jsonify

from pyimagesearch.VBOW.descriptor import Descriptor
from pyimagesearch.VBOW.my_searcher import Searcher
import cv2
import pickle

# create flask instance
app = Flask(__name__)

INDEX = os.path.join(os.path.dirname(__file__), 'index_test.txt')
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))



# main route
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    if request.method == "POST":

        RESULTS_ARRAY = []

        # get url

        image_url = request.form.get('img')
        with open('app/index_test.txt', 'rb') as handle:
            b = pickle.loads(handle.read())
        imageID = image_url[image_url.rfind("/") + 1:]

        try:

            print("before initialization")
            # initialize the image descriptor
            d = Descriptor()

            # load the query image and describe it
            print(imageID)
            query = cv2.imread(UPLOAD_FOLDER+image_url, cv2.COLOR_BGR2HSV)
            print("before describe")
            hist = d.hist(query, b[imageID])
            print("after describe")

            # perform the search

            print("in search")
            searcher = Searcher(b, INDEX)
            print("search finished")
            results = searcher.search(hist)
            print("done")

            # loop over the results, displaying the score and image name
            for (score, resultID) in results:
                RESULTS_ARRAY.append(
                    {"image": str(resultID), "score": str(score)})

            # return success
            return jsonify(results=(RESULTS_ARRAY[::-1][:3]))

        except:

            # return error
            return jsonify({"sorry": "Sorry, no results! Please try again."}), 500


# run!
if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
