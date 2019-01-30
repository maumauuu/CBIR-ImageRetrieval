import os

from flask import Flask, render_template, request, jsonify

from pyimagesearch.Matcher.descriptor import Descriptor
from pyimagesearch.Matcher.matcher import Matcher
import cv2

# create flask instance
app = Flask(__name__)

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
        try:

            # initialize the image descriptor
            cd = Descriptor()

            # load the query image and describe it
            query = cv2.imread(UPLOAD_FOLDER+image_url, cv2.COLOR_BGR2HSV)
            features = cd.describe(query)

            # perform the search
            matcher = Matcher()
            results = matcher.match(features)

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
