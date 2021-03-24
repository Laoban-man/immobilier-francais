# The Iris ML Flask App

import pickle
from flask import Flask, render_template, request
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/single-post.html', methods=['GET'])
def single_post():
    return render_template('single-post.html')

@app.route('/data_exploration.html', methods=['GET'])
def data_exploration():
    return render_template('data_exploration.html')

@app.route('/new_test.html', methods=['GET','POST'])
def new_test():
    upload_dir = os.path.join("..","img")
    if request.method == 'POST':
        # save the single "profile" file
        profile = request.files['profile']
        profile.save(os.path.join(uploads_dir, "test_image.png"))

        return redirect(url_for('upload'))
    return render_template('new_test.html')

@app.route('/prediction.html', methods=['GET'])
def prediction():

    return render_template('prediction.html')



if __name__ == '__main__':
    app.debug = True
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True)
