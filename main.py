# The Iris ML Flask App

import pickle
from flask import Flask, render_template, request
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gunicorn
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras import regularizers
#from tensorflow.keras.callbacks import ModelCheckpoint
#import keras

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/', methods=['POST'])
def models():


    if request.method == 'POST':
        if request.form.get("lr_button"):
            with open("linearmodel.pkl","rb") as f:
                lr=pickle.load(f)
            rooms = int(request.form['rooms'])
            area = int(request.form['area'])
            nb_lots = int(request.form['lots'])
            prediction=lr.predict(np.array([nb_lots,area,rooms]).reshape(1,3))
            return render_template('index.html',prediction=prediction)
        #elif request.form.get("nn_button"):
        #    my_model = tf.keras.models.load_model('my_model')
        #    rooms = int(request.form['rooms2'])
        #    area = int(request.form['area2'])
        #    nb_lots = int(request.form['lots2'])
        #    prediction=my_model.predict(np.array([nb_lots,area,rooms]).reshape(1,3))
        #    return render_template('index.html',prediction2=prediction)
        elif request.form.get("rent_button"):
            with open("rent_model.pkl","rb") as f:
                    lr=pickle.load(f)
            rooms = int(request.form['rooms3'])
            district = int(request.form['district3'])
            area = int(request.form['area3'])
            prediction=lr.predict(np.array([district,area,rooms]).reshape(1,3))
            ten_year_revenue=12*10*area*prediction
            return render_template('index.html',ten_year_revenue=ten_year_revenue)


@app.errorhandler(500)
def internal_server_error(e):
    return jsonify(error=str(e)), 500


if __name__ == '__main__':
    app.debug = True
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True)
