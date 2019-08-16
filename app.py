from __future__ import division, print_function

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import keras
print('Keras version: {}'.format(keras.__version__))
import tensorflow as tf
print('Tensorflow version: {}'.format(tf.__version__))

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# google Translator
from googletrans import Translator


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/your_model.h5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

from keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    #img = image.load_img(img_path, target_size=(224, 224))
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')
    # x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)

        pred_class = decode_predictions(preds, top=3)
        #print('Predicted:', decode_predictions(preds, top=3)[0])
        #print ('Predicted n 1:', str(pred_class[0][0][0])) # result n
        #print ('Predicted result 1:', str(pred_class[0][0][1])) # result text
        #print ('Predicted accuracy 1:', str(pred_class[0][0][2])) # result accuracy

        predictions_list_full = {}
        predictions_list = ''
        # Iterating using while loop 
        i = 0
        while i < len(pred_class[0]): 
            #print(f'prediction {i + 1}: ',pred_class[0][i])
            #print(f'Predicted result {i + 1}     Text: ',pred_class[0][i][1], ' Accuracy: ', pred_class[0][i][2])

            predictions_list_full.update({pred_class[0][i][1]: pred_class[0][i][2]})
            #print(f'>>>>>>>> Predictions: ',predictions_list_full)
            print(f'>>>>>>>> Predictions: ',i + 1, ' <<<<<<<<<<')
            prGreen(predictions_list_full)

            
            if len(predictions_list) > 0: 
                if pred_class[0][i][2] > 0.15: # accuracy treshhold
                    #print(f'accuracy: ',pred_class[0][i][2])
                    predictions_list = f"{predictions_list}, {pred_class[0][i][1]}"
            else: 
                predictions_list  = str(pred_class[0][i][1])
            i += 1


        #result = str(pred_class[0][0][1])               # Convert to string
        predictions_list = predictions_list.replace('_',' ')
        
        predictions_list = (predictions_list, '   -   ',  translate(predictions_list)) # result translation
        #print('predictions_list: ', predictions_list)

        encodedStr = convertTuple(predictions_list)
        return encodedStr
    return None


def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))


def convertTuple(tup): 
    str =  ''.join(tup) 
    return str

def translate(stringX):

    #translator    pip install googletrans
    translator = Translator()
    text = translator.translate(stringX, src='en', dest='ru').text
    #print(' translator......... ', text)
    return text;

if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
