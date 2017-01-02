##### Visual Search Demo - Image similarity with Inception.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import math
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
import json

app = Flask(__name__)
working_dir = os.getcwd() 

# Returns a dictionary of feature representations for an input list of JPEG images, 
# which is the next-to-last layer of the Inception network (trained on the 
# ImageNet 2012 Challenge dataset). Tensorflow's Inception reference: 
# https://tensorflow.org/tutorials/image_recognition/
def get_features():
  img_features = {}

  # Creates graph from saved graph_def.pb.
  graph_file = working_dir + '/inception/classify_image_graph_def.pb'
  with tf.gfile.FastGFile(graph_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

  with tf.Session() as sess:
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #  float description of the image.
    last_layer = sess.graph.get_tensor_by_name('pool_3:0')

    for file in os.listdir(working_dir + '/static/img'):
      if file.endswith(".jpg"):
        img_path = working_dir + '/static/img/' + file
        image_data = tf.gfile.FastGFile(img_path, 'rb').read()
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        features = sess.run(last_layer, {'DecodeJpeg/contents:0': image_data})
        img_features[file] = list([float(x) for x in features[0][0][0]])
  return img_features

# Serve the homepage, a static file
@app.route("/")
def index():
  return render_template('index.html')

@app.route('/get_img_features')
def get_img_features():
  img_features = get_features()
  print(type(img_features))
  print("server calculated features")
  return jsonify(img_features)

# Serve static files
@app.route('/<path:path>')
def static_proxy(path):
  # send_static_file will guess the correct MIME type
  return app.send_static_file(path)

# Run the server app
if __name__ == "__main__":
  app.debug = True
  app.run()
  app.run(debug = True)