
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import sys
import urllib.request
import matplotlib.pyplot as plt
from scipy.misc import imresize
import json
import cv2

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def read_image_from_url(image_url, input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    with urllib.request.urlopen(image_url) as f:
        pic = plt.imread(f, format='jpg')
        # pic = imresize(pic, (input_width, input_height))
        pic = cv2.resize(pic, (input_width, input_height))
        pic = pic - input_mean
        pic = pic / input_std
    return [pic]

class image_classification():
    def __init__(self):
        model_file = "data/inception_v3_2016_08_28_frozen.pb"
        self.label_file = "data/imagenet_slim_labels.txt"
        self.input_height = 299
        self.input_width = 299
        self.input_mean = 0
        self.input_std = 255
        self.input_layer = "input"
        self.output_layer = "InceptionV3/Predictions/Reshape_1"
        self.graph = load_graph(model_file)

    def on_post(self, req, resp, name):
        try:
            dt = req.stream.read()
            req = json.loads(dt.decode('utf-8'))

            if name == 'image_classification':
                image_path = ''
                top_k = 5
                if req.__contains__('image_path'):
                    image_path = req['image_path']
                if req.__contains__('top_k'):
                    top_k = int(req['top_k'])
                if image_path != 0:
                    res = self.classify_image(image_path, top_k)
                else:
                    res = 'Input image path or image url to classify'
        except:
            res = 'There are something failed'
        resp.body = json.dumps(res)

    def classify_image(self, file_name, top_k=5):
        try:
            if file_name.find('http:') == 0 or file_name.find('https:') == 0:
                t = read_image_from_url(
                    file_name,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    input_mean=self.input_mean,
                    input_std=self.input_std)
            else:
                t = read_tensor_from_image_file(
                    file_name,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    input_mean=self.input_mean,
                    input_std=self.input_std)
        except:
            return 'Cannot load image from: ' + file_name

        input_name = "import/" + self.input_layer
        output_name = "import/" + self.output_layer
        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        with tf.Session(graph=self.graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        top_k = results.argsort()[-top_k:][::-1]
        labels = load_labels(self.label_file)
        res = {}
        for i in top_k:
            print(labels[i], results[i])
            res[labels[i]] = results[i]
        return res

if __name__ == '__main__':
    img_classification = image_classification()
    file_name = '/home/mvn/Desktop/tensorflow/tensorflow-1.7.0/tensorflow/examples/label_image/data/grace_hopper.jpg'
    img_classification.classify_image(file_name)
    file_name = 'https://img.grouponcdn.com/deal/hfefAup1zQWBE2K8sWURgS27xax/hf-846x508/v1/c700x420.jpg'
    img_classification.classify_image(file_name)
    # read_tensor_from_image_file('/home/mvn/Desktop/tensorflow/tensorflow-1.7.0/tensorflow/examples/label_image/data/grace_hopper.jpg')
    # read_image_from_url('https://img.grouponcdn.com/deal/hfefAup1zQWBE2K8sWURgS27xax/hf-846x508/v1/c700x420.jpg')
