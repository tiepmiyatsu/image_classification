
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import sys
import urllib.request
import json
import falcon
from keras.models import Sequential, Model, load_model
import numpy as np
import cv2
import time

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
  # cv2.imshow(file_name, result[0])
  return result

def read_image_from_url(image_url, input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    with urllib.request.urlopen(image_url) as f:
        pic = np.asarray(bytearray(f.read()), dtype="uint8")
        pic = cv2.imdecode(pic, cv2.IMREAD_COLOR)
        pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
        pic = cv2.resize(pic, (input_width, input_height))
        pic = pic - input_mean
        pic = pic / input_std
    # cv2.imshow(file_name, pic)
    # cv2.waitKey()
    return [pic]


def read_image_from_path(image_path, input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    pic = cv2.imread(image_path)
    pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
    pic = cv2.resize(pic, (input_width, input_height))
    pic = pic - input_mean
    pic = pic / input_std
    return [pic]


class image_classification():
    def __init__(self):
        # imagenet
        print('load image net model...')
        model_file = "data/inception_v3_2016_08_28_frozen.pb"
        self.label_file = "data/imagenet_slim_labels.txt"
        self.input_height = 299
        self.input_width = 299
        self.input_mean = 0
        self.input_std = 255
        self.input_layer = "input"
        self.output_layer = "InceptionV3/Predictions/Reshape_1"
        self.graph = load_graph(model_file)
        self.imagenet_labels = load_labels(self.label_file)

        # food keras model
        print('Load food keras model...')
        self.food101_model = load_model('data/model4.20-0.20.hdf5')
        self.food101_label = []
        with open('data/food101_10classes.txt') as f:
            lines = f.readlines()
            for line in lines:
                self.food101_label.append(line.strip())

    def on_post(self, req, resp, name):
        res = {'res': 'Do nothing'}
        start_t = time.time()
        try:
            dt = req.stream.read()
            req = json.loads(dt.decode('utf-8'))
            if name == 'image_classification':
                image_path = ''
                top_k = 5
                dataset = 'image_net'
                if req.__contains__('image_path'):
                    image_path = req['image_path']
                if req.__contains__('top_k'):
                    top_k = int(req['top_k'])
                if req.__contains__('dataset'):
                    dataset = req['dataset']
                if image_path != 0:
                    if dataset == 'image_net' or dataset == "imagenet":
                        pred = self.classify_image(image_path, top_k=top_k)
                    elif dataset == 'food':
                        pred = self.keras_food_predict(image_path, top_k=top_k)
                    else:
                        pred = 'Please define dataset in payload: [image or food]'
                    res['res'] = pred
                else:
                    res = {'res': 'Input image path or image url to classify'}
        except Exception as ex:
            err_ms = "Unexpected error: " + str(ex)
            res = {'res': err_ms}
        end_t = time.time()
        res['time'] = '{0:0.2f} ms'.format((end_t-start_t) * 1000)

        res = self.make_payload(res)
        resp.body = json.dumps(res)

    def classify_image(self, file_name, top_k=5):
        res = {}
        print('Classify image at:', file_name)
        start_t = time.time()
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
            end_t = time.time()
            res['read_image_time'] = '{0:0.2f} ms'.format((end_t - start_t) * 1000)
            res['result'] = 'Cannot load image from: ' + file_name
            return res
        end_t = time.time()
        res['read_image_time'] = '{0:0.2f} ms'.format((end_t - start_t) * 1000)

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
        result = {}
        for i in top_k:
            print(self.imagenet_labels[i], results[i])
            result[self.imagenet_labels[i]] = str(results[i])
        res['result'] = result
        return res

    def keras_food_predict(self, file_name, top_k=5):
        res = {}
        start_t = time.time()
        print('Classify food at:', file_name)
        try:
            if file_name.find('http:') == 0 or file_name.find('https:') == 0:
                t = read_image_from_url(
                    file_name,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    input_mean=self.input_mean,
                    input_std=self.input_std)
            else:
                t = read_image_from_path(
                    file_name,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    input_mean=self.input_mean,
                    input_std=self.input_std)
        except:
            end_t = time.time()
            res['read_image_time'] = '{0:0.2f} ms'.format((end_t - start_t) * 1000)
            res['result'] = 'Cannot load image from: ' + file_name
            return res
        end_t = time.time()
        res['read_image_time'] = '{0:0.2f} ms'.format((end_t - start_t) * 1000)

        y_pred = self.food101_model.predict(np.array(t))[0]
        top_k = y_pred.argsort()[-top_k:][::-1]
        result = {}
        for i in top_k:
            print(self.food101_label[i], y_pred[i])
            result[self.food101_label[i]] = str(y_pred[i])
        res['result'] = result
        return res

    def make_payload(self, res):
        text = str(res)
        j = {"message": [{"res": text}]}
        return j

app = falcon.API()
app.add_route("/{name}", image_classification())

# img_classification = image_classification()
# file_name = '/home/mvn/Desktop/image_classification/data/Dog_CTA_Desktop_HeroImage.jpg'
# img_classification.classify_image(file_name)
# file_name = 'https://www.healthypawspetinsurance.com/Images/V3/DogAndPuppyInsurance/Dog_CTA_Desktop_HeroImage.jpg'
# img_classification.classify_image(file_name)
# file_name = '/home/mvn/Desktop/image_classification/data/sushi1.jpg'
# img_classification.keras_food_predict(file_name)
print('ready!!!!')
