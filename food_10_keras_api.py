import urllib.request
import json
import falcon
from keras.models import load_model
import numpy as np
import cv2


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
        self.input_height = 299
        self.input_width = 299
        self.input_mean = 0
        self.input_std = 255
        # food keras model
        self.food101_model = load_model('data/model4.20-0.20.hdf5')
        self.food101_label = []
        with open('data/food101_10classes.txt') as f:
            lines = f.readlines()
            for line in lines:
                self.food101_label.append(line.strip())

    def on_post(self, req, resp, name):
        res = 'Do nothing'
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
                    res = self.keras_food_predict(image_path, top_k=top_k)
                else:
                    res = 'Input image path or image url to classify'
        except:
            res = 'There are something failed'
        resp.body = json.dumps(res)

    def keras_food_predict(self, file_name, top_k=5):
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
            return 'Cannot load image from: ' + file_name

        y_pred = self.food101_model.predict(np.array(t))[0]
        top_k = y_pred.argsort()[-top_k:][::-1]
        res = {}
        for i in top_k:
            print(self.food101_label[i], y_pred[i])
            res[self.food101_label[i]] = str(y_pred[i])

        return res


app = falcon.API()
app.add_route("/{name}", image_classification())

# img_classification = image_classification()
# file_name = '/home/mvn/Desktop/image_classification/data/Dog_CTA_Desktop_HeroImage.jpg'
# img_classification.classify_image(file_name)
# file_name = 'https://www.healthypawspetinsurance.com/Images/V3/DogAndPuppyInsurance/Dog_CTA_Desktop_HeroImage.jpg'
# img_classification.classify_image(file_name)
# file_name = '/home/mvn/Desktop/image_classification/data/sushi1.jpg'
# img_classification.keras_food_predict(file_name)
print('ready!!!!!')
