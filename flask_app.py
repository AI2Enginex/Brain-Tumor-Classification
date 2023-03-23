import os
import cv2
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from flask import Flask, request, render_template


class Load_var:

    def __init__(self, size):

        self.model = load_model("./model_tumor.h5")
        self.labels = ["Negative - Type Healthy", "Positive - Type Glioma",
                       "Positive - Type Meningioma", "Positive - Type Pituitary"]
        self.mat_size = size

    def predict_result(self, img_path):

        try:
            self.image = cv2.imread(img_path)
            self.image_ = cv2.resize(
                self.image, (self.mat_size, self.mat_size))
            self.image_ = self.image_ / 255.0
            img_res = np.expand_dims(self.image_, axis=0)
            usr_prd = self.labels[np.argmax(self.model.predict(img_res))]
            return usr_prd
        except:
            return "error"


class Flask_app:

    def __init__(self, size):

        self.lv = Load_var(size)
        self.app = Flask(__name__, template_folder='templates')

    def index(self):
        return render_template('index.html')

    def upload(self):
        if request.method == 'POST':
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            preds = self.lv.predict_result(file_path)
            return preds
        return None

    def run_application(self):

        self.app.add_url_rule(
            '/', methods=['GET', 'POST'], view_func=self.index)
        self.app.add_url_rule(
            '/predict', methods=['GET', 'POST'], view_func=self.upload)
        self.app.run(debug=True)


if __name__ == '__main__':

    fa = Flask_app(size=48)
    fa.run_application()
