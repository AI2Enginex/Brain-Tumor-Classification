import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image


class Load_var:

    def __init__(self, img):

        self.img_mat = cv2.imread(img)
        self.model = load_model("D:\CNN_Projects\Brain_Tumor\model_tumor_.h5")
        self.display_str = ["Negative - Type None","Positive - Type Glioma","Positive - Type Meningioma","Positive - Type Pituitary"]


class Predict_image(Load_var):

    def __init__(self, img, size):

        super().__init__(img)
        self.image = self.img_mat
        self.cnn_model = self.model
        self.mat_size = size

    def predict_output(self):

        self.image_ = cv2.resize(self.image, (self.mat_size, self.mat_size))
        self.image_ = self.image_ / 255.0
        img_res = np.expand_dims(self.image_, axis=0)
        usr_prd = np.argmax(self.cnn_model.predict(img_res))
        return usr_prd


class Diplay_result(Load_var):

    def __init__(self, img, image_size, display_img_size):

        super().__init__(img)

        self.dis_img = display_img_size
        pr = Predict_image(img, image_size)
        self.outputs = self.display_str
        self.user_image = self.img_mat
        self.result = pr.predict_output()
    

    def display(self):

        self.user_image = cv2.resize(
            self.user_image, (self.dis_img, self.dis_img))
        
        cv2.putText(self.user_image, self.outputs[self.result], (80, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 6)
        cv2.imshow("cancer", self.user_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":

    dis = Diplay_result("D:\CNN_Projects\Brain_Tumor\default.jpg",
                        image_size=48, display_img_size=800)
    dis.display()
