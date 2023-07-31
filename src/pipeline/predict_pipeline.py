import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class PredictPipeline:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        model = tf.keras.models.load_model(self.model_path)
        return model

    def preprocess_image(self, image_path):
        image = load_img(image_path, target_size=(32, 32))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        return preprocess_input(image_array)

    def predict(self, image_path):
        preprocessed_image = self.preprocess_image(image_path)
        prediction = self.model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)
        return predicted_class

# Example usage:
if __name__ == "__main__":
    model_path = 'artifacts/model'
    image_path = "sign1.jpg"

    pipeline = PredictPipeline(model_path)
    predicted_class = pipeline.predict(image_path)

    print("Predicted class index:", predicted_class)