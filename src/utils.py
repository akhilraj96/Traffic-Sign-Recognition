import os
import sys
import pickle

from src.exception import CustomException
from src.logger import logging


def load_data(file):
    try:
        with open(file, mode='rb') as f:
            file_ = pickle.load(f)
        x_, y_ = file_['features'], file_['labels']
        logging.info(file+" Loaded")
        return x_, y_
    except Exception as e:
        raise CustomException(e, sys)


def save_data(data_X, data_y, file_path):
    try:
        data = {
            'features': data_X,
            'labels': data_y
        }

        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

        logging.info(f'data saved to {file_path}')
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise CustomException(e, sys)
