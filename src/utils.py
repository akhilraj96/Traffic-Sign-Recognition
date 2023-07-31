import os 
import sys
import pickle

from src.exception import CustomException
from src.logger import logging

def load_data(file):
    with open(file, mode='rb') as f:
        file_ = pickle.load(f)  
    x_, y_ = file_['features'], file_['labels']
    logging.info(file+" Loaded")
    return x_,y_

def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise CustomException(e, sys)