import pickle

from src.logger import logging

def load_data(file):
    with open(file, mode='rb') as f:
        file_ = pickle.load(f)  
    x_, y_ = file_['features'], file_['labels']
    logging.info(file+" Loaded")
    return x_,y_