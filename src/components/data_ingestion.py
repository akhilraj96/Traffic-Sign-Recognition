import os
import urllib.request
import zipfile
import pickle

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

print('Beginning Data download...')
logging.info("Beginning Data download...")

source_url = 'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip'
# urllib.request.urlretrieve(source_url, './data/traffic-signs-data.zip')

logging.info('Beginning file unzip')

ref = zipfile.ZipFile('./data/traffic-signs-data.zip', 'r')
ref.extractall('./data/')
ref.close()

logging.info('Download and Unzip completed')
print('Done')

training_file = './data/train.p'
validation_file= './data/valid.p'
testing_file = './data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


n_train = X_train.shape[0]
n_validation = X_valid.shape[0]
n_test = X_test.shape[0]
image_shape = X_train.shape[1:]
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)