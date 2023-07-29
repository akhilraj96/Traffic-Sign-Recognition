import os
import sys
import numpy as np
import pandas as pd
import cv2
import random
import pickle

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import load_data

from src.components.data_ingestion import DataIngestion


class DataTransformation:
    def __init__(self):
        pass
    
    def augment_brightness_camera_images(self,image):
        image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        image1[:,:,2] = image1[:,:,2]*random_bright
        image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
        return image1

    def transform_image(self,img,ang_range,shear_range,trans_range,brightness=0):
        # Rotation
        ang_rot = np.random.uniform(ang_range)-ang_range/2
        rows,cols,ch = img.shape    
        Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

        # Translation
        tr_x = trans_range*np.random.uniform()-trans_range/2
        tr_y = trans_range*np.random.uniform()-trans_range/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

        # Shear
        pts1 = np.float32([[5,5],[20,5],[5,20]])
        pt1 = 5+shear_range*np.random.uniform()-shear_range/2
        pt2 = 20+shear_range*np.random.uniform()-shear_range/2

        # Brightness
        pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
        shear_M = cv2.getAffineTransform(pts1,pts2)
        img = cv2.warpAffine(img,Rot_M,(cols,rows))
        img = cv2.warpAffine(img,Trans_M,(cols,rows))
        img = cv2.warpAffine(img,shear_M,(cols,rows))

        if brightness == 1:
            img = self.augment_brightness_camera_images(img)

        return img

    def cache_data(self,X,Y, file):
        n_samples=X.shape[0]
        try:
                with open(file, 'wb') as pfile:
                    pickle.dump(
                        {
                            'features': X.astype(np.float32),
                            'labels': Y
                        },
                        pfile, pickle.HIGHEST_PROTOCOL)
                    logging.info("Data Saved in :" + file)

        except Exception as e:
            logging.info('Unable to save data to a single file so splitting data into 3'+ file+ ':'+ e)
            with open(file+'_1', 'wb') as pfile:
                pickle.dump(
                    {
                        'features': X[0:int((1/3)*n_samples)].astype(np.float32),
                        'labels': Y[0:int((1/3)*n_samples)]
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)

            with open(file+'_2', 'wb') as pfile:
                pickle.dump(
                    {
                        'features': X[int(n_samples/3):int((2/3)*n_samples)].astype(np.float32),
                        'labels': Y[int(n_samples/3):int((2/3)*n_samples)]
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)

            with open(file+'_3', 'wb') as pfile:
                pickle.dump(
                    {
                        'features': X[int((2/3)*n_samples):n_samples-1].astype(np.float32),
                        'labels': Y[int((2/3)*n_samples):n_samples-1]
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
            
            logging.info("Data Saved in :"+file)
            logging.info('pickle file saved as 3 parts for data')
    
    def initiate_data_transformation(self,train_path,test_path,val_path):
        logging.info("Entered the pre processing method")

        try:
            X_train, y_train = load_data(train_path)
            X_valid, y_valid = load_data(val_path)
            X_test, y_test = load_data(test_path)

            _classes, counts = np.unique(y_train, return_counts=True)

            for _class, count in zip(_classes, counts):
                new_images = []
                new_classes = []
                
                if count < 1000:
                    y_train_length = y_train.shape[0]
                    index = 0
                    
                    for i in range(0, 1000-count):
                        while y_train[index] != _class:
                            index = random.randint(0, y_train_length-1)
                        new_images.append(self.transform_image(X_train[index],10,5,5,brightness=1))
                        new_classes.append(_class)
                    X_train = np.concatenate((X_train, np.array(new_images)))
                    y_train = np.concatenate((y_train, np.array(new_classes)))

            logging.info("Number of training examples = "+ str(X_train.shape[0]))

            X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
            X_test_gray = np.sum(X_test/3, axis=3, keepdims=True)
            X_valid_gray = np.sum(X_valid/3, axis=3, keepdims=True)


            X_train_gray -= np.mean(X_train_gray)
            X_test_gray -= np.mean(X_test_gray)
            X_train = X_train_gray
            X_test = X_test_gray

            X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=0)
            X_train, y_train = shuffle(X_train, y_train)

            self.cache_data(X_train, y_train, "artifacts/train.p")
            self.cache_data(X_validation, y_validation, "artifacts/valid.p")
            self.cache_data(X_test, y_test, "artifacts/test.p")

            return X_train.shape[0]


        except Exception as e:
            raise CustomException(e, sys)

    

if __name__=="__main__":
    data1 = DataIngestion()
    p1,p2,p3=data1.initiate_data_ingestion()
    data2 = DataTransformation()
    num_train_examples = data2.initiate_data_transformation(p1,p2,p3)
