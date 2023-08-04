import os
import sys
import urllib.request
import zipfile
import pickle

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('data', 'train.p')
    test_data_path: str = os.path.join('data', 'test.p')
    validation_data_path: str = os.path.join('data', 'valid.p')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            logging.info("Beginning Data download...")

            source_url = 'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip'
            urllib.request.urlretrieve(
                source_url, './data/traffic-signs-data.zip')

            logging.info('Beginning file unzip')

            ref = zipfile.ZipFile('./data/traffic-signs-data.zip', 'r')
            ref.extractall('./data/')
            ref.close()

            logging.info('Download and Unzip completed')
            logging.info("<><><><><><><><><><><><><><><><><><>")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.validation_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pass
