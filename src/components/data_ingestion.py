import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
from src.constants import *
from src.config.configuration import *
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    train_data_path: str = TRAIN_FILE_PATH
    test_data_path: str = TEST_FILE_PATH
    raw_data_path: str = RAW_FILE_PATH


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # read data
            data = pd.read_csv(DATA_FILE_PATH)

            # create raw-data folder if not exist
            os.makedirs(
                os.path.split(Path(self.data_ingestion_config.raw_data_path))[0],
                exist_ok=True,
            )

            # save the data to raw-data folder
            data.to_csv(Path(self.data_ingestion_config.raw_data_path), index=False)

            # split the data into train and test sets
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            os.makedirs(
                os.path.split(Path(self.data_ingestion_config.train_data_path))[0],
                exist_ok=True,
            )
            train_set.to_csv(
                Path(self.data_ingestion_config.train_data_path), header=True
            )

            os.makedirs(
                os.path.split(Path(self.data_ingestion_config.test_data_path))[0],
                exist_ok=True,
            )
            test_set.to_csv(
                Path(self.data_ingestion_config.test_data_path), header=True
            )

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
