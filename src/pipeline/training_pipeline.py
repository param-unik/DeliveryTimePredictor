import os, sys
from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import *
from src.components.data_transformation import (
    DataTrannsformationConfig,
    DataTransformation,
)
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer
from src.components.data_ingestion import DataIngestionConfig, DataIngestion


class TrainingPipeline:

    def __init__(self):
        self.c = 0
        print(f">>>>>> {self.c} <<<<<<<<<")

    def run(self):
        logging.info("Data Ingestion Pipeline: Initiated")
        data_ingestion = DataIngestion()
        logging.info("Data Ingestion Pipeline: completed..")

        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data Transformation Pipeline: Initiated")
        data_transformation = DataTransformation()
        train_arr, test_arr, processor_obj_file_path = (
            data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
        )
        logging.info("Data Transformation Pipeline: completed..")

        logging.info("Model Training Pipeline: Initiated")
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)
        logging.info("Model Training Pipeline: completed..")
