import os, sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.constants import *
from src.config.configuration import *
from src.utils import load_model

PREDICTION_DIR_PATH = "batch-prediction"
PREDICTION_FILE_PATH = "model-predictions"
PREDICTION_FILE_NAME = "prediction.csv"
FEATURE_ENG_DIR_PATH = "feature-eng"

ROOT_DIR = os.getcwd()

BATCH_PREDICTION_FILE_PATH = os.path.join(
    ROOT_DIR, PREDICTION_DIR_PATH, PREDICTION_FILE_PATH
)

FEATURE_ENG_FILE_PATH = os.path.join(ROOT_DIR, FEATURE_ENG_DIR_PATH)


class BatchPrediction:
    def __init__(
        self,
        input_file_path,
        model_file_path,
        transformed_file_path,
        feature_eng_file_path,
    ) -> None:
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformed_file_path = transformed_file_path
        self.feature_eng_file_path = feature_eng_file_path

    def start_batch_prediction(self):
        try:

            feature_pipeline = load_model(self.feature_eng_file_path)
            transformed_pipeline = load_model(self.transformed_file_path)
            model = load_model(self.model_file_path)

            feature_eng_pipeline = Pipeline(
                steps=[
                    ("feature-engineering", feature_pipeline),
                ]
            )

            df = pd.read_csv(self.input_file_path)
            df.to_csv("delivery_time_prediction_data.csv")

            # Applying feature engineering on the dataframe
            df = feature_eng_pipeline.transform(df)

            os.makedirs(FEATURE_ENG_FILE_PATH, exist_ok=True)
            file_path = os.path.join(FEATURE_ENG_FILE_PATH, "batch_feature_eng.csv")
            df.to_csv(file_path, index=False)

            df = df.drop("Time_taken (min)", axis=1)

            # Applying transformation on the dataframe
            transformed_data = transformed_pipeline.transform(df)
            logging.info(f"Transformed Data Shape: {transformed_data.shape}")

            logging.info(f"Loaded numpy from batch prediciton :{transformed_data}")

            file_path = os.path.join(FEATURE_ENG_FILE_PATH, "batch_transformed.csv")

            # Applying best model to predict the output
            predictions = model.predict(transformed_data)

            df_prediction = pd.DataFrame(predictions, columns=["prediction"])

            print("Prediction is as follows \n")
            print(df_prediction.head())
            print(df_prediction.shape)

            os.makedirs(BATCH_PREDICTION_FILE_PATH, exist_ok=True)
            file_path = os.path.join(BATCH_PREDICTION_FILE_PATH, "batch_prediction.csv")
            df_prediction.to_csv(file_path, index=False)

            logging.info("Batch prediction completed successfully!")

        except Exception as e:
            raise CustomException(e, sys)
