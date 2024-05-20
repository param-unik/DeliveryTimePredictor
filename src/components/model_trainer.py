import os, sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin

from src.constants import *
from src.exception import CustomException
from src.logger import logging
from src.config.configuration import *
from src.utils import evaluate_model, save_model_obj

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = MODEL_OBJECT_FILE_PATH


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models = {
                "XGBRegressor": XGBRegressor(),
                "SVR": SVR(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
            }

            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models
            )

            print(model_report)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(
                f"Best model name is :  {best_model_name}, and it's R2 Score is : {best_model_score}"
            )

            logging.info("Model training completed successfully")
            logging.info(
                f"Best model name is :  {best_model_name}, and it's R2 Score is : {best_model_score}"
            )

            save_model_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                model=best_model,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    model_training = ModelTrainer()

    train_arr = pd.read_csv(
        r"C:\Data-Science\ML Projects\DeliveryTimePredictor\artifacts\data-transformation\transformed-data\train_transformed_data.csv"
    ).to_numpy()

    test_arr = pd.read_csv(
        r"C:\Data-Science\ML Projects\DeliveryTimePredictor\artifacts\data-transformation\transformed-data\test_transformed_data.csv"
    ).to_numpy()

    model_training.initiate_model_training(train_arr, test_arr)
