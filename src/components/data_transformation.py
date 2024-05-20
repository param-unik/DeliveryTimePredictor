import os, sys
import pandas as pd
import numpy as np

# custom imports
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path

# project imports
from src.utils import save_model_obj
from src.exception import CustomException
from src.logger import logging
from src.config.configuration import *
from src.constants import *


@dataclass
class DataTrannsformationConfig:
    processed_obj_file_path: str = PROCESSED_OBJ_FILE_PATH
    transformed_train_file_path: str = TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_file_path: str = TRANSFORMED_TEST_FILE_PATH
    feature_eng_obj_file_path: str = FEATURE_ENG_OBJ_FILE_PATH


class FeatureEngineering(BaseEstimator, TransformerMixin):

    def __init__(self):
        logging.info(">>>> Feature Engineering Initialized and started..")

    def calculate_distance(self, data, lat1, long1, lat2, long2):
        p = np.pi / 180
        a = (
            0.5
            - np.cos((data[lat2] - data[lat1]) * p) / 2
            + np.cos(data[lat1] * p)
            * np.cos(data[lat2] * p)
            * (1 - np.cos((data[long2] - data[long1]) * p))
            / 2
        )

        data["distance"] = 12734 * np.arccos(np.sort(a))

    def transform_data(self, data):
        try:
            self.calculate_distance(
                data,
                "Restaurant_latitude",
                "Restaurant_longitude",
                "Delivery_location_latitude",
                "Delivery_location_longitude",
            )

            columns_to_drop = [
                "ID",
                "Delivery_person_ID",
                "Restaurant_latitude",
                "Restaurant_longitude",
                "Delivery_location_latitude",
                "Delivery_location_longitude",
                "Order_Date",
                "Time_Orderd",
                "Time_Order_picked",
            ]

            transformed_data = data.drop(columns_to_drop, axis=1)
            logging.info(
                f"<<<< dropping these columns -> {columns_to_drop} from the original dataset."
            )

            return transformed_data
        except Exception as e:
            raise CustomException(e, sys)

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        try:
            transformed_data = self.transform_data(X)
            return transformed_data
        except Exception as e:
            logging.exception("transform function got an exception!!")
            raise CustomException(e, sys)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTrannsformationConfig()

    def apply_data_transformation(self):
        try:
            road_traffic_categories = ["Low", "Medium", "High", "Jam"]
            weather_conditions = [
                "Sunny",
                "Cloudy",
                "Fog",
                "Sandstorms",
                "Windy",
                "Stormy",
            ]

            nominal_categorical_columns = [
                "Type_of_order",
                "Type_of_vehicle",
                "Festival",
                "City",
            ]
            ordinal_categorical_columns = ["Road_traffic_density", "Weather_conditions"]
            numerical_columns = [
                "Delivery_person_Age",
                "Delivery_person_Ratings",
                "Vehicle_condition",
                "multiple_deliveries",
                "distance",
            ]

            # Numerical Pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            # Nominal Categorical Pipeline
            nominal_categorical_pipeline = Pipeline(
                steps=[
                    ("imuter", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            # ordinal categorical pipeline
            ordinal_categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ordinal",
                        OrdinalEncoder(
                            categories=[road_traffic_categories, weather_conditions]
                        ),
                    ),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    (
                        "nominal_categorical_pipeline",
                        nominal_categorical_pipeline,
                        nominal_categorical_columns,
                    ),
                    (
                        "ordinal_categorical_pipeline",
                        ordinal_categorical_pipeline,
                        ordinal_categorical_columns,
                    ),
                ],
            )

            logging.info("Pipeline has been completed..")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def apply_feature_engineering(self):
        try:
            feature_engineering = Pipeline(
                steps=[("feature_eng", FeatureEngineering())]
            )

            return feature_engineering
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Feature Engineering has been initiated...")
            fe_obj = self.apply_feature_engineering()

            train_data = fe_obj.fit_transform(train_data)
            test_data = fe_obj.transform(test_data)

            logging.info("Feature engineered train data ")
            logging.info(f"{train_data.head()}")
            print("Feature eng train data is \n")
            print(train_data.head())

            logging.info("\n")
            logging.info("Feature engineered test data ")
            logging.info(f"{test_data.head()}")
            print("Feature eng test data is \n")
            print(test_data.head())
            logging.info("\n")

            train_data.to_csv("train_fe_data.csv")
            test_data.to_csv("test_fe_data.csv")

            processing_obj = self.apply_data_transformation()
            target_column = "Time_taken (min)"

            X_train = train_data.drop([target_column], axis=1)
            y_train = train_data[target_column]

            X_test = test_data.drop([target_column], axis=1)
            y_test = test_data[target_column]

            X_train = processing_obj.fit_transform(X_train)
            X_test = processing_obj.transform(X_test)

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            logging.info("Transformed train data ")
            logging.info(f"{df_train.head()}")
            logging.info("\n")

            print("Transformed train data is \n")
            print(df_train.head())

            logging.info("Transformed test data ")
            logging.info(f"{df_test.head()}")
            logging.info("\n")
            print("Transformed test data is \n")
            print(df_test.head())

            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.transformed_train_file_path
                ),
                exist_ok=True,
            )
            df_train.to_csv(
                self.data_transformation_config.transformed_train_file_path,
                index=False,
                header=True,
            )
            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.transformed_test_file_path
                ),
                exist_ok=True,
            )
            df_test.to_csv(
                self.data_transformation_config.transformed_test_file_path,
                index=False,
                header=True,
            )

            save_model_obj(
                file_path=self.data_transformation_config.processed_obj_file_path,
                model=processing_obj,
            )

            save_model_obj(
                file_path=self.data_transformation_config.feature_eng_obj_file_path,
                model=fe_obj,
            )

            logging.info("Data Transformation has been completed...")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.processed_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_transformation = DataTransformation()

    train_arr, test_arr, processed_obj_file_path = (
        data_transformation.initiate_data_transformation(
            Path(
                r"C:\Data-Science\ML Projects\DeliveryTimePredictor\artifacts\data-ingestion\2024-05-20 07-57-34\ingested-data\train_data.csv"
            ),
            Path(
                r"C:\Data-Science\ML Projects\DeliveryTimePredictor\artifacts\data-ingestion\2024-05-20 07-57-34\ingested-data\test_data.csv"
            ),
        )
    )
