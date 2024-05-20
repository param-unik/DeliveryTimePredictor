import os, sys
from datetime import datetime


def get_current_timestamp():
    return str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))


CURRENT_TIMESTAMP = get_current_timestamp()
ROOT_DIR = os.getcwd()
DATA_DIR = "Data"
DATA_FILE_NAME = "finalTrain.csv"
ARTIFACTS_DIR = "artifacts"

# data ingestion pipeline constants

DATA_INGESTION_ROOT_DIR = "data-ingestion"
DATA_INGESTION_RAWDATA_DIR = "raw-data"  # downloaded data
DATA_INGESTION_INGESTED_DATA_DIR = "ingested-data"
RAW_DATA_FILENAME = "raw_data.csv"
TRAIN_DATA_FILENAME = "train_data.csv"
TEST_DATA_FILENAME = "test_data.csv"


# data transformation pipeline constants

DATA_TRANSFORMATION_ROOT_DIR = "data-transformation"
DATA_PROCESSOR_DIR = "processor"
DATA_TRANSFORMED_DATA_DIR = "transformed-data"
DATA_PROCESSED_OBJ_FILE_MAME = "model.joblib"
DATA_TRANSFORMED_TRAIN_FILENAME = "train_transformed_data.csv"
DATA_TRANSFORMED_TEST_FILENAME = "test_transformed_data.csv"

# Model training constants
MODEL_TRAINER_ROOT_DIR = "model-trainier"
MODEL_OBJECT_FILE_MAME = "model.joblib"
