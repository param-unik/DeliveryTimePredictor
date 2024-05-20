import os, sys, joblib
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging


def save_model_obj(file_path, model):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(model, file_obj)

    except Exception as e:
        raise CustomException("Error while saving model object", e)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):

            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            model_test_score = r2_score(y_test, y_pred)

            report[list(models.keys())[i]] = model_test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_model(file_path):
    try:
        with open(file_path, "rb") as f:
            return joblib.load(f)

    except Exception as e:
        raise CustomException(e, sys)
