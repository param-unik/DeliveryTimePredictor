import os, sys
from flask import Flask, render_template, request
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
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from Prediction.batch_prediction import BatchPrediction
from werkzeug.utils import secure_filename

feature_eng_file_path = FEATURE_ENG_OBJ_FILE_PATH
treansformer_file_path = PROCESSED_OBJ_FILE_PATH
model_file_path = MODEL_OBJECT_FILE_PATH

UPLOAD_DIR_PATH = "batch-prediction/uploaded_File"

app = Flask(__name__, template_folder="templates")

ALLOWED_EXTENSIONS = {"csv"}


# localhost:5000/
@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else:
        data = CustomData(
            Delivery_person_Age=int(request.form.get("Delivery_person_Age")),
            Delivery_person_Ratings=int(request.form.get("Delivery_person_Ratings")),
            Weather_conditions=request.form.get("Weather_conditions"),
            Road_traffic_density=request.form.get("Road_traffic_density"),
            Vehicle_condition=int(request.form.get("Vehicle_condition")),
            multiple_deliveries=int(request.form.get("multiple_deliveries")),
            distance=float(request.form.get("distance")),
            Type_of_order=request.form.get("Type_of_order"),
            Type_of_vehicle=request.form.get("Type_of_vehicle"),
            Festival=request.form.get("Festival"),
            City=request.form.get("City"),
        )

        final_new_data = data.get_data_as_dataframe()
        print("final new dataframe is \n")
        print(final_new_data.head())
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)
        result = int(pred[0])

        return render_template("form.html", final_result=result)


@app.route("/batch", methods=["GET", "POST"])
def batch_prediction():
    if request.method == "GET":
        return render_template("batch.html")
    else:
        file = request.files["csv_file"]
        directory_path = UPLOAD_DIR_PATH
        os.makedirs(directory_path, exist_ok=True)
        print(file.filename)
        print(file.filename.rsplit(".", 1)[1].lower())

        if (
            file
            and "." in file.filename
            and file.filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
        ):

            for filename in os.listdir(os.path.join(UPLOAD_DIR_PATH)):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_DIR_PATH, filename)
            file.save(file_path)
            print("File saved to: " + file_path)

            logging.info("Batch prediction started...")

            batch = BatchPrediction(
                file_path,
                MODEL_OBJECT_FILE_PATH,
                PROCESSED_OBJ_FILE_PATH,
                FEATURE_ENG_OBJ_FILE_PATH,
            )

            batch.start_batch_prediction()

            output = "Batch prediction completed successfully.."
            return render_template(
                "batch.html", prediction_result=output, prediction_type="batch"
            )
        else:
            return render_template(
                "batch.html", prediction_type="batch", error="Invalid File Type"
            )


@app.route("/train", methods=["GET", "POST"])
def training():
    if request.method == "GET":
        return render_template("train.html")
    else:
        try:
            pipeline = TrainingPipeline()
            pipeline.run()

            return render_template(
                "train.html", message="Successfully trained the model."
            )
        except Exception as e:
            logging.error("Error occurred while training the model.")
            error_mssage = str(e)
            return render_template("index.html", error=error_mssage)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8888", debug=True)
