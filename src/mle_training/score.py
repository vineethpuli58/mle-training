import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mle_training.train import data_pipeline


def score(log_level, experiment_id, folder_name="housing"):
    """This function does the evaluation of the models.The final metrics used to evaluate is mean squared error,root mean squared error,mean absolute error.

    Parameters
    ----------
    log_level : _type_
        set log-level
    experiment_id: int
        mlflow experiment id
    folder_name : str, optional
        set folder name, by default "housing"
    """
    with mlflow.start_run(
        run_name="MODEL_SCORING",
        experiment_id=experiment_id,
        description="Model Scoring",
        nested=True,
    ) as model_scoring:

        mlflow.log_param("child", "yes")
        logger = logging.getLogger(__name__)

        logging.basicConfig(
            filename="../logs/" + folder_name + "/log.log",
            format="%(asctime)s %(message)s",
            filemode="a",
        )

        # Setting the threshold of logger to DEBUG
        logger.setLevel(log_level)

        HOUSING_PATH = os.path.join("../data/processed", folder_name)
        artifacts_path = os.path.join("../artifacts", folder_name)

        train_x = pd.read_csv(HOUSING_PATH + "/train_x.csv")
        test_x = pd.read_csv(HOUSING_PATH + "/test_x.csv")
        test_y = pd.read_csv(HOUSING_PATH + "/test_y.csv")

        pipeline = data_pipeline(train_x)
        pipeline.fit(train_x)
        test_x = pipeline.transform(test_x)

        # Load the model from the file
        lin_reg_model = joblib.load(artifacts_path + "/lin_reg_model.pkl")

        # Use the loaded model to make predictions
        housing_predictions = lin_reg_model.predict(test_x)

        lin_mse = mean_squared_error(test_y, housing_predictions)
        lin_rmse = np.sqrt(lin_mse)
        lin_mae = mean_absolute_error(test_y, housing_predictions)
        mlflow.log_metric("lin_mse", lin_mse)
        mlflow.log_metric("lin_rmse", lin_rmse)
        mlflow.log_metric("lin_mae", lin_mae)

        tree_reg_model = joblib.load(artifacts_path + "/tree_reg_model.pkl")
        housing_predictions = tree_reg_model.predict(test_x)

        tree_mse = mean_squared_error(test_y, housing_predictions)
        tree_rmse = np.sqrt(tree_mse)
        tree_mae = mean_absolute_error(test_y, housing_predictions)
        mlflow.log_metric("tree_mse", tree_mse)
        mlflow.log_metric("tree_rmse", tree_rmse)
        mlflow.log_metric("tree_mae", tree_mae)

        random_forest_final_model = joblib.load(artifacts_path + "/random_forest_final_model.pkl")
        final_predictions = random_forest_final_model.predict(test_x)

        final_mse = mean_squared_error(test_y, final_predictions)
        final_rmse = np.sqrt(final_mse)
        final_mae = mean_absolute_error(test_y, final_predictions)
        mlflow.log_metric("final_mse", final_mse)
        mlflow.log_metric("final_rmse", final_rmse)
        mlflow.log_metric("final_mae", final_mae)
