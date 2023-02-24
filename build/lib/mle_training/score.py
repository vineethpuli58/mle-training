import argparse
import logging
import os
import tarfile

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import randint
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedShuffleSplit, train_test_split)
from sklearn.tree import DecisionTreeRegressor


def score(log_level, folder_name="housing"):
    """This function does the evaluation of the models.The final metrics used to evaluate is mean squared error,root mean squared error,mean absolute error.

    Parameters
    ----------
    log_level : _type_
        set log-level
    folder_name : str, optional
        set folder name, by default "housing"
    """

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        filename="../logs/" + folder_name + "/log.log",
        format="%(asctime)s %(message)s",
        filemode="a",
    )

    # Setting the threshold of logger to DEBUG
    logger.setLevel(log_level)

    HOUSING_PATH = os.path.join("../data/processed", folder_name + "/")
    artifacts_path = os.path.join("../artifacts", folder_name + "/")

    test_x = pd.read_csv(HOUSING_PATH + "test_x.csv")
    test_y = pd.read_csv(HOUSING_PATH + "test_y.csv")

    # Load the model from the file
    lin_reg_model = joblib.load(artifacts_path + "/lin_reg_model.pkl")

    # Use the loaded model to make predictions
    housing_predictions = lin_reg_model.predict(test_x)

    lin_mse = mean_squared_error(test_y, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(test_y, housing_predictions)

    logger.info("lin_mse - " + str(lin_mse))
    logger.info("lin_rmse - " + str(lin_rmse))
    logger.info("lin_mae - " + str(lin_mae))

    tree_reg_model = joblib.load(artifacts_path + "/tree_reg_model.pkl")
    housing_predictions = tree_reg_model.predict(test_x)

    tree_mse = mean_squared_error(test_y, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_mae = mean_absolute_error(test_y, housing_predictions)

    logger.info("tree_mse - " + str(tree_mse))
    logger.info("tree_rmse - " + str(tree_rmse))
    logger.info("tree_mae - " + str(tree_mae))

    random_forest_final_model = joblib.load(
        artifacts_path + "/random_forest_final_model.pkl"
    )
    final_predictions = random_forest_final_model.predict(test_x)

    final_mse = mean_squared_error(test_y, final_predictions)
    final_rmse = np.sqrt(final_mse)
    final_mae = mean_absolute_error(test_y, final_predictions)

    logger.info("random_forest_final_mse - " + str(final_mse))
    logger.info("random_forest__final_rmse - " + str(final_rmse))
    logger.info("random_forest__final_mae - " + str(final_mae))
