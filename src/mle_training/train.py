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


def train(log_level, folder_name="housing"):
    """This function trains three Machine Learning Model (Linear regression,Decision Tree,Random Forest).

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
    logger.debug(artifacts_path)
    housing_prepared = pd.read_csv(HOUSING_PATH + "/train_x.csv")
    housing_labels = pd.read_csv(HOUSING_PATH + "/train_y.csv")

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    # Save the model as a pickle in a file

    joblib.dump(lin_reg, str(artifacts_path) + "/lin_reg_model.pkl")

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    joblib.dump(tree_reg, str(artifacts_path) + "/tree_reg_model.pkl")

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    grid_search.best_params_
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    final_model = grid_search.best_estimator_
    joblib.dump(
        final_model, str(artifacts_path) + "/random_forest_final_model.pkl"
    )
