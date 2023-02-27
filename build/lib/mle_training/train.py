import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attributes].values


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args, **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):  # y=None is added to have same style across API
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def data_pipeline(X, add_bedrooms_per_room=False):
    """This function return pipeline object containing all data preparation methods.

    Parameters
    ----------
    X : pd.DataFrame
        pandas DataFrame on which data preparation to be done
    add_bedrooms_per_room : bool
        Flag to add bedrooms_per_room column or not

    Returns
    ----------
    sklearn.pipeline.Pipeline object
    """

    num_columns = list(X.select_dtypes("number").columns)
    cat_columns = ["ocean_proximity"]
    num_pipeline = Pipeline(
        [
            ("selector", DataFrameSelector(num_columns)),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "attribs_addr",
                CombinedAttributesAdder(add_bedrooms_per_room=add_bedrooms_per_room),
            ),
            ("std_scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("selector", DataFrameSelector(cat_columns)),
            ("one_hot_encoder", OneHotEncoder(sparse=False)),
        ]
    )

    full_pipeline = FeatureUnion(
        transformer_list=[("num_pipeline", num_pipeline), ("cat_pipeline", cat_pipeline)]
    )

    return full_pipeline


def train(log_level, experiment_id, folder_name="housing"):
    """This function trains three Machine Learning Model (Linear regression,Decision Tree,Random Forest).

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
        run_name="MODEL_TRAINING",
        experiment_id=experiment_id,
        description="Model Training",
        nested=True,
    ) as model_training:

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
        logger.debug(artifacts_path)
        X_train = pd.read_csv(HOUSING_PATH + "/train_x.csv")
        y_train = pd.read_csv(HOUSING_PATH + "/train_y.csv")

        pipeline = data_pipeline(X_train)
        housing_prepared = pipeline.fit_transform(X_train)
        housing_labels = y_train.copy()

        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)

        # Save the model as a pickle in a file

        joblib.dump(lin_reg, str(artifacts_path) + "/lin_reg_model.pkl")
        mlflow.sklearn.log_model(lin_reg, "Linear Regression")

        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(housing_prepared, housing_labels)

        joblib.dump(tree_reg, str(artifacts_path) + "/tree_reg_model.pkl")
        mlflow.sklearn.log_model(tree_reg, "Decision Tree Regression")
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

        final_model = grid_search.best_estimator_
        joblib.dump(final_model, str(artifacts_path) + "/random_forest_final_model.pkl")
        best_params = grid_search.best_params_
        for param in best_params:
            mlflow.log_param(param, best_params[param])
        mlflow.sklearn.log_model(final_model, "Grid Search Random Forest model")
