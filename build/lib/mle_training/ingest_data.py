import argparse
import logging
import os
import tarfile

import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedShuffleSplit, train_test_split)
from sklearn.tree import DecisionTreeRegressor


def fetch_housing_data(housing_url, housing_path):

    """This function fetch housing data.

    Parameters
    ----------
    housing_url : _type_
        housing data url
    housing_path : os.path
        directory path where housing data will be stored.
    """

    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    """This function loads housing data.

    Parameters
    ----------
    housing_path : os.path
        directory path where housing data will be stored.

    Returns
    -------
    pd.DataFrame

        returns pandas Dataframe
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    """This function estimates proportion of income categories.

    Parameters
    ----------
    data : pd.DataFrame
         pandas DataFrame

    Returns
    -------
    float
        _description_
    """
    return data["income_cat"].value_counts() / len(data)


def ingest_data(log_level, folder_name="housing"):
    """This function ingest housing data in local directory.

    Parameters
    ----------
    log_level : _type_
        set log-level
    folder_name : str, optional
        set folder name, by default "housing"
    """

    # Create and configure logger

    # Creating an object

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename="../logs/" + folder_name + "/log.log",
        format="%(asctime)s %(message)s",
        filemode="a",
    )

    # Setting the threshold of logger to DEBUG
    logger.setLevel(log_level)

    HOUSING_PATH = os.path.join("../data/raw", folder_name)
    logger.debug(HOUSING_PATH)
    DOWNLOAD_ROOT = (
        "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    )
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)

    housing = load_housing_data(HOUSING_PATH)

    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    logger.info(corr_matrix)
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

    HOUSING_PATH = os.path.join("../data/processed/" + folder_name)
    os.makedirs(HOUSING_PATH, exist_ok=True)
    housing_prepared.to_csv(HOUSING_PATH + "/train_x.csv")
    housing_labels.to_csv(HOUSING_PATH + "/train_y.csv")

    logger.debug(HOUSING_PATH)
    # ---------------------------------------------------------

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(
        pd.get_dummies(X_test_cat, drop_first=True)
    )

    HOUSING_PATH = os.path.join("../data/processed/" + folder_name)
    os.makedirs(HOUSING_PATH, exist_ok=True)
    logger.debug(HOUSING_PATH)
    X_test_prepared.to_csv(HOUSING_PATH + "/test_x.csv")
    y_test.to_csv(HOUSING_PATH + "/test_y.csv")
