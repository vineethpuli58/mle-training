import logging
import os
import tarfile

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


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


def stratified_split(data):
    """This function performs the stratified splitting
    of data based on newly created 'income_cat' column.

    Parameters
    ----------
    data : pd.DataFrame
         pandas DataFrame

    Returns
    -------
    data : pd.DataFrame
         pandas DataFrame
    """
    data_copy = data.copy()
    data_copy["income_cat"] = np.ceil(data_copy.median_income / 1.5)
    data_copy["income_cat"].where(data_copy.income_cat < 5, 5.0, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=21)
    for train_index, test_index in split.split(data_copy, data_copy.income_cat):
        strat_train_set = data_copy.loc[train_index]
        strat_test_set = data_copy.loc[test_index]

    # drop the income cat column after splitting
    strat_train_set.drop("income_cat", axis=1, inplace=True)
    strat_test_set.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def ingest_data(log_level, experiment_id, folder_name="housing"):
    """This function ingest housing data in local directory.

    Parameters
    ----------
    log_level : _type_
        set log-level
    experiment_id: int
        mlflow experiment id
    folder_name : str, optional
        set folder name, by default "housing"
    """

    # Create and configure logger

    # Creating an object
    with mlflow.start_run(
        run_name="DATA_PREPARATION",
        experiment_id=experiment_id,
        description="Data Preparation",
        nested=True,
    ) as data_prep:

        mlflow.log_param("child", "yes")

        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename="../logs/" + folder_name + "/log.log",
            format="%(asctime)s %(message)s",
            filemode="a",
        )
        print("y")

        # Setting the threshold of logger to DEBUG
        logger.setLevel(log_level)

        HOUSING_PATH = os.path.join("../data/raw", folder_name)
        logger.debug(HOUSING_PATH)
        DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
        HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

        fetch_housing_data(HOUSING_URL, HOUSING_PATH)
        housing = load_housing_data(HOUSING_PATH)

        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

        housing["income_cat"] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

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

        train_set, test_set = stratified_split(housing)

        housing = train_set.copy()
        housing.plot(kind="scatter", x="longitude", y="latitude")
        housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

        corr_matrix = housing.corr()
        logger.info(corr_matrix)
        corr_matrix["median_house_value"].sort_values(ascending=False)

        X_train = train_set.drop("median_house_value", axis=1)
        y_train = train_set["median_house_value"].copy()
        X_test = test_set.drop("median_house_value", axis=1)
        y_test = test_set["median_house_value"].copy()

        HOUSING_PATH = os.path.join("../data/processed/" + folder_name)
        os.makedirs(HOUSING_PATH, exist_ok=True)
        X_train.to_csv(HOUSING_PATH + "/train_x.csv", index=False)
        y_train.to_csv(HOUSING_PATH + "/train_y.csv", index=False)
        X_test.to_csv(HOUSING_PATH + "/test_x.csv", index=False)
        y_test.to_csv(HOUSING_PATH + "/test_y.csv", index=False)

        logger.debug(HOUSING_PATH)
        # --------------------------------------------------------
