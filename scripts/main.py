import argparse
import logging
import os
import pathlib

import mlflow

from mle_training import ingest_data, score, train

if not os.path.isdir("../mlruns"):
    os.makedirs("../mlruns")
mlflow.set_tracking_uri(pathlib.Path("../mlruns/"))

if __name__ == "__main__":
    experiment_id = mlflow.create_experiment("mle-training")
    with mlflow.start_run(
        run_name="PARENT_RUN",
        experiment_id=experiment_id,
        description="parent",
    ) as parent_run:

        mlflow.log_param("parent", "yes")
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--data_folder_name", default="housing", help="data folder name"
        )
        parser.add_argument(
            "--artifacts_folder_name",
            default="housing",
            help="artifacts folder name",
        )
        parser.add_argument("--log-level", default=logging.DEBUG, help="log-level")

        args = parser.parse_args()

        ingest_data.ingest_data(args.log_level, experiment_id, args.data_folder_name)
        train.train(args.log_level, experiment_id, args.artifacts_folder_name)
        score.score(args.log_level, experiment_id, args.data_folder_name)
        print("run success")
