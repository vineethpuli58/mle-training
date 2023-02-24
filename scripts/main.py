import argparse
import logging

from mle_training import ingest_data, score, train

if __name__ == "__main__":

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

    ingest_data.ingest_data(args.log_level, args.data_folder_name)
    train.train(args.log_level, args.artifacts_folder_name)
    score.score(args.log_level, args.data_folder_name)
    print("run success")
