# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse
import os

from xprize_predictor import XPrizePredictor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed weights for the standard predictor.
MODEL_WEIGHTS_FILE = os.path.join(ROOT_DIR, "models", "trained_model_weights.h5")

DATA_FILE = os.path.join(ROOT_DIR, 'data', "OxCGRT_latest.csv")

def predict(full_start_date: str,
            start_date: str,
            end_date: str,
            path_to_ips_file: str,
            output_file_path: str,
            old_preds_file=None,
            old_ips_file=None) -> None:
    """
    Generates and saves a file with daily new cases predictions for the given countries, regions and intervention
    plans, between start_date and end_date, included.
    :param start_date: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between inception date (Jan 1 2020)
     and end_date, for the countries and regions for which a prediction is needed
    :param output_file_path: path to file to save the predictions to
    :return: Nothing. Saves the generated predictions to an output_file_path CSV file
    with columns "CountryName,RegionName,Date,PredictedDailyNewCases"
    """
    # !!! YOUR CODE HERE !!!
    predictor = XPrizePredictor()
    # Generate the predictions
    preds_df = predictor.predict(full_start_date, start_date, end_date, path_to_ips_file, old_preds_file, old_ips_file)
    # Create the output path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # Save to a csv file
    preds_df.to_csv(output_file_path, index=False)
    print(f"Saved predictions to {output_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-fs", "--full_start_date",
                        dest="full_start_date",
                        type=str,
                        required=True,
                        help="Full start date from which to predict, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to predict, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prediction, included, as YYYY-MM-DD. For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_plan",
                        dest="ip_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    parser.add_argument("-p", "--old_predictions",
                        dest="old_preds",
                        type=str,
                        required=False,
                        help="The path to old predictions .csv file")
    parser.add_argument("-i", "--old_ips",
                        dest="old_ips",
                        type=str,
                        required=False,
                        help="The path to old ips .csv file")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to the CSV file where predictions should be written")
    args = parser.parse_args()
    print(f"Generating predictions from {args.start_date} to {args.end_date}...")
    predict(args.full_start_date, args.start_date, args.end_date, args.ip_file, args.output_file, args.old_preds, args.old_ips)
    print("Done!")
