# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import subprocess
import tempfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path

from covid_xprize.validation.scenario_generator import get_raw_data, generate_scenario

# URL for Oxford data
DATA_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"

# Path to where this script lives
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Data directory (we will download the Oxford data to here)
DATA_PATH = ROOT_DIR / 'data'

# Path to Oxford data file
HIST_DATA_FILE_PATH = DATA_PATH / 'OxCGRT_latest.csv'

# Path to predictor module
PREDICT_MODULE = ROOT_DIR.parent.parent.parent / 'standard_predictor' / 'predict.py'

ADDITIONAL_CONTEXT_FILE = os.path.join(DATA_PATH, "Additional_Context_Data_Global.csv")
ADDITIONAL_US_STATES_CONTEXT = os.path.join(DATA_PATH, "US_states_populations.csv")
ADDITIONAL_UK_CONTEXT = os.path.join(DATA_PATH, "uk_populations.csv")
ADDITIONAL_BRAZIL_CONTEXT = os.path.join(DATA_PATH, "brazil_populations.csv")
US_PREFIX = "United States / "


CASES_COL = ['NewCases']

PRED_CASES_COL = ['PredictedDailyNewCases']

IP_COLS = ['C1_School closing',
            'C2_Workplace closing',
            'C3_Cancel public events',
            'C4_Restrictions on gatherings',
            'C5_Close public transport',
            'C6_Stay at home requirements',
            'C7_Restrictions on internal movement',
            'C8_International travel controls',
            'H1_Public information campaigns',
            'H2_Testing policy',
            'H3_Contact tracing',
            'H6_Facial Coverings']

IP_MAX_VALUES = {
    'C1_School closing': 3,
    'C2_Workplace closing': 3,
    'C3_Cancel public events': 2,
    'C4_Restrictions on gatherings': 4,
    'C5_Close public transport': 2,
    'C6_Stay at home requirements': 3,
    'C7_Restrictions on internal movement': 2,
    'C8_International travel controls': 4,
    'H1_Public information campaigns': 2,
    'H2_Testing policy': 3,
    'H3_Contact tracing': 2,
    'H6_Facial Coverings': 4
}


def add_geo_id(df):
    df["GeoID"] = np.where((df["RegionName"].isnull()) | (df.RegionName.str.len() == 0),
                                      df["CountryName"],
                                      df["CountryName"] + ' / ' + df["RegionName"])
    return df

# Function that performs basic loading and preprocessing of historical df
def prepare_historical_df():

    # Download data if it we haven't done that yet.
    if not os.path.exists(HIST_DATA_FILE_PATH):
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        urllib.request.urlretrieve(DATA_URL, HIST_DATA_FILE_PATH)

    # Load raw historical data
    df1 = pd.read_csv(HIST_DATA_FILE_PATH,
                  parse_dates=['Date'],
                  encoding="ISO-8859-1",
                  error_bad_lines=False)
    # Additional context df (e.g Population for each country)
    df2 = load_additional_context_df()

    # Add GeoID column for easier manipulation
    df1 = add_geo_id(df1)
    df1['RegionName'] = df1['RegionName'].fillna("")
    # Merge the 2 DataFrames
    df = df1.merge(df2, on=['GeoID'], how='left', suffixes=('', '_y'))

    # Add new cases column
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)

    # Fill any missing case values by interpolation and setting NaNs to 0
    df.update(df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))

    # Fill any missing IPs by assuming they are the same as previous day
    for ip_col in IP_MAX_VALUES:
        df.update(df.groupby('GeoID')[ip_col].ffill().fillna(0))

    return df


# Function to load an IPs file, e.g., passed to prescribe.py
def load_ips_file(path_to_ips_file):
    df = pd.read_csv(path_to_ips_file,
                     parse_dates=['Date'],
                     encoding="ISO-8859-1",
                     error_bad_lines=False)
    df['RegionName'] = df['RegionName'].fillna("")
    df = add_geo_id(df)
    return df

# Function that wraps predictor in order to query
# predictor when prescribing.
def get_predictions(start_date_str, end_date_str, pres_df, countries=None):

    # Concatenate prescriptions with historical data
    raw_df = get_raw_data(HIST_DATA_FILE_PATH)
    hist_df = generate_scenario(start_date_str, end_date_str, raw_df,
                                countries=countries, scenario='Historical')
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    hist_df = hist_df[hist_df.Date < start_date]
    ips_df = pd.concat([hist_df, pres_df])

    with tempfile.NamedTemporaryFile() as tmp_ips_file:
        # Write ips_df to file
        ips_df.to_csv(tmp_ips_file.name)

        with tempfile.NamedTemporaryFile() as tmp_pred_file:
            # Run script to generate predictions
            output_str = subprocess.check_output(
                [
                    'python', PREDICT_MODULE,
                    '--start_date', start_date_str,
                    '--end_date', end_date_str,
                    '--interventions_plan', tmp_ips_file.name,
                    '--output_file', tmp_pred_file.name
                ],
                stderr=subprocess.STDOUT
            )

            # Print output from running script
            print(output_str.decode("utf-8"))

            # Load predictions to return
            df1 = pd.read_csv(tmp_pred_file)
            df1 = add_geo_id(df1)

            # Additional context df (e.g Population for each country)
            df2 = load_additional_context_df()

            # Merge the 2 DataFrames
            df = df1.merge(df2, on=['GeoID'], how='left', suffixes=('', '_y'))

    return df

def load_additional_context_df():
    # File containing the population for each country
    # Note: this file contains only countries population, not regions
    additional_context_df = pd.read_csv(ADDITIONAL_CONTEXT_FILE,
                                        usecols=['CountryName', 'Population'])
    additional_context_df['GeoID'] = additional_context_df['CountryName']

    # US states population
    additional_us_states_df = pd.read_csv(ADDITIONAL_US_STATES_CONTEXT,
                                            usecols=['NAME', 'POPESTIMATE2019'])
    # Rename the columns to match measures_df ones
    additional_us_states_df.rename(columns={'POPESTIMATE2019': 'Population'}, inplace=True)
    # Prefix with country name to match measures_df
    additional_us_states_df['GeoID'] = US_PREFIX + additional_us_states_df['NAME']

    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_us_states_df)

    # UK population
    additional_uk_df = pd.read_csv(ADDITIONAL_UK_CONTEXT)
    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_uk_df)

    # Brazil population
    additional_brazil_df = pd.read_csv(ADDITIONAL_BRAZIL_CONTEXT)
    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_brazil_df)

    return additional_context_df