import os
import urllib.request

import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


# See https://github.com/OxCGRT/covid-policy-tracker
DATA_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data')
DATA_FILE_PATH = os.path.join(DATA_PATH, 'OxCGRT_latest.csv')

ID_COLUMNS = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']
CASES_COLUMN = ['NewCases']
NPI_COLUMNS = ['C1_School closing',
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
NB_LOOKBACK_DAYS = 30
NB_TEST_DAYS = 60
WINDOW_SIZE = 7
US_PREFIX = "United States / "
NUM_TRIALS = 1
MAX_NB_COUNTRIES = 20
# For testing, restrict training data to that before a hypothetical predictor submission date
HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-12-10")

class XPrizeLinearPredictor(object):

    def __init__(self, path_to_model, data_url):
        if path_to_model:
            self.model_path = path_to_model
            # Make sure data is available to make predictions
            if not os.path.exists(DATA_FILE_PATH):
                urllib.request.urlretrieve(DATA_URL, DATA_FILE_PATH)

        self.df = self._prepare_dataframe(data_url)
        self.country_samples = self._create_country_samples(self.df)

    def predict(self, start_date_str: str, end_date_str: str, path_to_ips_file: str, verbose=False):
        """
        Generates a file with daily new cases predictions for the given countries, regions and npis, between
        start_date and end_date, included.
        :param start_date_str: day from which to start making predictions, as a string, format YYYY-MM-DDD
        :param end_date_str: day on which to stop making predictions, as a string, format YYYY-MM-DDD
        :param path_to_ips_file: path to a csv file containing the intervention plans between inception_date and end_date
        :param verbose: True to print debug logs
        :return: a Pandas DataFrame containing the predictions
        """
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

        # Load historical intervention plans, since inception
        hist_ips_df = pd.read_csv(path_to_ips_file,
                                parse_dates=['Date'],
                                encoding="ISO-8859-1",
                                dtype={"RegionName": str},
                                error_bad_lines=True)

        # Add GeoID column that combines CountryName and RegionName for easier manipulation of data",
        hist_ips_df['GeoID'] = np.where(hist_ips_df["RegionName"].isnull(),
                                        hist_ips_df["CountryName"],
                                        hist_ips_df["CountryName"] + ' / ' + hist_ips_df["RegionName"])
        # Fill any missing NPIs by assuming they are the same as previous day
        for npi_col in NPI_COLUMNS:
            hist_ips_df.update(hist_ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

        # Intervention plans to forecast for: those between start_date and end_date
        ips_df = hist_ips_df[(hist_ips_df.Date >= start_date) & (hist_ips_df.Date <= end_date)]

        # Load historical data to use in making predictions in the same way
        # This is the data we trained on
        # We stored it locally as for predictions there will be no access to the internet
        hist_cases_df = pd.read_csv(DATA_FILE_PATH,
                                    parse_dates=['Date'],
                                    encoding="ISO-8859-1",
                                    dtype={"RegionName": str,
                                        "RegionCode": str},
                                    error_bad_lines=False)
        # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
        hist_cases_df['GeoID'] = np.where(hist_cases_df["RegionName"].isnull(),
                                        hist_cases_df["CountryName"],
                                        hist_cases_df["CountryName"] + ' / ' + hist_cases_df["RegionName"])
        # Add new cases column
        hist_cases_df['NewCases'] = hist_cases_df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
        # Fill any missing case values by interpolation and setting NaNs to 0
        hist_cases_df.update(hist_cases_df.groupby('GeoID').NewCases.apply(
            lambda group: group.interpolate()).fillna(0))
        # Keep only the id and cases columns
        hist_cases_df = hist_cases_df[ID_COLUMNS + CASES_COLUMN]

        # Load model
        with open(self.model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        # Make predictions for each country,region pair
        geo_pred_dfs = []
        for g in ips_df.GeoID.unique():
            if verbose:
                print('\nPredicting for', g)

            # Pull out all relevant data for country c
            hist_cases_gdf = hist_cases_df[hist_cases_df.GeoID == g]
            last_known_date = hist_cases_gdf.Date.max()
            ips_gdf = ips_df[ips_df.GeoID == g]
            past_cases = np.array(hist_cases_gdf[CASES_COLUMN])
            past_npis = np.array(hist_ips_df[NPI_COLUMNS])
            future_npis = np.array(ips_gdf[NPI_COLUMNS])

            # Make prediction for each day
            geo_preds = []
            # Start predicting from start_date, unless there's a gap since last known date
            current_date = min(last_known_date + np.timedelta64(1, 'D'), start_date)
            days_ahead = 0
            while current_date <= end_date:
                # Prepare data
                X_cases = past_cases[-NB_LOOKBACK_DAYS:]
                X_npis = past_npis[-NB_LOOKBACK_DAYS:]
                X = np.concatenate([X_cases.flatten(),
                                    X_npis.flatten()])

                # Make the prediction (reshape so that sklearn is happy)
                pred = model.predict(X.reshape(1, -1))[0]
                pred = max(0, pred)  # Do not allow predicting negative cases
                # Add if it's a requested date
                if current_date >= start_date:
                    geo_preds.append(pred)
                    if verbose:
                        print(f"{current_date.strftime('%Y-%m-%d')}: {pred}")
                else:
                    if verbose:
                        print(f"{current_date.strftime('%Y-%m-%d')}: {pred} - Skipped (intermediate missing daily cases)")

                # Append the prediction and npi's for next day
                # in order to rollout predictions for further days.
                past_cases = np.append(past_cases, pred)
                past_npis = np.append(past_npis, future_npis[days_ahead:days_ahead + 1], axis=0)

                # Move to next day
                current_date = current_date + np.timedelta64(1, 'D')
                days_ahead += 1

            # Create geo_pred_df with pred column
            geo_pred_df = ips_gdf[ID_COLUMNS].copy()
            geo_pred_df['PredictedDailyNewCases'] = geo_preds
            geo_pred_dfs.append(geo_pred_df)

        # Combine all predictions into a single dataframe
        pred_df = pd.concat(geo_pred_dfs)

        # Drop GeoID column to match expected output format
        pred_df = pred_df.drop(columns=['GeoID'])
        return pred_df


    def _prepare_dataframe(self, data_url: str) -> pd.DataFrame:
        """
        Loads the Oxford dataset, cleans it up and prepares the necessary columns. Depending on options, also
        loads the Johns Hopkins dataset and merges that in.
        :param data_url: the url containing the original data
        :return: a Pandas DataFrame with the historical data
        """
        # Original df from Oxford
        df = self._load_original_data(data_url)

        #  Keep only needed columns
        columns = ID_COLUMNS + NPI_COLUMNS + ['ConfirmedCases', 'ConfirmedDeaths']
        df = df[columns]

        # Fill in missing values
        self._fill_missing_values(df)

        df = df[df.Date <= HYPOTHETICAL_SUBMISSION_DATE]

        # Compute number of new cases and deaths each day
        df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
        df['NewDeaths'] = df.groupby('GeoID').ConfirmedDeaths.diff().fillna(0)
        # Fill any missing case values by interpolation and setting NaNs to 0
        df.update(df.groupby('GeoID').NewCases.apply(
            lambda group: group.interpolate()).fillna(0))

        # Replace negative values (which do not make sense for these columns) with 0
        df['NewCases'] = df['NewCases'].clip(lower=0)
        df['NewDeaths'] = df['NewDeaths'].clip(lower=0)
        

        return df

    @staticmethod
    def _load_original_data(data_url):
        latest_df = pd.read_csv(data_url,
                                parse_dates=['Date'],
                                encoding="ISO-8859-1",
                                dtype={"RegionName": str,
                                       "RegionCode": str},
                                error_bad_lines=False)
        # GeoID is CountryName / RegionName
        # np.where usage: if A then B else C
        latest_df["GeoID"] = np.where(latest_df["RegionName"].isnull(),
                                      latest_df["CountryName"],
                                      latest_df["CountryName"] + ' / ' + latest_df["RegionName"])
        return latest_df

    @staticmethod
    def _fill_missing_values(df):
        """
        # Fill missing values by interpolation, ffill, and filling NaNs
        :param df: Dataframe to be filled
        """
        df.update(df.groupby('GeoID').ConfirmedCases.apply(
            lambda group: group.interpolate(limit_area='inside')))
        # Drop country / regions for which no number of cases is available
        df.dropna(subset=['ConfirmedCases'], inplace=True)
        df.update(df.groupby('GeoID').ConfirmedDeaths.apply(
            lambda group: group.interpolate(limit_area='inside')))
        # Drop country / regions for which no number of deaths is available
        df.dropna(subset=['ConfirmedDeaths'], inplace=True)
        for npi_column in NPI_COLUMNS:
            df.update(df.groupby('GeoID')[npi_column].ffill().fillna(0))

    # Helpful function to compute mae
    @staticmethod
    def mae(pred, true):
        return np.mean(np.abs(pred - true))

    @staticmethod
    def _create_country_samples(df: pd.DataFrame):
        """
        For each country, creates numpy arrays for Keras
        :param df: a Pandas DataFrame with historical data for countries (the "Oxford" dataset)
        :param geos: a list of geo names
        :return: a dictionary of train and test sets, for each specified country
        """
        # Create training data across all countries for predicting one day ahead
        X_samples = []
        y_samples = []
        geo_ids = df.GeoID.unique()
        for g in geo_ids:
            gdf = df[df.GeoID == g]
            all_case_data = np.array(gdf[CASES_COLUMN])
            all_npi_data = np.array(gdf[NPI_COLUMNS])

            # Create one sample for each day where we have enough data
            # Each sample consists of cases and npis for previous nb_lookback_days
            nb_total_days = len(gdf)
            for d in range(NB_LOOKBACK_DAYS, nb_total_days - 1):
                X_cases = all_case_data[d-NB_LOOKBACK_DAYS:d]

                # Take negative of npis to support positive
                # weight constraint in Lasso.
                X_npis = -all_npi_data[d - NB_LOOKBACK_DAYS:d]

                # Flatten all input data so it fits Lasso input format.
                X_sample = np.concatenate([X_cases.flatten(),
                                            X_npis.flatten()])
                y_sample = all_case_data[d + 1]
                X_samples.append(X_sample)
                y_samples.append(y_sample)

        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples).flatten()
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_samples,
                                                    y_samples,
                                                    test_size=0.2,
                                                    random_state=301)
        return X_train, X_test, y_train, y_test

    def train(self):
        # Create and train Lasso model.
        # Set positive=True to enforce assumption that cases are positively correlated
        # with future cases and npis are negatively correlated.
        self.model = Lasso(alpha=0.1,
              precompute=True,
              max_iter=10000,
              positive=True,
              selection='random')

        X_train, _, y_train, _ = self.country_samples
        

        # Clip outliers
        MIN_VALUE = 0.
        MAX_VALUE = 2.
        X_train = np.clip(X_train, MIN_VALUE, MAX_VALUE)
        y_train = np.clip(y_train, MIN_VALUE, MAX_VALUE)

        # Fit model
        self.model.fit(X_train, y_train)

        # Save model to file
        if not os.path.exists('models'):
            os.mkdir('models')
        with open('model.pkl', 'wb') as model_file:
            pickle.dump(self.model, model_file)

        return self.model