# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

"""
This is the prescribe.py script for a simple example prescriptor that
generates IP schedules that trade off between IP cost and cases.

The prescriptor is "blind" in that it does not consider any historical
data when making its prescriptions.

The prescriptor is "greedy" in that it starts with all IPs turned off,
and then iteratively turns on the unused IP that has the least cost.

Since each subsequent prescription is stricter, the resulting set
of prescriptions should produce a Pareto front that highlights the
trade-off space between total IP cost and cases.

Note this file has significant overlap with ../random/prescribe.py.
"""

import os
import argparse
import numpy as np
import pandas as pd
import math
import time
import warnings

from datetime import datetime

from covid_xprize.examples.prescriptors.blind_greedy.utils import add_geo_id
from covid_xprize.examples.prescriptors.blind_greedy.utils import get_predictions
from covid_xprize.examples.prescriptors.blind_greedy.utils import load_ips_file
from covid_xprize.examples.prescriptors.blind_greedy.utils import prepare_historical_df

# Constant imports from utils
from covid_xprize.examples.prescriptors.blind_greedy.utils import CASES_COL
from covid_xprize.examples.prescriptors.blind_greedy.utils import IP_COLS
from covid_xprize.examples.prescriptors.blind_greedy.utils import IP_MAX_VALUES
from covid_xprize.examples.prescriptors.blind_greedy.utils import PRED_CASES_COL

NUM_PRESCRIPTIONS = 2
ACTION_DURATION = 21

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

IP_COLS = list(IP_MAX_VALUES.keys())
warnings.simplefilter(action='ignore', category=FutureWarning)

def quad(x): 
    return (-pow(x, 2) + 2*x) * .96


def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_hist_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:

    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

    # Load the past IPs data
    print("Loading past IPs data...")
    past_ips_df = load_ips_file(path_to_hist_file)
    geos = past_ips_df['GeoID'].unique()

    # Load historical data with basic preprocessing
    print("Loading historical data...")
    df = prepare_historical_df()
    df["PredictedDailyNewCases"] = 0
    df["NewCases"] = 0
    df["PredictionRatio"] = 0

    # Restrict it to dates before the start_date
    df = df[df['Date'] < start_date]

    # Set up dictionary for keeping track of prescription
    df_dicts = []
    past_cases = []
    past_ips = []
    baseline_cases = []
    for _ in range(1, NUM_PRESCRIPTIONS):
        df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
        for ip_col in sorted(IP_MAX_VALUES.keys()):
            df_dict[ip_col] = []
        df_dicts.append(df_dict)
        
        past_cases.append(df)
        max_date_str = df.Date.max()
        max_date = pd.to_datetime(max_date_str, format='%Y-%m-%d')
        trunc_ips = past_ips_df[past_ips_df['Date'] <= max_date]
        past_ips.append(trunc_ips)
        baseline_case = {}
        for geo in geos:
            baseline_case[geo] = 0.
        baseline_cases.append(baseline_case)

    # Fill in any missing case data before start_date
    # using predictor given past_ips_df.
    # Note that the following assumes that the df returned by prepare_historical_df()
    # has the same final date for all regions. This has been true so far, but relies
    # on it being true for the Oxford data csv loaded by prepare_historical_df().
    last_historical_data_date_str = df['Date'].max()
    last_historical_data_date = pd.to_datetime(last_historical_data_date_str,
                                               format='%Y-%m-%d')
    if last_historical_data_date + pd.Timedelta(days=1) < start_date:
        print("Filling in missing data...")
        missing_data_start_date = last_historical_data_date + pd.Timedelta(days=1)
        missing_data_start_date_str = datetime.strftime(missing_data_start_date,
                                                           format='%Y-%m-%d')
        missing_data_end_date = start_date - pd.Timedelta(days=1)
        missing_data_end_date_str = datetime.strftime(missing_data_end_date,
                                                           format='%Y-%m-%d')
        pred_df = get_predictions(missing_data_start_date_str,
                                  missing_data_start_date_str,
                                  missing_data_end_date_str,
                                  past_ips_df,
                                  None,
                                  None)
        pred_df = add_geo_id(pred_df)
        for i in range(NUM_PRESCRIPTIONS-1):
            curr_presc_pred_df = past_cases[i]
            past_cases[i] = pd.concat([curr_presc_pred_df, pred_df])
            curr_ip_df = past_ips[i]
            missing_ip_df = past_ips_df[(past_ips_df["Date"] >= missing_data_start_date) & (past_ips_df["Date"] <= missing_data_end_date)]
            past_ips[i] = pd.concat([curr_ip_df, missing_ip_df])
    else:
        print("No missing data.")
    
    prescription_dict = {
        'PrescriptionIndex': [],
        'CountryName': [],
        'RegionName': [],
        'Date': []
    }
    for ip in IP_COLS:
        prescription_dict[ip] = []

    
    # Load IP costs to condition prescriptions
    cost_df = pd.read_csv(path_to_cost_file)
    cost_df['RegionName'] = cost_df['RegionName'].fillna("")
    cost_df = add_geo_id(cost_df)
    geo_costs = {}
    max_stringency = {}
    
    print("Calculating max stringencies")
    for geo in geos: 
        costs = cost_df[cost_df['GeoID'] == geo]
        cost_arr = np.array(costs[IP_COLS])[0]
        geo_costs[geo] = cost_arr
        max_values = np.array([IP_MAX_VALUES[ip_col] for ip_col in IP_COLS]).reshape(1,-1)
        max_stringency[geo] = np.sum(np.multiply(cost_arr, max_values))
        geo_split = geo.split(' / ')
        if len(geo_split) == 1:
            region_name = np.nan
        else:
            region_name = geo_split[1]
        country_name = geo_split[0]
        for date in pd.date_range(start_date, end_date):
            prescription_dict['PrescriptionIndex'].append(0)
            prescription_dict['CountryName'].append(country_name)
            prescription_dict['RegionName'].append(region_name)
            prescription_dict['Date'].append(date.strftime("%Y-%m-%d"))
            for ip in IP_COLS:
                prescription_dict[ip].append(0)

    action_start_date = start_date
    pred_start_date = action_start_date
    while action_start_date <= end_date:
        action_start_date_str = action_start_date.strftime("%Y-%m-%d")
        pred_start_date_str = pred_start_date.strftime("%Y-%m-%d")
        print(action_start_date_str)
        for i in range(1, NUM_PRESCRIPTIONS):
            print("Prescription ", i)
            pred_df = None
            pres_df = None
            if action_start_date != start_date:
                # print("Getting predictions from ", start_date_str, "to ", action_start_date_str)
                # Make prediction given prescription for all countries
                pres_df = pd.DataFrame(df_dicts[i-1])
                pres_df = pres_df[(pres_df['Date'] >= start_date_str) & (pres_df['Date'] <= action_start_date_str)]
                pres_df = add_geo_id(pres_df)
                start_time = time.time()
                # print(past_cases[i-1])
                # print(past_ips[i-1])
                pred_df = get_predictions(start_date_str, pred_start_date_str, action_start_date_str, pres_df, past_cases[i-1], past_ips[i-1])
                # pred_df = get_predictions(start_date_str, pred_start_date_str, action_start_date_str, pres_df, None, None)
                print("Seconds taken: ", time.time() - start_time)
                pred_start_date = action_start_date
                pred_df = add_geo_id(pred_df)
                past_cases_df = past_cases[i-1]
                past_cases[i-1] = pd.concat([past_cases_df, pred_df]).drop_duplicates(subset=['GeoID', 'Date'])
            for geo in geos:
                # print("Geo ", geo)
                geo_split = geo.split(' / ')
                if len(geo_split) == 1:
                    region_name = np.nan
                else:
                    region_name = geo_split[1]
                country_name = geo_split[0]
                curr_ips = {ip: 0 for ip in IP_MAX_VALUES}
                curr_ip_idx = 0
                curr_stringency = 0.
                fill_stringency = True
                multiplier = 1
                if pred_df is not None:
                    seven_day = action_start_date - pd.DateOffset(days=7)
                    seven_day_str = seven_day.strftime("%Y-%m-%d")
                    new_pred_df = pred_df[(pred_df['Date'] >= seven_day_str) & (pred_df['Date'] < action_start_date_str)]
                    geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]
                    cases_mean = geo_pred[PRED_CASES_COL].mean()['PredictedDailyNewCases']
                    # print("Baseline: ", baseline_cases[i-1][geo])
                    # print("Cases: ", cases_mean)
                    if cases_mean > baseline_cases[i-1][geo]:
                        # print("Use max stringency")
                        fill_stringency = True
                    else:
                        #print("Can pull back on NPIs")
                        fill_stringency = False
                        multiplier = cases_mean/baseline_cases[i-1][geo] if baseline_cases[i-1][geo] > 0 else 1.
                        # print("Geo: ", geo)
                        # print("Multiplier:", multiplier)
                        quad_multiplier = quad(multiplier)
                        multiplier = quad_multiplier if quad_multiplier > multiplier else multiplier
                        multiplier = min(multiplier, 1.)
                        # print("New multiplier: ", multiplier)
                    baseline_cases[i-1][geo] = cases_mean
                else:
                    cases_df = past_cases[i-1]
                    cases_geo_df = cases_df[cases_df["GeoID"] == geo]
                    cases_geo_df = cases_geo_df[cases_geo_df.ConfirmedCases.notnull()]
                    prev_confirmed_cases = np.array(cases_geo_df.ConfirmedCases)
                    if len(prev_confirmed_cases) > 2:
                        baseline_cases[i-1][geo] = prev_confirmed_cases[-1] - prev_confirmed_cases[-2]
                curr_max_stringency = (max_stringency[geo] / (NUM_PRESCRIPTIONS - 1)) * i if fill_stringency else multiplier * (max_stringency[geo] / (NUM_PRESCRIPTIONS - 1)) * i
                ip_weights = geo_costs[geo]
                ip_names = IP_COLS
                sorted_ips = [(weight, ip) for weight, ip in sorted(zip(ip_weights, ip_names))]
                while curr_stringency < curr_max_stringency:
                    if (curr_ip_idx >= len(sorted_ips)):
                        break
                    next_ip = sorted_ips[curr_ip_idx][1]
                    next_weight = sorted_ips[curr_ip_idx][0]
                    value = 1
                    while value < IP_MAX_VALUES[next_ip]:
                        if value * next_weight + curr_stringency > curr_max_stringency:
                            break
                        value += 1
                    curr_ips[next_ip] = value
                    curr_ip_idx += 1
                    if curr_stringency + value * next_weight < curr_max_stringency:
                        curr_stringency += value * next_weight
                    else:
                        break
                curr_df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
                for ip_col in sorted(IP_MAX_VALUES.keys()):
                    curr_df_dict[ip_col] = []
                for date in pd.date_range(action_start_date, periods=ACTION_DURATION):
                    if date.strftime("%Y-%m-%d") > end_date_str:
                        break
                    
                    prescription_dict['PrescriptionIndex'].append(i)
                    prescription_dict['CountryName'].append(country_name)
                    prescription_dict['RegionName'].append(region_name)
                    prescription_dict['Date'].append(date.strftime("%Y-%m-%d"))
                    for ip in IP_COLS:
                        prescription_dict[ip].append(curr_ips[ip])
                    curr_df_dict['CountryName'].append(country_name)
                    curr_df_dict['RegionName'].append(region_name)
                    curr_df_dict['Date'].append(date.strftime("%Y-%m-%d"))
                    for ip in IP_COLS:
                        curr_df_dict[ip].append(curr_ips[ip])
                    
                df_dicts[i-1]['CountryName'] += curr_df_dict['CountryName']
                df_dicts[i-1]['RegionName'] += curr_df_dict['RegionName']
                df_dicts[i-1]['Date'] += curr_df_dict['Date']
                for ip in IP_COLS:
                    df_dicts[i-1][ip] += curr_df_dict[ip]
                past_ips_df = past_ips[i-1]
                curr_ips_df = pd.DataFrame(curr_df_dict)
                curr_ips_df = add_geo_id(curr_ips_df)
                past_ips[i-1] = pd.concat([past_ips_df, curr_ips_df]).drop_duplicates(subset=['GeoID', 'Date'])


        action_start_date += pd.DateOffset(days=ACTION_DURATION)

    # Create dataframe from dictionary.
    prescription_df = pd.DataFrame(prescription_dict)

    # Create the directory for writing the output file, if necessary.
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save output csv file.
    prescription_df.to_csv(output_file_path, index=False)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to prescribe, included, as YYYY-MM-DD."
                             "For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prescription, included, as YYYY-MM-DD."
                             "For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_past",
                        dest="prev_file",
                        type=str,
                        required=True,
                        help="The path to a .csv file of previous intervention plans")
    parser.add_argument("-c", "--intervention_costs",
                        dest="cost_file",
                        type=str,
                        required=True,
                        help="Path to a .csv file containing the cost of each IP for each geo")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    args = parser.parse_args()
    print(f"Generating prescriptions from {args.start_date} to {args.end_date}...")
    prescribe(args.start_date, args.end_date, args.prev_file, args.cost_file, args.output_file)
    print("Done!")
