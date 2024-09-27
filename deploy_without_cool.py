import os
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import joblib
import utils
import time  

# Helper function to round up to the nearest 5th minute
def round_up_to_nearest_5_minutes(dt):
    minutes_to_next_interval = (5 - dt.minute % 5) % 5
    rounded_time = dt + timedelta(minutes=minutes_to_next_interval)
    rounded_time = rounded_time.replace(second=0, microsecond=0)
    return rounded_time

# Function to generate time series
def generate_time_series(start_time, step_minutes):
    tz = pytz.timezone('Asia/Kolkata')
    current_time = start_time
    while True:
        yield current_time
        current_time += timedelta(minutes=step_minutes)

def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M', errors='coerce', dayfirst=True)
    return df

def process_for_time(current_time, cabins, results_df):
    current_time = pd.to_datetime(current_time, format='%d-%m-%Y %H:%M:%S', errors='coerce')
    
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize('Asia/Kolkata')
    else:
        current_time = current_time.tz_convert('Asia/Kolkata')
    
    # Prepare a DataFrame for predictions
    df_time_cabins = pd.DataFrame({'Time': [current_time] * len(cabins), 'Cabin_No': cabins})
    df_selected_columns = utils.create_time_series_features(df_time_cabins.set_index('Time'))
    df_selected_columns['weekofyear'] = df_selected_columns['weekofyear'].astype('int32')
    df_selected_columns = df_selected_columns[['Cabin_No', 'hour', 'dayofweek', 'quarter', 'month', 'dayofyear', 'dayofmonth', 'weekofyear']]

    # Load classifiers
    xgb_classifier = joblib.load('/home/ubuntu/BLUE_STAR/lgb_classifier_idu.joblib')
    model_temp = joblib.load('/home/ubuntu/BLUE_STAR/lgb_temp_reg.joblib')
    ada_classifier_fan = joblib.load('/home/ubuntu/BLUE_STAR/ada_classifier_fan.joblib')
    ada_classifier_mode = joblib.load('/home/ubuntu/BLUE_STAR/ada_classifier_mode.joblib')

    orgin = df_time_cabins.copy()
    orgin['y_pred_Idu_Status'] = xgb_classifier.predict(df_selected_columns)

    # Predict temperature
    df_for_temp = orgin[['Time', 'Cabin_No', 'y_pred_Idu_Status']].rename(columns={'y_pred_Idu_Status': 'Idu_Status'}).set_index('Time')
    df_for_temp = utils.create_time_series_features(df_for_temp)
    df_for_temp['weekofyear'] = df_for_temp['weekofyear'].astype('int32')
    df_for_temp = df_for_temp[['Cabin_No', 'hour', 'Idu_Status', 'dayofweek', 'quarter', 'month', 'dayofyear', 'dayofmonth', 'weekofyear']]
    # Make predictions and apply conditions in one step
    # Make predictions and apply conditions in one step
    predictions = np.round(np.where(model_temp.predict(df_for_temp) < 5, 0, model_temp.predict(df_for_temp)))

    # Define the mapping dictionary
    temp_mapping = {0: 0, 1: 20, 2: 21, 3: 22, 4: 23, 5: 24, 6: 25, 7: 26}

    # Create a reverse mapping
    reverse_temp_mapping = {value: key for key, value in temp_mapping.items()}

    # Replace the rounded values with their corresponding keys using the reverse mapping
    orgin['y_pred_Temperature'] = [reverse_temp_mapping.get(value, value) for value in predictions]

    # Print the updated temperature predictions
    print(orgin['y_pred_Temperature'])
    df_for_fan = orgin[['Time', 'Cabin_No', 'y_pred_Idu_Status', 'y_pred_Temperature']].set_index('Time')
    df_for_fan = utils.create_time_series_features(df_for_fan)
    df_for_fan['weekofyear'] = df_for_fan['weekofyear'].astype('int32')
    df_for_fan = df_for_fan.rename(columns={'y_pred_Idu_Status': 'Idu_Status', 'y_pred_Temperature': 'Temperature'})
    df_for_fan = df_for_fan[['Cabin_No', 'hour', 'Idu_Status', 'Temperature', 'dayofweek', 'quarter', 'month', 'dayofyear', 'dayofmonth', 'weekofyear']]
    orgin['y_pred_FanSpeed'] = ada_classifier_fan.predict(df_for_fan)

    # Predict mode
    df_for_mode = orgin[['Time', 'Cabin_No', 'y_pred_Temperature', 'y_pred_Idu_Status', 'y_pred_FanSpeed']].set_index('Time')
    df_for_mode = utils.create_time_series_features(df_for_mode)
    df_for_mode['weekofyear'] = df_for_mode['weekofyear'].astype('int32')
    df_for_mode = df_for_mode.rename(columns={'y_pred_Idu_Status': 'Idu_Status', 'y_pred_Temperature': 'Temperature', 'y_pred_FanSpeed': 'FanSpeed'})
    df_for_mode = df_for_mode[['Cabin_No', 'hour', 'Idu_Status', 'FanSpeed', 'Temperature', 'dayofweek', 'quarter', 'month', 'dayofyear', 'dayofmonth', 'weekofyear']]
    orgin['y_pred_Mode'] = ada_classifier_mode.predict(df_for_mode)

    remapping_dict = {
        'Idu_Status': {0: 'OFF', 1: 'ON'},
        'Temperature': {0: 0, 1: 20, 2: 21, 3: 22, 4: 23, 5: 24, 6: 25, 7: 26},
        'FanSpeed': {0: '0', 1: 'High', 2: 'Low'},
        'Mode': {0: '0', 1: 'Cool'}
    }

    for cabin in cabins:
        formatted_time = current_time.strftime('%d-%m-%Y %H:%M:%S')
        predicted = {
            'Idu_Status': orgin.loc[orgin['Cabin_No'] == cabin, 'y_pred_Idu_Status'].values[0],
            'Temperature': orgin.loc[orgin['Cabin_No'] == cabin, 'y_pred_Temperature'].values[0],
            'FanSpeed': orgin.loc[orgin['Cabin_No'] == cabin, 'y_pred_FanSpeed'].values[0],
            'Mode': orgin.loc[orgin['Cabin_No'] == cabin, 'y_pred_Mode'].values[0]
        }
        
        # Remap predictions
        remapped_predicted = {
            'Idu_Status': remapping_dict['Idu_Status'][predicted['Idu_Status']],
            'Temperature': remapping_dict['Temperature'][predicted['Temperature']],
            'FanSpeed': remapping_dict['FanSpeed'][predicted['FanSpeed']],
            'Mode': remapping_dict['Mode'][predicted['Mode']]
        }

        # Create a new row for the results
        new_row = pd.DataFrame([{
            'Time': formatted_time,
            'Cabin_No': cabin,
            'Idu_Status': remapped_predicted['Idu_Status'],
            'Temperature': remapped_predicted['Temperature'],
            'FanSpeed': remapped_predicted['FanSpeed'],
            'Mode': remapped_predicted['Mode'],
            'Source': 'Prediction'
        }])
        
        # Append new row to results DataFrame
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    return results_df  # return updated results DataFrame

if __name__ == "__main__":
    # Set timezone to IST
    ist = pytz.timezone('Asia/Kolkata')

    # Get the current time in IST
    start_time = datetime.now(ist)
    print(f"Original Start Time (IST): {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Round the start time to the nearest 5th minute
    rounded_start_time = round_up_to_nearest_5_minutes(start_time)
    print(f"Rounded Start Time (IST): {rounded_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate the difference between the current start time and the rounded start time
    time_difference = (rounded_start_time - start_time).total_seconds()
    print(f"Time difference: {time_difference} seconds")

    # Wait for the difference in seconds before starting the regular processing
    if time_difference > 0:
        print(f"Waiting for {time_difference} seconds to align with the rounded start time...")
        time.sleep(time_difference)  # Sleep for the calculated difference

    cabins = [1, 2, 3]
    time_step_minutes = 5
    csv_file = 'results_1.csv'
    
    # Create a time generator that yields times at 5-minute intervals
    time_gen = generate_time_series(rounded_start_time, time_step_minutes)

    # Initialize variables
    next_save_time = rounded_start_time
    results_df = pd.DataFrame(columns=['Time', 'Cabin_No', 'Idu_Status', 'Temperature', 'FanSpeed', 'Mode', 'Source'])
    
    # Infinite loop to process data
    while True:
        current_time = next(time_gen)  # Get the next time in the series

        # Process data for the current time and cabins
        results_df = process_for_time(current_time, cabins, results_df)
        filtered_df = results_df[results_df['Time'] == current_time.strftime('%d-%m-%Y %H:%M:%S')]

        # Check if it's time to save the results
        if current_time >= next_save_time:
            if not filtered_df.empty:
                # Append filtered results to CSV
                filtered_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
                print(f"Appended filtered results at: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"No results found for {current_time.strftime('%Y-%m-%d %H:%M:%S')} to append.")
                
            next_save_time = current_time + timedelta(minutes=time_step_minutes)

        # Print the last few rows of the DataFrame to track progress
        print(results_df.tail())

        # Sleep until the next save time or next time step
        time_to_sleep = (next_save_time - current_time).total_seconds()
        print(f"Sleeping for {time_to_sleep} seconds until next time step or save time.")
        time.sleep(time_to_sleep)


