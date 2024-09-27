
# Prediction System for Cabin Data

This project is a prediction system that processes time series data and predicts the status, temperature, fan speed, and mode for different cabins. The system uses multiple machine learning models to generate predictions for each cabin at regular intervals.

## Project Structure

- **main.py**: The main script that runs the time series prediction process.
- **deploy_without_cool.py**: Contains utility functions for generating time series features.
- **lgb_classifier_idu.joblib**: Model used to predict the Idu_Status.
- **lgb_temp_reg.joblib**: Model used to predict the temperature.
- **ada_classifier_fan.joblib**: Model used to predict the fan speed.
- **ada_classifier_mode.joblib**: Model used to predict the mode.

## Dependencies

Make sure to install the dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

1. Set up your environment by installing the necessary dependencies.
2. Run the main script (`deploy_without_cool.py`) to start the prediction process.
3. The system will generate predictions for different cabins and save the results in `results_1.csv`.

### Features

- Rounds the current time to the nearest 5-minute interval.
- Generates predictions for Idu Status, Temperature, Fan Speed, and Mode.
- Saves the predictions to a CSV file at regular intervals.

## Configuration

- Time series data generation is done at 5-minute intervals.
- The prediction models are loaded from pre-trained `.joblib` files.

