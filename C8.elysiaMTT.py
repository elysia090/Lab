import numpy as np
import pandas as pd

class DataLoader:
    @staticmethod
    def load_data(file_path, columns):
        try:
            data = pd.read_csv(file_path, usecols=columns)
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except pd.errors.EmptyDataError:
            print(f"File is empty: {file_path}")
            return None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

class FeatureImportanceCalculator:
    @staticmethod
    def calculate_feature_importance(train_data, obs_data):
        try:
            # Merge data and remove duplicates
            df = pd.concat([train_data] + obs_data, axis=1)
            df = df.loc[:, ~df.columns.duplicated()]

            # Convert 'date' column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime, invalid values will be NaT

            # Drop rows with missing or invalid date values
            df = df.dropna(subset=['date'])

            # Convert date column to numerical feature (day of year)
            df['date'] = df['date'].dt.dayofyear.astype(float)

            # Compute correlation matrix
            correlation_matrix = df.corr()

            if not correlation_matrix.empty:
                # Exclude the first row ('sales' column) and calculate feature importance
                feature_importance = correlation_matrix.iloc[0].abs()
                feature_importance /= feature_importance.sum()
                return feature_importance.values[1:]  # Exclude 'sales' column
            else:
                print("Correlation matrix is empty.")
                return None
        except Exception as e:
            print("Error calculating feature importance:", e)
            return None

class StateSpaceModel:
    def __init__(self, state_dim, process_noise_cov):
        self.state_dim = state_dim
        self.process_noise_cov = process_noise_cov

class ExtendedKalmanFilter:
    def __init__(self, state_space_model):
        self.state_space_model = state_space_model
        self.state_estimation = None
        self.state_covariance = None

    def initialize(self, initial_state, initial_state_cov):
        self.state_estimation = initial_state.astype(float)
        self.state_covariance = initial_state_cov.astype(float)

    def predict(self, external_factors):
        predicted_state = self.state_estimation  # No state transition function for the given state space model
        self.state_estimation = predicted_state + external_factors

    def update(self, observation):
        kalman_gain = np.dot(self.state_covariance, np.linalg.inv(self.state_covariance + self.state_space_model.process_noise_cov))
        self.state_estimation += np.dot(kalman_gain, (observation - self.state_estimation))
        self.state_covariance = np.dot((np.eye(self.state_space_model.state_dim) - kalman_gain), self.state_covariance)

class CWLEM:
    def __init__(self, beta_length):
        self.beta = np.zeros(beta_length, dtype=float)

    def set_beta(self, beta):
        self.beta[:len(beta)] = beta

    def predict(self, external_factors):
        return np.dot(external_factors, self.beta)

class PerformanceEvaluator:
    @staticmethod
    def calculate_rmsle(predictions, true_values):
        return np.sqrt(np.mean(np.square(np.log(predictions + 1) - np.log(true_values + 1))))

def main():
    base_path = '/kaggle/input/store-sales-time-series-forecasting/'
    train_file = base_path + 'train.csv'
    obs_files = [base_path + 'holidays_events.csv',
                 base_path + 'oil.csv',
                 base_path + 'transactions.csv']
    obs_columns = [['date'], ['date', 'dcoilwtico'], ['date', 'transactions']]

    train_data = DataLoader.load_data(train_file, ['date', 'sales'])
    obs_data = [DataLoader.load_data(file, cols) for file, cols in zip(obs_files, obs_columns)]

    if train_data is not None and all(obs is not None for obs in obs_data):
        try:
            print("Calculating feature importance...")
            feature_importance = FeatureImportanceCalculator.calculate_feature_importance(train_data, obs_data)
            if feature_importance is not None:
                print("Feature importance calculation successful.")

                # Initialize Extended Kalman Filter
                state_dim = len(train_data.columns) + sum(len(cols) - 1 for cols in obs_columns)
                process_noise_cov = np.eye(state_dim)
                state_space_model = StateSpaceModel(state_dim, process_noise_cov)
                ekf = ExtendedKalmanFilter(state_space_model)
                initial_state = np.zeros(state_dim)
                initial_state_cov = np.eye(state_dim)
                ekf.initialize(initial_state, initial_state_cov)

                # Initialize CWLEM for sales prediction
                cwlem = CWLEM(len(feature_importance))
                cwlem.set_beta(feature_importance)

                predicted_sales = []
                true_sales = train_data['sales'].values

                for i, observation in enumerate(true_sales):
                    external_factors = []
                    for obs_df, cols in zip(obs_data, obs_columns):
                        # Check if the DataFrame contains the 'date' column
                        if 'date' in obs_df.columns:
                            # Filter the DataFrame to get the row corresponding to the current date
                            obs_row = obs_df.loc[obs_df['date'] == train_data.iloc[i]['date']]
                            # Check if any rows are found
                            if not obs_row.empty:
                                # Extract external factors from additional columns
                                external_factors.extend(obs_row[cols[1:]].values.flatten())
                            else:
                                # Handle missing data for the current date
                                print(f"No data found for date {train_data.iloc[i]['date']} in observation {i}")
                                external_factors.extend([np.nan] * (len(cols) - 1))
                        else:
                            # Handle missing 'date' column
                            print("No 'date' column found in observation data.")
                            external_factors.extend([np.nan] * (len(cols) - 1))

                    print("External factors:", external_factors)  # Debugging print
                    cwlem_prediction = cwlem.predict(external_factors)
                    ekf.predict(cwlem_prediction)
                    ekf.update(observation)
                    predicted_sales.append(ekf.state_estimation[-1])

                # Evaluate performance
                rmsle = PerformanceEvaluator.calculate_rmsle(predicted_sales, true_sales)
                print("RMSLE:", rmsle)

            else:
                print("Feature importance calculation failed.")
        except Exception as e:
            print("An error occurred:", e)
    else:
        print("Train data or observation data is missing or empty.")

if __name__ == "__main__":
    main()








