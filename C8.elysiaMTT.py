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

class DataProcessor:
    @staticmethod
    def align_time_periods(data):
        try:
            # Convert index to DatetimeIndex if not already
            for i, df in enumerate(data):
                if not isinstance(df.index, pd.DatetimeIndex):
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    data[i] = df
                    
            # Find common start and end dates
            common_start_date = max(df.index[0] for df in data)
            common_end_date = min(df.index[-1] for df in data)
            
            # Find common columns
            common_columns = list(set.intersection(*(set(df.columns) for df in data)))
            
            # Filter, resample, and interpolate dataframes
            filtered_data = []
            for df in data:
                df = df[common_columns]
                df = df.loc[common_start_date:common_end_date].resample('D').mean().interpolate()
                filtered_data.append(df)
            
            return filtered_data
        except Exception as e:
            print(f"Error aligning time periods: {e}")
            return None
    
    @staticmethod
    def calculate_correlation_matrix(train_data, obs_data):
        try:
            df = pd.concat([train_data, *obs_data], axis=1)
            correlation_matrix = df.corr()
            print("Correlation Matrix:")
            print(correlation_matrix)
            return correlation_matrix
        except Exception as e:
            print(f"Error calculating correlation matrix: {e}")
            return None

    @staticmethod
    def extract_feature_importance(correlation_matrix):
        try:
            if correlation_matrix.empty:
                print("Correlation matrix is empty.")
                return None
            
            feature_importance = correlation_matrix.iloc[0].abs()
            feature_importance /= feature_importance.sum()
            return feature_importance.values
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            return None

class FeatureImportanceCalculator:
    @staticmethod
    def calculate_feature_importance(train_data, obs_data):
        try:
            train_data, *obs_data = DataProcessor.align_time_periods([train_data, *obs_data])
            
            # Calculate correlation matrix
            correlation_matrix = DataProcessor.calculate_correlation_matrix(train_data, obs_data)
            
            if correlation_matrix is None:
                raise ValueError("Correlation matrix calculation failed or returned None.")
            
            if correlation_matrix.empty:
                print("Correlation matrix is empty.")
                return None
            
            # Extract and normalize feature importance
            feature_importance = DataProcessor.extract_feature_importance(correlation_matrix)
            
            if feature_importance is None:
                raise ValueError("Feature importance calculation failed.")
            
            return feature_importance
        except Exception as e:
            print(f"Feature importance calculation error: {e}")
            return None

class StateSpaceModel:
    def __init__(self, ar_params, ma_params):
        self.ar_params = ar_params
        self.ma_params = ma_params

    def state_transition_function(self, state):
        return np.dot(self.ar_params, state)

class ExtendedKalmanFilter:
    def __init__(self, state_space_model, process_noise_cov):
        self.state_space_model = state_space_model
        self.process_noise_cov = process_noise_cov
        self.state_estimation = None
        self.state_covariance = None

    def initialize(self, initial_state, initial_state_cov):
        self.state_estimation = initial_state
        self.state_covariance = initial_state_cov

    def predict(self, external_factors):
        predicted_state = self.state_space_model.state_transition_function(self.state_estimation)
        self.state_estimation = predicted_state + np.dot(external_factors, self.state_space_model.ar_params)

    def update(self, observation):
        state_transition_jacobian = np.eye(len(self.state_estimation))
        state_transition_jacobian[-1, -1] = 1
        kalman_gain = np.dot(np.dot(self.state_covariance, state_transition_jacobian.T),
                             np.linalg.inv(np.dot(np.dot(state_transition_jacobian, self.state_covariance), state_transition_jacobian.T) + self.process_noise_cov))
        predicted_state = self.state_space_model.state_transition_function(self.state_estimation)
        self.state_estimation += np.dot(kalman_gain, (observation - predicted_state))
        self.state_covariance = np.dot((np.eye(len(self.state_covariance)) - np.dot(kalman_gain, state_transition_jacobian)), self.state_covariance)

class CWLEM:
    def __init__(self, beta):
        self.beta = beta

    def predict(self, external_factors):
        return np.dot(external_factors, self.beta)

class PerformanceEvaluator:
    @staticmethod
    def calculate_rmsle(predictions, true_values):
        return np.sqrt(np.mean(np.square(np.log(predictions + 1) - np.log(true_values + 1))))

def end_to_end_pipeline(train_data, obs_data):
    try:
        feature_importance = FeatureImportanceCalculator.calculate_feature_importance(train_data, obs_data)
        if feature_importance is not None:
            state_space_model = StateSpaceModel([], [])
            initial_state = np.zeros(len(train_data.columns) + len(obs_data))
            initial_state_cov = np.eye(len(initial_state))
            process_noise_cov = np.eye(len(initial_state))
            ekf = ExtendedKalmanFilter(state_space_model, process_noise_cov)
            ekf.initialize(initial_state, initial_state_cov)
            cwlem = CWLEM(feature_importance)

            predicted_sales = []
            true_sales = train_data['sales'].values
            for i, observation in enumerate(true_sales):
                external_factors = np.array([obs_data[j].iloc[i].values[0] for j in range(len(obs_data))])
                cwlem_prediction = cwlem.predict(external_factors)
                ekf.predict(cwlem_prediction)
                ekf.update(observation)
                predicted_sales.append(ekf.state_estimation[-1])

            rmsle = PerformanceEvaluator.calculate_rmsle(predicted_sales, true_sales)
            print("RMSLE:", rmsle)
        else:
            print("Feature importance calculation failed.")
    except Exception as e:
        print("An error occurred:", e)

def main():
    base_path = '/kaggle/input/store-sales-time-series-forecasting/'
    train_file = base_path + 'train.csv'
    obs_files = [base_path + 'holidays_events.csv',
                 base_path + 'oil.csv',
                 base_path + 'transactions.csv']
    obs_columns = [['date'], ['date', 'dcoilwtico'], ['date', 'transactions']]

    train_data = DataLoader.load_data(train_file, ['date', 'sales'])
    obs_data = [DataLoader.load_data(file, cols) for file, cols in zip(obs_files, obs_columns)]

    # Debugging: Print the loaded dataframes
    print("Train Data:")
    print(train_data.head())
    print("\nObservation Data:")
    for i, df in enumerate(obs_data):
        print(f"\nDataFrame {i+1}:")
        print(df.head())

    if train_data is not None and all(obs is not None for obs in obs_data):
        end_to_end_pipeline(train_data, obs_data)
    else:
        print("Train data or observation data is missing or empty.")

if __name__ == "__main__":
    main()


