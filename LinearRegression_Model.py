import numpy as np

class LinearRegression:
    def __init__(self, features_count = 1, learning_rate = 0.1, iterations = 500, initial_weights = None, intercept = 0):
        self.features_count = features_count
        self.learning_rate = learning_rate
        self.weights = np.zeros(self.features_count) if initial_weights is not None else initial_weights
        self.intercept = intercept
        self.epochs = iterations

    def normalize_by_minmax(self, datapoints):
        min = np.min(datapoints, axis=0)
        max = np.max(datapoints, axis=0)
        normalized_features = (datapoints - min) / (max - min)
        return [normalized_features, min, max]

    def normalize_by_zscore(self, datapoints, mean = 0, std_deviation = 0):
        mean = np.mean(datapoints, axis=0) if mean == 0 else mean
        std_deviation = np.std(datapoints, axis=0) if std_deviation == 0 else std_deviation
        normalized_features = (datapoints - mean) / std_deviation
        return [normalized_features, mean, std_deviation]

    # Denormalize the predicted weights to compute output prediction

    def loss_function_MSE(self, predictions, actuals):
        error = predictions - actuals
        SSE = error ** 2
        MSE = np.mean(SSE)
        return MSE

    def get_cost_derivative(self, predictions, actuals, features_data):
        n = actuals.shape[0]
        # np.dot with the Transposed features matrix.
        # Shape: (2, n) dot (n, 1) = (2, 1) output
        feature_derivative_SSE = (2 / n) * np.dot(features_data.T, (predictions - actuals))
        intercept_derivative_SE = (2 / n) * np.sum(predictions - actuals)
        
        return feature_derivative_SSE, intercept_derivative_SE

    def update_coefficients_GD(self, weight_gradient, intercept_gradient):
        self.weights -= (self.learning_rate * weight_gradient)
        self.intercept -= (self.learning_rate * intercept_gradient)

    def training_prediction(self, features_data):
        predictions = np.dot() + self.intercept
        return predictions

    def train(self):
        for epoch in range(self.epochs):
            ...