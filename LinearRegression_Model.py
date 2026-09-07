import numpy as np

class LinearRegression:
    def __init__(self, features_count = 1, learning_rate = 0.1, epochs = 1000, initial_weights = None, intercept = 0):
        self.features_count = features_count
        self.learning_rate = learning_rate
        self.weights = np.zeros(self.features_count) if initial_weights is None else np.asarray(initial_weights, dtype=float)
        self.intercept = intercept
        self.epochs = epochs
        self.min_vals = None
        self.max_vals = None
        self.mean_vals = None
        self.std_vals = None

    def normalize_by_minmax(self, datapoints):
        min = np.min(datapoints, axis=0)
        max = np.max(datapoints, axis=0)
        feature_range = np.where(max - min == 0, 1, max - min)
        normalized_features = (datapoints - min) / feature_range
        self.min_vals = min
        self.max_vals = max
        return [normalized_features, min, max]

    def normalize_by_zscore(self, datapoints, mean = 0, std_deviation = 0):
        mean = np.mean(datapoints, axis=0) if mean == 0 else mean
        std_deviation = np.std(datapoints, axis=0) if std_deviation == 0 else std_deviation
        std_deviation = np.where(std_deviation == 0, 1, std_deviation)
        normalized_features = (datapoints - mean) / std_deviation
        return [normalized_features, mean, std_deviation]

    def denormalize_coefficients(self, method="zscore"):
        if method == "minmax" and self.min_vals is not None and self.max_vals is not None:
            feature_range = np.where(self.max_vals - self.min_vals == 0, 1, self.max_vals - self.min_vals)
            weights = self.weights / feature_range
            intercept = self.intercept - np.sum(self.weights * self.min_vals / feature_range)
            return weights, intercept

        if method != "zscore" or self.mean_vals is None or self.std_vals is None:
            return self.weights, self.intercept

        weights = self.weights / self.std_vals
        intercept = self.intercept - np.sum(self.weights * self.mean_vals / self.std_vals)
        return weights, intercept

    def loss_function_MSE(self, predictions, actuals):
        error = predictions - actuals
        SSE = error ** 2
        MSE = np.mean(SSE)
        return MSE

    def get_cost_derivative(self, predictions, actuals, features_data):
        n = actuals.shape[0]
        # np.dot with the Transposed features matrix.
        # Shape: (2, n) dot (n, 1) = (2, 1) output
        feature_derivative_SSE = ((2 / n) * np.dot(
            features_data.T, (predictions - actuals)
        )).reshape(-1)
        intercept_derivative_SE = (2 / n) * np.sum(predictions - actuals)
        
        return feature_derivative_SSE, intercept_derivative_SE

    def update_coefficients_GD(self, weight_gradient, intercept_gradient):
        self.weights -= (self.learning_rate * weight_gradient)
        self.intercept -= (self.learning_rate * intercept_gradient)

    def training_prediction(self, features_data):
        # Compute predictions: y = X * weights + intercept
        # features_data shape: (n_samples, features_count), weights shape: (features_count,)
        predictions = np.dot(features_data, self.weights) + self.intercept
        # Return as nx1 column vector
        return predictions.reshape(-1, 1)

    def train(self, features_data, outputs):
        features_data = np.asarray(features_data, dtype=float)
        outputs = np.asarray(outputs, dtype=float).reshape(-1, 1)

        if features_data.ndim != 2 or features_data.shape[1] != self.features_count:
            raise ValueError(
                f"features_data must have shape (n_samples, {self.features_count})"
            )

        # Normalize before gradient descent; all epoch calculations use this matrix.
        normalized_features, self.mean_vals, self.std_vals = self.normalize_by_zscore(features_data)

        for epoch in range(self.epochs):
            predictions = self.training_prediction(normalized_features)
            cost = self.loss_function_MSE(predictions, outputs)

            TOLERANCE = 1e-12  # "Close enough to zero"
            if cost < TOLERANCE:
                print(f"Cost near zero ({cost:.6f}) at epoch {epoch}!")
                print(" Model has converged perfectly!")
                break

            if epoch % 50 == 0:
                # Show denormalized values for interpretation
                denormalized_weights, denormalized_intercept = self.denormalize_coefficients()
                print(f"Epoch {epoch}: Cost = {cost:.6f}, "
                      f"Weights = {denormalized_weights}, "
                      f"Intercept = {denormalized_intercept:.4f}")
            
            weights_gradient, intercept_gradient = self.get_cost_derivative(
                predictions, outputs, normalized_features
            )
            self.update_coefficients_GD(weights_gradient, intercept_gradient)

        return self.denormalize_coefficients()

    # Model inference
    def predict(self, features):
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.features_count:
            raise ValueError(
                f"features must have shape (n_samples, {self.features_count})"
            )

        # Inference uses raw features and coefficients converted back to raw scale.
        weights, intercept = self.denormalize_coefficients()
        predictions = np.dot(features, weights) + intercept
        return predictions.reshape(-1, 1)