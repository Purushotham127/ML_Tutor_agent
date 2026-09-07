import numpy as np

class LinearRegression:
    def __init__(self, features_count = 1, learning_rate = 0.1, epochs = 500, initial_weights = None, intercept = 0):
        self.features_count = features_count
        self.learning_rate = learning_rate
        self.weights = np.zeros(self.features_count) if initial_weights is not None else initial_weights
        self.intercept = intercept
        self.epochs = epochs

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
        # Update the prediction to weights * features + intercept
        predictions = np.dot() + self.intercept
        return predictions

    def train(self, features_data, outputs):
        for epoch in range(self.epochs):
            predictions = self.training_prediction(features_data)
            cost = self.loss_function_MSE(predictions, outputs)

            TOLERANCE = 1e-6  # "Close enough to zero"
            if cost < TOLERANCE:
                print(f"🎯 Cost near zero ({cost:.6f}) at epoch {epoch}!")
                print(" Model has converged perfectly!")
                break

            if epoch % 50 == 0:
                # Show denormalized values for interpretation
                # Convert back to original scale
                denormalized_weights = None

                #Print them which will be useful for debugging
            
            weights_gradient, intercept_gradient = self.get_cost_derivative(predictions,outputs,features_data)
            self.update_coefficients_GD(weights_gradient, intercept_gradient)

    # Model inference
    def predict(self, features):
        return self.training_prediction(features)