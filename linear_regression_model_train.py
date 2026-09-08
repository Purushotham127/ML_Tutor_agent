import os
import numpy as np
import pandas as pd
from LinearRegression_Model import LinearRegression

# Read data from csv file
dataset = pd.read_csv('./dataset/data.csv')

dataset_np_array = np.array(dataset)
features_data = dataset_np_array[:, :-1]  # Ignores last col(assuming output), every feature will be copied
outputs = dataset_np_array[:, -1]

print(f"Shape of Features from dataset : {features_data.shape}")
print(f"Shape of Output from dataset : {outputs.shape}")

features_count = features_data.shape[1]

# Intialize hyper parameters
learning_rate = 0.05
epochs = 1000

# Train/test split (80/20), shuffled so row order in the csv doesn't bias either set
np.random.seed(42)
n_samples = features_data.shape[0]
shuffled_indices = np.random.permutation(n_samples)
test_size = int(n_samples * 0.2)
test_indices = shuffled_indices[:test_size]
train_indices = shuffled_indices[test_size:]

features_train = features_data[train_indices]
outputs_train = outputs[train_indices]
features_test = features_data[test_indices]
outputs_test = outputs[test_indices]

# Intialize the model
model = LinearRegression(features_count, learning_rate, epochs)

# Train only on the training split
weights, intercept = model.train(features_train, outputs_train)
model.save_parameters("linear_regression_params.pkl")
print(f"Model training completed! weights : {weights} and intercept(bias) : {intercept}")

# Evaluate on the held-out test split
test_predictions = model.predict(features_test).reshape(-1)
mse = np.mean((test_predictions - outputs_test) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(test_predictions - outputs_test))

# R^2: proportion of variance in outputs explained by the model (regression's "accuracy")
ss_res = np.sum((outputs_test - test_predictions) ** 2)
ss_tot = np.sum((outputs_test - np.mean(outputs_test)) ** 2)
r2_score = 1 - (ss_res / ss_tot)

print(f"Test MSE  : {mse:.4f}")
print(f"Test RMSE : {rmse:.4f}")
print(f"Test MAE  : {mae:.4f}")
print(f"Test R^2  : {r2_score:.4f}")