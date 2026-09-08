#!/usr/bin/env python3
"""
Simple Linear Regression with Feature Normalization
"""

import numpy as np
# import matplotlib.pyplot as plt

feature_1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
target = np.array([2.0, 4.5, 7.0, 9.5, 12.0, 14.5, 17.0, 19.5, 22.0, 24.5, 27.0, 29.5, 32.0, 34.5, 37.0, 39.5, 42.0, 44.5, 47.0, 49.5])

# ============================================================
# FEATURE NORMALIZATION (CRITICAL for GD convergence!)
# ============================================================
feature_mean = np.mean(feature_1)
feature_std = np.std(feature_1)

# Normalized features (zero mean, unit variance)
feature_1_norm = (feature_1 - feature_mean) / feature_std

# Store for denormalization later
feature_mean_orig = feature_mean
feature_std_orig = feature_std

print("=" * 70)
print("FEATURE NORMALIZATION")
print("=" * 70)
print(f"Original features: {np.min(feature_1)} to {np.max(feature_1)}")
print(f"Mean: {feature_mean:.2f}, Standard Deviation: {feature_std:.2f}")
print(f"Normalized features: {np.min(feature_1_norm):.2f} to {np.max(feature_1_norm):.2f}")
print(f"  (Now have μ≈0, σ≈1)")
print()

# ============================================================
# DEFINE FUNCTIONS (using normalized features internally)
# ============================================================

def loss_function_MSE(predictions, actuals):
    no_of_datapoints = len(actuals)
    SSE = 0
    for i in range(no_of_datapoints):
        error = predictions[i] - actuals[i]
        SSE += (error ** 2)
    MSE = 1 * SSE / float(no_of_datapoints)
    return MSE

def get_cost_derivative(predictions, actuals, X1_data, intercept = False):
    no_of_datapoints = len(actuals)
    cost_derivative = 0
    derivative_SSE = 0
    for i in range(no_of_datapoints):
        feature_data = X1_data[i] if not intercept else 1
        derivative_SSE += ((predictions[i] - actuals[i]) * feature_data)
    cost_derivative = 1 * derivative_SSE / float(no_of_datapoints)
    return cost_derivative

def update_coefficients_GD(coefficient, learning_rate, gradient):
    return (coefficient - learning_rate * gradient)

# ============================================================
# INITIALIZE PARAMETERS (in normalized space)
# ============================================================
# Start with zeros in normalized space
X1_norm = 0  # weight in normalized feature space
b = 0        # bias (not affected by normalization scale)

# hyper params - adjusted for normalized features
learning_rate = 0.1  # α - works well with normalized features
iterations = 500

# ============================================================
# GRADIENT DESCENT LOOP (using normalized features)
# ============================================================
cost_history = []  # Track cost for convergence analysis

print("=" * 70)
print("GRADIENT DESCENT PROGRESS")
print("=" * 70)

for iteration in range(iterations):
    pred_list = []
    for index in range(len(feature_1)):
        # USE NORMALIZED FEATURES FOR PREDICTION
        prediction = X1_norm * feature_1_norm[index] + b
        pred_list.append(prediction)
    pred_np_array = np.array(pred_list)
    cost = loss_function_MSE(pred_np_array, target)
    cost_history.append(cost)  # Store cost for this iteration
    
    # Check if cost is close to zero (allow for floating-point precision)
    TOLERANCE = 1e-6  # "Close enough to zero"
    
    if cost < TOLERANCE:
        print(f"🎯 Cost near zero ({cost:.6f}) at iteration {iteration}!")
        print("   Model has converged perfectly!")
        break
    
    # Also check if cost stopped improving (extra safety)
    if iteration > 0 and abs(cost - cost_history[-2]) < 1e-8:
        print("🛑 Cost plateau detected - stopping early")
        break

    if iteration % 50 == 0:
        # Show denormalized values for interpretation
        X1_denorm = X1_norm / feature_std_orig  # Convert back to original scale
        print(f"Iteration {iteration}: X1_norm={X1_norm:.4f}, "
              f"X1_original≈{X1_denorm:.4f}, b={b:.4f}, cost={cost:.2f}")

    #parameter update (using normalized gradients)
    X1_norm = update_coefficients_GD(X1_norm, learning_rate, 
                                     get_cost_derivative(pred_np_array, target, feature_1_norm))
    b = update_coefficients_GD(b, learning_rate, 
                              get_cost_derivative(pred_np_array, target, feature_1_norm, True))

# ============================================================
# DENORMALIZE FINAL COEFFICIENTS
# ============================================================
# Convert back to original feature scale
X1_original = X1_norm / feature_std_orig
b_original = b - X1_norm * (feature_mean_orig / feature_std_orig)

print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Final values (normalized space): X1_norm = {X1_norm:.4f}, b = {b:.4f}")
print(f"Final values (original scale):   X1 = {X1_original:.4f}, b = {b_original:.4f}")
print()
print(f"✅ Expected: X1 ≈ 2.5, b ≈ -0.5")
print(f"The linear equation => {X1_original:.4f} * feature_1 + {b_original:.4f}")