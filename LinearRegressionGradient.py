import numpy as np

def calculate_mean(values):
    return np.mean(values)

def calculate_slope(X, Y, mean_X, mean_Y):
    numerator = np.sum((X - mean_X) * (Y - mean_Y))
    denominator = np.sum((X - mean_X) ** 2)
    return numerator / denominator

def calculate_intercept(mean_X, mean_Y, slope):
    return mean_Y - slope * mean_X

def predict(X, theta_0, theta_1):
    return theta_0 + theta_1 * X

def calculate_mse(Y, Y_pred):
    return np.mean((Y - Y_pred) ** 2)

def gradient_descent(X, Y, theta_0, theta_1, learning_rate, iterations):
    m = len(Y)
    for _ in range(iterations):
        Y_pred = predict(X, theta_0, theta_1)
        d_theta_0 = -2 / m * np.sum(Y - Y_pred)
        d_theta_1 = -2 / m * np.sum((Y - Y_pred) * X)
        theta_0 -= learning_rate * d_theta_0
        theta_1 -= learning_rate * d_theta_1
    return theta_0, theta_1

def fit_linear_regression(X, Y, learning_rate=0.01, iterations=1000):
    mean_X = calculate_mean(X)
    mean_Y = calculate_mean(Y)
    theta_1 = calculate_slope(X, Y, mean_X, mean_Y)
    theta_0 = calculate_intercept(mean_X, mean_Y, theta_1)
    Y_pred_initial = predict(X, theta_0, theta_1)
    mse_initial = calculate_mse(Y, Y_pred_initial)
    print(f"Initial MSE: {mse_initial}")
    theta_0, theta_1 = gradient_descent(X, Y, theta_0, theta_1, learning_rate, iterations)
    Y_pred_optimized = predict(X, theta_0, theta_1)
    mse_optimized = calculate_mse(Y, Y_pred_optimized)
    print(f"Optimized MSE: {mse_optimized}")
    return theta_0, theta_1

def test_model():
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Y = np.array([30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000])
    learning_rate = 0.01
    iterations = 1000
    theta_0, theta_1 = fit_linear_regression(X, Y, learning_rate, iterations)
    print(f"Final Theta_0: {theta_0}")
    print(f"Final Theta_1: {theta_1}")
    Y_pred = predict(X, theta_0, theta_1)
    mse = calculate_mse(Y, Y_pred)
    print(f"Final MSE: {mse}")

if __name__ == "__main__":
    test_model()
