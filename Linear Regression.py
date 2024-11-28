def calculate_mean(values):
    return sum(values) / len(values)

def calculate_slope(X, Y, mean_X, mean_Y):
    numerator = sum((x - mean_X) * (y - mean_Y) for x, y in zip(X, Y))
    denominator = sum((x - mean_X) ** 2 for x in X)
    return numerator / denominator

def calculate_intercept(mean_X, mean_Y, slope):
    return mean_Y - slope * mean_X

def predict(X, theta_0, theta_1):
    return [theta_0 + theta_1 * x for x in X]

def calculate_mse(Y, Y_pred):
    return sum((y - y_pred) ** 2 for y, y_pred in zip(Y, Y_pred)) / len(Y)

def fit_linear_regression(X, Y):
    mean_X = calculate_mean(X)
    mean_Y = calculate_mean(Y)
    slope = calculate_slope(X, Y, mean_X, mean_Y)
    intercept = calculate_intercept(mean_X, mean_Y, slope)
    return intercept, slope

if __name__ == "__main__":
    X = [1, 2, 3, 4, 5]
    Y = [2, 4, 5, 7, 8]
    
    theta_0, theta_1 = fit_linear_regression(X, Y)
    print(f"Model Parameters: Intercept (theta_0) = {theta_0}, Slope (theta_1) = {theta_1}")
    
    Y_pred = predict(X, theta_0, theta_1)
    print(f"Predictions: {Y_pred}")
    
    mse = calculate_mse(Y, Y_pred)
    print(f"Mean Squared Error (MSE): {mse}")
