import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-10  
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y) 
    for i in range(iterations):
        z = np.dot(X, weights)
        predictions = sigmoid(z)
          
        gradient = np.dot(X.T, (predictions - y)) / m
        
        weights -= learning_rate * gradient
        
        if i % 100 == 0:
            loss = cross_entropy_loss(y, predictions)
            print(f"Iteration {i}, Loss: {loss}")
    
    return weights


def predict(X, weights):
    return sigmoid(np.dot(X, weights)) >= 0.5


def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    weights = np.zeros(X.shape[1])
    
  
    weights = gradient_descent(X, y, weights, learning_rate, iterations)
    
    return weights


def evaluate(y_true, y_pred):
    """
    Evaluate accuracy.
    """
    return np.mean(y_true == y_pred)


def main():
    data = np.array([
        [0.1, 1.1, 0],
        [1.2, 0.9, 0],
        [1.5, 1.6, 1],
        [2.0, 1.8, 1],
        [2.5, 2.1, 1],
        [0.5, 1.5, 0],
        [1.8, 2.3, 1],
        [0.2, 0.7, 0],
        [1.9, 1.4, 1],
        [0.8, 0.6, 0]
    ])
    
    X = data[:, :-1]
    y = data[:, -1]
    
   
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    X = np.c_[np.ones(X.shape[0]), X]
    
    
    learning_rate = 0.1
    iterations = 1000
    weights = logistic_regression(X, y, learning_rate, iterations)

    y_pred = predict(X, weights)
    

    accuracy = evaluate(y, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
    probabilities = sigmoid(np.dot(grid, weights)).reshape(xx.shape)
    
    plt.contourf(xx, yy, probabilities, levels=[0, 0.5, 1], alpha=0.8, cmap="coolwarm")
    plt.scatter(X[:, 1], X[:, 2], c=y, edgecolor="k", cmap="coolwarm")
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1 (Normalized)")
    plt.ylabel("Feature 2 (Normalized)")
    plt.show()

if __name__ == "__main__":
    main()
