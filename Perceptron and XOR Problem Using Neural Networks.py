import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Perceptron Implementation
def perceptron(X, y, learning_rate=0.1, epochs=100):
    """
    Train a Perceptron model.
    """
    num_features = X.shape[1]
    weights = np.zeros(num_features)
    bias = 0

    for epoch in range(epochs):
        for i in range(X.shape[0]):
            linear_output = np.dot(X[i], weights) + bias
            prediction = 1 if linear_output >= 0 else 0
            error = y[i] - prediction

            # Update weights and bias
            weights += learning_rate * error * X[i]
            bias += learning_rate * error

    return weights, bias

def plot_perceptron_boundary(X, y, weights, bias):
    """
    Visualize the decision boundary for the Perceptron.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias
    Z = Z >= 0
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title("Perceptron Decision Boundary")
    plt.show()

# XOR Neural Network Implementation
def train_xor_nn(X, y):
    """
    Train a neural network to solve the XOR problem.
    """
    model = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', max_iter=10000, random_state=0)
    model.fit(X, y)
    return model

def visualize_xor_boundary(model, X, y):
    """
    Visualize the decision boundary for the XOR problem.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title("XOR Neural Network Decision Boundary")
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Dataset for Perceptron
    X_perceptron = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_perceptron = np.array([0, 0, 0, 1])

    # Train Perceptron
    weights, bias = perceptron(X_perceptron, y_perceptron)
    plot_perceptron_boundary(X_perceptron, y_perceptron, weights, bias)

    # Dataset for XOR
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])

    # Train Neural Network
    xor_model = train_xor_nn(X_xor, y_xor)
    visualize_xor_boundary(xor_model, X_xor, y_xor)
