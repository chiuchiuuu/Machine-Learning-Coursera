import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, y):
    """
    Plots the data points X and y into a new figure
    
    Parameters:
    --------
    X: array_like, (m, n)
    y: array_like, (m, )
    """
    
    pos = np.array(y) == 1
    neg = np.array(y) == 0
    
    plt.figure(figsize=(10, 8))
    plt.plot(X[pos, 0], X[pos, 1], 'go', linewidth=2, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'rx', linewidth=2, markersize=7)

    
    
def cost_function(theta, X, y):
    """
    Compute cost for logistic regression
    
    Parameters:
    --------
    theta: array_like, (n, )
        inital theta
    X: array_like, (m, n)
    y: array_like, (m, )
    """
    m, n = np.shape(X)
    theta = np.reshape(theta, (n, 1))
    y = np.reshape(y, (m, 1))
    
    log_loss1 = y.T @ np.log(sigmoid(X @ theta))
    log_loss2 = (1 - y).T @ 
    J = 1 / m * (-y.T.dot(np.log(sigmoid(X.dot(theta)))) - (1 - y).T.dot(np.log(1 - sigmoid(X.dot(theta)))))
    return J
    
def gradient(theta, X, y):
    """
    Compute gradient for cost_function
    
    Parameters:
    --------
    theta: array_like, (n, )
        inital theta
    X: array_like, (m, n)
    y: array_like, (m, )
    """
    m, n = np.shape(X)
    theta = np.reshape(theta, (n, 1))
    y = np.reshape(y, (m, 1))
    grad = 1 / m * np.dot(X.T, sigmoid(np.dot(X, theta)) - y)
    return grad.flatten()
    
def cost_function_reg(theta, X, y, lambda_):
    pass


def map_feature(X1, X2):
    pass

def plot_decision_boundary(theta, X, y):
    pass

def predict(theta, X):
    pass


def sigmoid(z):
    """
    Compute sigmoid function
    """

    return 1 / (1 + np.exp(-z))