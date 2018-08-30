import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    X

def warmup_exercise():
    """
    an example function that returns the 5x5 identity matrix
    """
    return np.eye(5)

def plot_data(X, y):
    """
    Plots the data points x and y into a new figure 
    """
    plt.figure(figsize=(10, 8))
    plt.plot(X, y, 'rx', markersize=10, label="Trainng Data")
    plt.ylabel("Profit in $10,000s")
    plt.xlabel("Population of City in 10,000s")

def compute_cost(X, y, theta):
    """
    compute cost
    
    Parameters:
    --------
    X: array_like, (m, n)
    y: array_like, (m, )
    theta: shape (n, )
    
    Returns:
    --------
    J: float
        cost value
    """
    m = len(y)
    J = np.sum((np.dot(X, theta) - y) ** 2) / (2 * m)
    return J

def gradient_descent(X, y, theta, alpha, n_iters):
    """
    Performs gradient descent to learn theta
    
    Parameters:
    --------
    X: array_like, (m, n)
    y: array_like, (m, )
    theta: (n, )
    alpha: float
        learning rate
    n_iters: int
        number of iterations
        
    Returns:
    --------
    theta: array_like, shape (n, )
        final weight
    J_history: array_like, shape (n_iters, )
        list of values of cost function
    """
    m = len(y)
    J_history = np.zeros(n_iters)
    for i in range(n_iters):
        theta = theta - alpha / m * np.dot(X.T, (np.dot(X, theta) - y))
        J_history[i] = compute_cost(X, y, theta)
    return theta, J_history


def feature_normalize(X):
        """
    returns a normalized version of X 
    where the mean value of each feature is 0
    and the standard deviation is 1.

    Parameters:
    --------
    X: array_like, (m, n)
    
    Returns:
    --------
    X_norm: array_like, (m, n)
        normalize training set
    mu: array_like, (n, )
        mean value of each feature
    sigma: array_like (n, )
        standard deviation of each feature
    """
    
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def normal_equation(X, y):
    """
    Computes the closed-form solution to linear regression using normal euqation
    
    Parameters:
    --------
    X: array_like, (m, n)
    y: array_like, (m, )
    
    Returns:
    theta: array_like, (n, )
    --------
    """
    return np.linalg.pinv(np.dot(X.T, X)).dot(X.T).dot(y)