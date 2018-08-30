import numpy as np
import matplotlib.pyplot as plt

def warmup_exercise():
    """
    an example function that returns the 5x5 identity matrix
    """
    pass

def plot_data(X, y):
    """
    Plots the data points x and y into a new figure 
    """
    pass

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
    pass


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
    pass


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
    pass

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
    pass