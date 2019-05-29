#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Learning Online Class
Exercise 1: Linear regression with multiple variables

author: zhuzi   date: 2019/05/23    version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt


def featureNormalize(X):
    """
    FEATURENORMALIZE Normalizes the features in X
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    """

    # 计算每一列的均值、方差
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)


def computeCostMulti(X, y, theta):
    """
    COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """

    return (np.mean((X @ theta - y)**2) / 2)


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    m = len(y)  # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter in range(num_iters):
        theta -= alpha * ((X @ theta - y).T @ X).T / m
        # Save the cost J in every iteration
        J_history[iter] = computeCostMulti(X, y, theta)

    return (theta, J_history)


def normalEqn(X, y):
    """
    NORMALEQN Computes the closed-form solution to linear regression
    NORMALEQN(X,y) computes the closed-form solution to linear
    regression using the normal equations.
    """

    return np.linalg.pinv(X.T @ X) @ X.T @ y


if __name__ == '__main__':

    # ================ Part 1: Feature Normalization ================
    print('Loading data ...')

    # Load Data
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2].reshape(-1, 1)
    m = len(y)

    # Print out some data points
    print('First 10 examples from the dataset:')
    for row in data[0:10, :]:
        print(' x = [%.0f %.0f], y = %.0f' % (row[0], row[1], row[2]))

    input('Program paused. Press enter to continue.')
    # Scale features and set them to zero mean
    print('Normalizing Features ...')

    (X, mu, sigma) = featureNormalize(X)

    # Add intercept term to X
    X = np.insert(X, 0, values=np.ones(m), axis=1)

    # ================ Part 2: Gradient Descent ================
    print('Running gradient descent ...')
    # Choose some alpha value
    alpha = 0.01
    num_iters = 400

    # Init Theta and Run Gradient Descent
    theta = np.zeros((3, 1))
    (theta, J_history) = gradientDescentMulti(X, y, theta, alpha, num_iters)

    # Plot the convergence graph
    plt.plot(range(J_history.size), J_history, '-b', linewidth=2)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    # Display gradient descent's result
    print('Theta computed from gradient descent:\n', theta)
    # Estimate the price of a 1650 sq-ft, 3 br house
    price = np.c_[np.array([[1]]), (np.array(
        [[1650, 3]]) - mu) / sigma] @ theta

    print(
        'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): $%f' %
        price)

    input('Program paused. Press enter to continue.')
    # ================ Part 3: Normal Equations ================
    print('Solving with normal equations...')

    # Load Data
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2].reshape(-1, 1)

    # Add intercept term to X
    X = np.insert(X, 0, values=np.ones(m), axis=1)

    # Calculate the parameters from the normal equation
    theta = normalEqn(X, y)

    # Display normal equation's result
    print('Theta computed from the normal equations:\n', theta)

    # Estimate the price of a 1650 sq-ft, 3 br house
    price = np.array([[1, 1650, 3]]) @ theta

    print(
        'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): $%f' %
        price)