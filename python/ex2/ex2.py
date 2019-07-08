#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Learning Online Class - Exercise 2: Logistic Regression

Author: Six     Date: 2019/06/21    Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


def plotData(X, y):
    # PLOTDATA Plots the data points X and y into a new figure
    # PLOTDATA(x,y) plots the data points with + for the positive examples
    # and o for the negative examples. X is assumed to be a Mx2 matrix.
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    plt.scatter(X[pos, 0], X[pos, 1], marker='+', label='Admitted')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', label='Not admitted')


def costFunction(theta, X, y):
    # COSTFUNCTION Compute cost and gradient for logistic regression
    # J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    # parameter for logistic regression and the gradient of the cost
    # w.r.t. to the parameters.

    # Initialize some useful values
    m = np.size(y)  # number of training examples

    h = sigmoid(X @ theta.reshape(-1, 1))
    J = (-y.T @ np.log(h) - (1 - y).T @ np.log(1 - h)) / m

    return J


def Gradient(theta, X, y):

    m = np.size(y)  # number of training examples

    h = sigmoid(X @ theta.reshape(-1, 1))
    grad = ((h - y).T @ X).T / m

    return grad


def sigmoid(z):
    # SIGMOID Compute sigmoid function
    # g = SIGMOID(z) computes the sigmoid of z.

    return 1 / (1 + np.exp(-z))


def plotDecisionBoundary(theta, X, y):
    # PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    # the decision boundary defined by theta
    # PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    # positive examples and o for the negative examples. X is assumed to be
    # a either
    # 1) Mx3 matrix, where the first column is an all-ones column for the
    # intercept.
    # 2) MxN, N>3 matrix, where the first column is all-ones

    # Plot Data
    plotData(X[:, 1:3], y)

    if np.size(X, 1) <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        # 解释一下，求出的theta其实是最小代价所代表的那个水平切平面，θ0 + θ1‧X1 + θ2‧X2 = 0
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y, label='Decision Boundary')

        # Legend, specific for the exercise
        plt.legend(loc=0)
        plt.axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros([len(u), len(v)])
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i][j] = mapFeature(u[i], v[j]) @ theta.reshape(-1, 1)

        z = z.T  # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the range [0, 0]，因为theta*X=0是决策边界
        plt.contour(u, v, z, label='Decision Boundary')


def predict(theta, X):
    # PREDICT Predict whether the label is 0 or 1 using learned logistic
    # regression parameters theta
    # p = PREDICT(theta, X) computes the predictions for X using a
    # threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    # 直接四舍五入
    p = np.round(sigmoid(X @ theta.reshape(-1, 1)))
    return p


def mapFeature(X1, X2):
    # MAPFEATURE Feature mapping function to polynomial features
    #
    # MAPFEATURE(X1, X2) maps the two input features
    # to quadratic features used in the regularization exercise.
    #
    # Returns a new feature array with more features, comprising of
    # X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    # Inputs X1, X2 must be the same size
    #

    degree = 6
    out = np.ones(np.size(X1)).reshape(-1, 1)
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.c_(out, np.power(X1, i - j) * np.power(X2, j))

    return out


if __name__ == '__main__':

    data = np.loadtxt('ex2data1.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2].reshape(-1, 1)

    # ==================== Part 1: Plotting ====================
    print('Plotting data with + indicating (y = 1) examples and indicating (y = 0) examples.')
    plotData(X, y)

    # Labels and Legend
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    # Specified in plot order
    plt.legend(loc=1)
    plt.show()

    print('\nProgram paused. Press enter to continue.')

    # ============ Part 2: Compute Cost and Gradient ============

    #  Setup the data matrix appropriately, and add ones for the intercept term
    (m, n) = X.shape

    # Add intercept term to x and X_test
    X = np.c_[np.ones(m), X]

    # Initialize fitting parameters
    initial_theta = np.zeros(n + 1)

    # Compute and display initial cost and gradient
    cost = costFunction(initial_theta, X, y)
    grad = Gradient(initial_theta, X, y)

    print('Cost at initial theta (zeros): %s' % cost)
    print('Expected cost (approx): 0.693')
    print('Gradient at initial theta (zeros):')
    print(grad)
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')

    # Compute and display cost and gradient with non-zero theta
    test_theta = np.array([-24, 0.2, 0.2])
    cost = costFunction(test_theta, X, y)
    grad = Gradient(test_theta, X, y)

    print('\nCost at test theta: %s' % cost)
    print('Expected cost (approx): 0.218')
    print('Gradient at test theta:')
    print(grad)
    print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647')

    print('\nProgram paused. Press enter to continue.')

    # ============= Part 3: Optimizing using fminunc  =============
    result = op.minimize(
        fun=costFunction,
        x0=initial_theta,
        args=(
            X,
            y),
        method='TNC',
        jac=Gradient)
    cost = result.fun
    theta = result.x

    # Print theta to screen
    print('Cost at theta found by fminunc: %s' % cost)
    print('Expected cost (approx): 0.203')
    print('theta: ')
    print(theta)
    print('Expected theta (approx):')
    print(' -25.161\n 0.206\n 0.201')

    # Plot Boundary
    plotDecisionBoundary(theta, X, y)

    # Put some labels
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()

    print('\nProgram paused. Press enter to continue.')

    # ============== Part 4: Predict and Accuracies ==============
    #  Predict probability for a student with score 45 on exam 1
    #  and score 85 on exam 2

    prob = sigmoid(np.array([1, 45, 85]) @ theta.reshape(-1, 1))
    print(
        'For a student with scores 45 and 85, we predict an admission probability of %f' %
        prob)
    print('Expected value: 0.775 +/- 0.002\n')
    # Compute accuracy on our training set
    p = predict(theta, X)

    print('Train Accuracy: %f' % (np.mean(p == y) * 100))
    print('Expected accuracy (approx): 89.0')
