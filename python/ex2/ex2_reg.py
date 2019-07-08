#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Learning Online Class - Exercise 2: Logistic Regression

Author: Six     Date: 2019/07/02    Version: 1.0
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

    plt.scatter(X[pos, 0], X[pos, 1], marker='+', label='y=1')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', label='y=0')


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
            a = np.power(X1, i - j) * np.power(X2, j)
            out = np.c_[out, np.power(X1, i - j) * np.power(X2, j)]

    return out


def sigmoid(z):
    # SIGMOID Compute sigmoid function
    # g = SIGMOID(z) computes the sigmoid of z.

    return 1 / (1 + np.exp(-z))


def costFunctionReg(theta, X, y, reg_lambda):
    # COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    # J = COSTFUNCTIONREG(theta, X, y, reg_lambda) computes the cost of using
    # theta as the parameter for regularized logistic regression and the
    # gradient of the cost w.r.t. to the parameters.

    # Initialize some useful values
    m = np.size(y) # number of training examples

    h = sigmoid(X @ theta.reshape(-1, 1))
    J = np.mean(-y*np.log(h) - (1-y)*np.log(1-h)) + reg_lambda/(2*m)*np.sum(theta**2)

    return J


def Gradient(theta, X, y, reg_lambda):

    m = np.size(y) # number of training examples
    h = sigmoid(X @ theta.reshape(-1, 1))
    grad = ((h - y).T @ X).T/m + reg_lambda/m*theta.reshape(-1, 1)
    # 更新theta0的梯度，因为X的第一列都是1,所以没有写入公式
    grad[0] = np.mean(h - y)

    return grad


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
        # Notice you need to specify the levels [0]，因为theta*X=0是决策边界
        C = plt.contour(u, v, z, levels=[0])
        plt.legend([C.collections[0]], ['Decision Boundary'])


def predict(theta, X):
    # PREDICT Predict whether the label is 0 or 1 using learned logistic
    # regression parameters theta
    # p = PREDICT(theta, X) computes the predictions for X using a
    # threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    # 直接四舍五入
    p = np.round(sigmoid(X @ theta.reshape(-1, 1)))
    return p


if __name__ == '__main__':

    # Load Data
    # The first two columns contains the X values and the third column
    # contains the label (y).
    data = np.loadtxt('ex2data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2].reshape(-1, 1)

    plotData(X, y)

    # Put some labels
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(loc=1)
    plt.show()

    # =========== Part 1: Regularized Logistic Regression ============
    # Add Polynomial Features
    X = mapFeature(X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1))

    # Initialize fitting parameters
    initial_theta = np.zeros(np.size(X, 1))

    # Set regularization parameter reg_lambda to 1
    reg_lambda = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost = costFunctionReg(initial_theta, X, y, reg_lambda)
    grad = Gradient(initial_theta, X, y, reg_lambda)

    print('Cost at initial theta (zeros): %f' % cost)
    print('Expected cost (approx): 0.693')
    print('Gradient at initial theta (zeros) - first five values only:')
    # print时不使用科学计数法
    np.set_printoptions(suppress=True)
    print(grad[0:5])
    print('Expected gradients (approx) - first five values only:')
    print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115')

    print('\nProgram paused. Press enter to continue.')

    # Compute and display cost and gradient
    # with all-ones theta and reg_lambda = 10
    test_theta = np.ones(np.size(X, 1))
    cost = costFunctionReg(test_theta, X, y, 10)
    grad = Gradient(test_theta, X, y, 10)

    print('\nCost at test theta (with reg_lambda = 10): %f' % cost)
    print('Expected cost (approx): 3.16')
    print('Gradient at test theta - first five values only:')
    print(grad[0:5])
    print('Expected gradients (approx) - first five values only:')
    print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922')

    print('\nProgram paused. Press enter to continue.')
    # ============= Part 2: Regularization and Accuracies =============
    # Initialize fitting parameters
    initial_theta = np.zeros(np.size(X, 1))

    # Set regularization parameter reg_lambda to 1 (you should vary this)
    # 强烈建议多试几次，并保存图像进行对比（0即不进行正则化）
    reg_lambda = 1

    result = op.minimize(
        fun=costFunctionReg,
        x0=initial_theta,
        args=(
            X,
            y,
            reg_lambda),
        method='TNC',
        jac=Gradient)
    cost = result.fun
    theta = result.x

    # Plot Boundary
    plotDecisionBoundary(theta, X, y)
    plt.title('reg_lambda = %s' % reg_lambda)

    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.show()

    # Compute accuracy on our training set
    p = predict(theta, X)

    print('Train Accuracy: %f' % (np.mean(p == y) * 100))
    print('Expected accuracy (with reg_lambda = 1): 83.1 (approx)')
