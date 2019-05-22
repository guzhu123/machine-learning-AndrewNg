#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Machine Learning Online Class - Exercise 1: Linear Regression

x refers to the population size in 10,000s
y refers to the profit in $10,000s

author: zhuzi   version: 1.0    date: 2019/05/15
"""

# COMPUTECOST Compute cost for linear regression
# J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
# parameter for linear regression to fit the data points in X and y

def computeCost(X, y, theta):

    return np.mean((X @ theta - y)**2) / 2


def gradientDescent(X, y, theta, alpha, num_iters):

    # GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha
    m = y.size # number of training examples
    J_history = np.zeros(num_iters)

    for iter in range(num_iters):
        theta -= (alpha * ((X @ theta - y).T @ X).T / m)
        # Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta)

    return (theta, J_history)


if __name__ == '__main__':

    # ======================= Part 1: Plotting =======================
    print('Plotting Data ...\n')
    data = np.loadtxt('ex1data1.txt', delimiter = ',')
    # data[:, 0]得到的shape为(*,)，本质是行向量，所以这里需要转为列向量
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    m = y.size # number of training examples

    # Plot Data
    plt.scatter(X, y, label = 'Training data')
    #plt.show()

    input('Program paused. Press enter to continue.\n')

    # =================== Part 2: Cost and Gradient descent ===================
    X = np.c_[np.ones(m), X] # Add a column of ones to x
    theta = np.zeros([2, 1]) # initialize fitting parameters

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    print('Testing the cost function ...\n')
    # compute and display initial cost
    J = computeCost(X, y, theta)
    print('With theta = [0 ; 0]\nCost computed = %f' % J)
    print('Expected cost value (approx) 32.07\n')

    # further testing of the cost function
    J = computeCost(X, y, np.array([[-1], [2]]))
    print('With theta = [-1 ; 2]\nCost computed = %f' % J)
    print('Expected cost value (approx) 54.24\n')

    input('Program paused. Press enter to continue.\n')
    print('Running Gradient Descent ...')
    # run gradient descent
    (theta, J_history) = gradientDescent(X, y, theta, alpha, iterations)

    # print theta to screen
    print('Theta found by gradient descent:')
    print(theta)
    print('Expected theta values (approx)')
    print(' -3.6303\n  1.1664\n')

    # Plot the linear fit
    plt.plot(X[:, 1], X @ theta, color = 'red', label = 'Linear regression')
    plt.legend(loc = 'upper right')
    plt.show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.array([1, 3.5]) @ theta
    print('For population = 35,000, we predict a profit of ', end = '')
    print(predict1 * 10000)
    predict2 = np.array([1, 7]) @ theta
    print('For population = 70,000, we predict a profit of ', end = '')
    print(predict2 * 10000)

    # ============= Part 3: Visualizing J(theta_0, theta_1) =============
    print('Visualizing J(theta_0, theta_1) ...')

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

    # Fill out J_vals
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
            J_vals[i][j] = computeCost(X, y, t)

    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals.T
    # Surface plot
    fig = plt.figure()
    ax = Axes3D(fig)
    (x_axis, y_axis) = np.meshgrid(theta0_vals, theta1_vals)
    ax.plot_surface(x_axis, y_axis, J_vals)
    ax.set_xlabel('theta_0')
    ax.set_ylabel('theta_1')
    plt.show()

    # Contour plot
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    plt.contour(x_axis, y_axis, J_vals, np.logspace(-2, 3, 20))
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.plot(theta[0], theta[1], 'rx')
    plt.show()
