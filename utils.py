# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff

import numpy as np
import math


def euclidean_distance(x1, x2):
    euclid_dist=0
    for i in range(len(x1)):
        euclid_dist+=(x1[i]-x2[i])**2

    return math.sqrt(euclid_dist)
    #euclid_dist = np.linalg.norm(x2 - x1) #We can implement this manually but I want to test out the inbuilt operationsin numpy
"""
Computes and returns the Euclidean distance between two vectors.

Args:
    x1: A numpy array of shape (n_features,).
    x2: A numpy array of shape (n_features,).
"""

    #raise NotImplementedError('This function must be implemented by the student.')
def manhattan_distance(x1, x2): #taking the absoulte value of difference bw elements at each poistion in the array

        manhattan_distance = 0
        for a, b in zip(x1, x2):
            diff = b - a
            absolute_difference = abs(diff)
            manhattan_distance += absolute_difference

        return manhattan_distance

    
"""
        Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
"""

def identity(x, derivative = False):
    return x

    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

def sigmoid(x, derivative = False):

    #sigmoid function : 1/(1+e^-x)
    return 1/(1 + np.exp(-x))
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """


def tanh(x, derivative = False):
    return np.tanh(x)
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """


def relu(x, derivative = False):
    #relu function returns the number if the number>0 , it returns 0 for negative numbers 
    #this activation function solves the vanishing gradient issue in sigmoid function
    return np.maximum(0, x)

    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

def softmax(x, derivative = False):
    # e = np.exp(x)
    # return e / e.sum()
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    loss=-np.sum(y*np.log(p))
    return loss/float(p.shape[0])  # we divide it with number of samples to normalise the loss
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """

#reffered to this smart way of implementing OH encoding , I really liked it hence using it here
#https://www.educative.io/answers/how-to-convert-an-array-of-indices-to-one-hot-encoded-numpy-array

def one_hot_encoding(y):
    encoded_array = np.zeros((y.size, y.max()+1), dtype=int)

    #replacing 0 with a 1 at the index of the original array
    encoded_array[np.arange(y.size),y] = 1 


    return encoded_array
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """