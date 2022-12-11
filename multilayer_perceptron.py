# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: [Anurag Sangem] -- [ansangem]
#
# Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff

import numpy as np
import math
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    def _initialize(self, X, y):

        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X = X
        self._y = one_hot_encoding(y) #using the one hot encoding function to encode categorical features

        #inititalizing weights

        self._h_weights = np.random.randn(self._X.shape[1], self.n_hidden)*0.01
        self._h_bias = np.zeros((1, self.n_hidden))
        self._o_weights = np.random.randn(self.n_hidden, self._y.shape[1])*0.01
        self._o_bias = np.zeros((1,self._y.shape[1]))



    def fit(self, X, y):

        self._initialize(X, y) #initializing the train set , and encode y column


        #updating bias and getting predictions for n_iterations
        for iter in range(self.n_iterations):

            Hidden_layer = np.dot(self._X, self._h_weights) + self._h_bias
            Hidden_activation = self.hidden_activation(Hidden_layer)

            Output_layer = np.dot(Hidden_activation, self._o_weights) + self._o_bias
            Output_actication = self._output_activation(Output_layer)

            #Backward propogation from output later
            error_temp = Output_actication - self._y
            d_weights_op = np.dot(Hidden_activation.T, error_temp)
            d_bias_op = (1 / self._X.shape[0]) * np.sum(error_temp, axis=0, keepdims=True)

            #Backward propogation from hidden later
            d_hidden = np.dot(error_temp, self._o_weights.T) * self.hidden_activation(Hidden_layer, True)
            d_weights_hidden = (1 / self._X.shape[0]) * np.dot(self._X.T, d_hidden)
            d_bias_hidden = (1 / self._X.shape[0]) * np.sum(d_hidden, axis=0, keepdims=True)

            #update the weights and proceed to another iteration

            self._h_weights = self._h_weights - self.learning_rate * d_weights_hidden
            self._o_weights = self._o_weights - self.learning_rate * d_weights_op
            self._h_bias = self._h_bias - self.learning_rate * d_bias_hidden
            self._o_bias = self._o_bias - self.learning_rate * d_bias_op
   

    def predict(self, X):
        self.output=[]

        op_layer1 = np.dot(X, self._h_weights) + self._h_bias
        activation_layer1 = self.hidden_activation(op_layer1)
        op_layer2 = np.dot(activation_layer1, self._o_weights) + self._o_bias
        activation_layer2 = self._output_activation(op_layer2)

        pred_lst = []

        for row in activation_layer2:
            op = np.argmax(row, axis=0)
            pred_lst.append(op)
        self.output = np.array(pred_lst)

        return self.output
