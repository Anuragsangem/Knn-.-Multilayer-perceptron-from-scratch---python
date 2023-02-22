
#Report
Q1)

1)Abstractions : 
a)distance_list - stores the distances of all the train set data from the test set data.
b)neighbors - stores the k nearest neighbors in the test set 
c)all the other absatractions are given in the initialize function in the class

Ideology:
Knn-
a)By finding the K data points in the training set that are closest to the new data point, and using those K data points to make a prediction. 
For example, in a binary classification task, KNN might predict that a new data point belongs to the "positive" class 
if the majority of the K nearest neighbors belong to the "positive" class.
b)Knn algorithm is also called as Lazy learner because there is no training that happens in the algorithm , it just calculates the distance 
of a test set point to all the train set data points during the testing.


Metrics : 
a)euclidean_distance : It is calculated as the square root of the sum of the squares of the differences between the x-coordinates 
and y-coordinates of the two points.
b)manhattan_distance :The Manhattan distance is also known as the "taxi cab" or "rectilinear" distance, 
because it is the distance a taxi would have to travel in a city laid out in a grid pattern to get from one point to another. 
It is different from the Euclidean distance, which is the straight-line distance between two points


Q2)

#Report:
a)Abstractions : 
1)_h_weights - Weights of the hidden layer ->I Initialized the hidden weights 
2)_h_bias - Bias of the hidden layer
3)_o_weights - Weights of the output layer
4)_o_bias - Bias of the output layer
5)The remaining absatractions arem given in comments

b)Ideology : 
A multilayer perceptron (MLP) is a type of feedforward artificial neural network. 
It consists of an input layer, one or more hidden layers, and an output layer. 
Each layer consists of a set of neurons, which are connected to the neurons in the previous and next layers. 
The input layer receives the input data, which is then passed through the hidden layers where it is transformed by a series of computations.
Activation Function : Each layer is passed through a activation function which processes the In artificial neural networks,
the activation function is a key component of a neuron. It takes in the weighted sum of all the inputs to the neuron, 
along with a bias term, and produces an output. 
We Defined activation functions like sigmoid , tanh , identity , ReLu etc in utiils.py file.

c)Work Flow:
1)We Initialize weights , bias.
2)Fit :In the fit method we run each row of the train set and pass it through hidden layer , output layer , we calculate the error of the actual y value 
and the predicted y value.
3)Backpropogation : We use this error to back propogate the layers and to update weights.
4)In the same way we update weighs for every train set record and update weights through all the iterations
5)Predict :We use the final values of the weights and bias values to run through the test set data and calculate the pedictions.


