# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [Anurag Sangem] -- [ansangem]
#
# Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff



#Reference : 
# 1)K-Nearest Neighbors (KNN) – Theoryhttp://www.datasciencelovers.com › machine-learning
# 2)https://www.youtube.com/watch?v=ngLyX54e1LU&list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E&ab_channel=PatrickLoeber

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance



    def fit(self, X, y):
        #since knn uses the train data only while testing , there is no computation requried in fit method , so im just saving the data X,y to _X,_y which is my X_train and y_train
        self._X=X  
        self._y=y

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        #raise NotImplementedError('This function must be implemented by the student.')
        y_pred=[]
        for x in X:
            distance_list=[] #we reinitialise this for every row in test set
            neighbors=[]
            for train_data_row in self._X:
                distance_list.append(self._distance(x,train_data_row))
            sorted_distances_index=np.argsort(distance_list)[:self.n_neighbors] #this returns the index of the neighbors in the train set
            
            #we take the labels of the nearest neighbors in the train set to take a majority vote
            for i in sorted_distances_index:
                neighbors.append(self._y[i])

            pred_value_for_row_test=max([(neighbors.count(p_class),p_class) for p_class in set(neighbors)])[1]
            #pred_value_for_row_test = Counter(neighbors).most_common(1)
            y_pred.append(pred_value_for_row_test)

        #y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)