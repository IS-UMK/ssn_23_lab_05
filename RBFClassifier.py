from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.base import BaseEstimator

def gauss(x, centers, sigma=1.0):

    betas =  1.0 / (2.0 * sigma**2)
    diffs = x - centers
    dist = np.sum(diffs**2, axis=1)
    z = np.exp(-betas * dist)

    return z

class RBFClassifier(BaseEstimator):
    
    def __init__(self, n_hidden=10, sigma=1):
        
        self.sigma = sigma
        self.n_hidden = n_hidden   

    def init_centers(self, X, y):
        # ustaw pozycje centrow, macierz o kształscie [ n_hidden, X.shape[1]]

        return self

    def init_sigmas(self, X, y):
        # ustaw rozycia dla funkcji radialnych, wektor o długości [ n_hidden ]

        return self

    def update_weights(self, X, y):
        # ustaw macierz wag W i wektor wyrazów wolbych b dla warstwy wyjściowej
        # macierz W o kształcie [n_hidden, n_outputs], wektor b o długości [n_outputs]

        return self

    def hidden_activation(self, X):
        # zwraca macierz aktywacji warstwy ukrytej, macierz o kształcie [X.shape[0], n_hidden]

        return 

    def output_activation(self, X):
        # zwraca macierz aktywacji warstwy wyjściowj, macierz o kształcie [X.shape[0], n_outputs]

        return 

    def fit(self, X, y):
        self.init_centers(X, y)
        self.init_sigmas(X, y)
        self.update_weights(X, y)
        return self

    def predict(self, X):
        return self.output_activation(X).argmax(axis=1)

    def score(self, X, y):
        return (self.predict(X) == y).mean()
