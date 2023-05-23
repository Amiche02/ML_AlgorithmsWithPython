# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Chapter 1 : pecetron algorithm ---> 1 book
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(object):
    """Perceptron classifier.
    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    random_state : int
    Random number generator seed for random weight
    initialization.
    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    errors_ : list
    Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, x, y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples is the number of
        examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
        Target values.
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+x.shape[1])
        
        self.errors_ = []
        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(x,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, x):
        """calculate net input"""
        return np.dot(x, self.w_[1:] + self.w_[0])
    
    def predict(self, x):
        """Return class label after unit step"""
        return np.where(self.net_input(x) >= 0.0, 1, -1)



class AdalineGD(object):
        """ADAptive LInear NEuron classifier.
        Parameters
        ------------
        eta : float
        Learning rate (between 0.0 and 1.0)
        n_iter : int
        Passes over the training dataset.
        random_state : int
        Random number generator seed for random weight initialization.
        Attributes
        -----------
        w_ : 1d-array
        Weights after fitting.
        cost_ : list
        Sum-of-squares cost function value in each epoch.
        """
        
        def __init__(self, eta=0.01, n_iter=50, random_state=1):
            self.eta = eta
            self.n_iter = n_iter
            self.random_state = random_state
            
        def fit(self, x, y):
            """ Fit training data.
            Parameters
            ----------
            X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples
            is the number of examples and
            n_features is the number of features.
            y : array-like, shape = [n_examples]
            Target values.
            Returns
            -------
            self : object
            """
            rgen = np.random.RandomState(self.random_state)
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+x.shape[1])
            self.cost_ = []
            
            for i in range(self.n_iter):
                net_input = self.net_input(x)
                output = self.activation(net_input)
                errors = (y-output)
                self.w_[1:] += self.eta * x.T.dot(errors)
                self.w_[0] += self.eta * errors.sum()
                
                cost = (errors**2).sum()/2.0
                self.cost_.append(cost)
            return self
        
        def net_input(self, x):
            return np.dot(x, self.w_[1:]) + self.w_[0]
        
        def activation(self, x):
            return x
        
        def predict(self, x):
            return np.where(self.activation(self.net_input(x)) >= 0.0, 1, -1)



class AdalineSGB(object):
    """ADAptive LInear NEuron classifier.
        Parameters
        ------------
        eta : float
        Learning rate (between 0.0 and 1.0)
        n_iter : int
        Passes over the training dataset.
        shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent
        cycles.
        random_state : int
        Random number generator seed for random weight
        initialization.
        Attributes
        -----------
        w_ : 1d-array
        Weights after fitting.
        cost_ : list
        Sum-of-squares cost function value averaged over all
        training examples in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, x, y):
        """ Fit training data.
            Parameters
            ----------
            X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.
            y : array-like, shape = [n_examples]
            Target values.
            Returns
            -------
            self : object
        """
        self._initialize_weights(x.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                x, y = self._shuffle(x, y)
            cost = []
            for xi, target in zip(x, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, x, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initiali1ze_weights(x.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(x, y)
        return self
    
    def _shuffle(self, x, y):
        """shuffle the training data"""
        r = self.rgen.permutation(len(y))
        return x[r], y[r]
    
    def _initialize_weights(self, m):
        """Initilize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, x):
        """calculate net input"""
        return np.dot(x, self.w_[1:] + self.w_[0])
    
    def activation(self, x):
        """compute linear activation"""
        return x
    
    def predict(self, x):
        return np.where(self.activation(self.net_input(x))>=0.0, 1, -1)
    

#Decision  boundery plot
def plot_decision_regions(x, y, classifier, resolution=0.02):
    #stetup marker generator and colormap
    markers = ("s", "x", "o", '^', 'v')
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #plot the decision surface
    x1_min , x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min , x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), x2_max.max())
    
    #plot class examples
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x = x[y == c1, 0], y = x[y == c1, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=c1, edgecolor='black')
        