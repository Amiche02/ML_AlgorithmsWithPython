import numpy as np
from matplotlib.colors import ListedColormap 
import matplotlib.pyplot as plt 

class LogisticRegressionGD(object):
    """Logistic Regression Classifier using gradient descent.
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
    cost_ : list
    Logistic cost function value in each epoch.
    """
    
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """ Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples is the number of
        examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
        Target values
        Returns
        -------
        self : object
        """
        
        # Initialize the weights randomly
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        # Create an empty list to store the cost at each epoch
        self.cost_ = []
        
        # Loop through the specified number of iterations
        for i in range(self.n_iter):
            # Compute the net input (weighted sum) of the inputs
            # and the current weights
            net_input = self.net_input(X)
            
            # Apply the logistic sigmoid activation function
            # to obtain the output (a value between 0 and 1)
            output = self.activation(net_input)
            
            # Compute the error between the true class labels (y)
            # and the predicted probabilities (output)
            errors = (y - output)
            
            # Update the weights using the gradient descent rule
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            # Compute the logistic 'cost' function value
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            
            # Append the cost to the list of cost values
            self.cost_.append(cost)
        
        # Return the trained model
        return self
    
    def net_input(self, X):
        """Calculate the net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)





def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02): 
    # setup marker generator and color map 
    markers = ('s', 'x', 'o', '^', 'v') 
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') 
    cmap = ListedColormap(colors[:len(np.unique(y))]) 
    
    # plot the decision surface 
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)) 
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) 
    Z = Z.reshape(xx1.shape) 
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap) 
    plt.xlim(xx1.min(), xx1.max()) 
    plt.ylim(xx2.min(), xx2.max()) 
    for idx, cl in enumerate(np.unique(y)): 
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black') 
    # highlight test examples 
    if test_idx: 
        # plot all examples 
        X_test, y_test = X[test_idx, :], y[test_idx] 
        plt.scatter(X_test[:, 0], X_test[:, 1], c='green', edgecolor='black', alpha=0.3, linewidth=1, marker='o', s=100, label='test set') 
