import numpy as np
from sklearn.model_selection import train_test_split
import pdb 
from sklearn.tree import DecisionTreeRegressor
from io import StringIO  
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pydotplus

from sklearn.datasets import fetch_california_housing

import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

def plot_tree(dtree, feature_names):
    """ helper function """
    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    print('exporting tree to dtree.png')
    graph.write_png('dtree.png')


class RegressionStump():
    
    def __init__(self):
        """ The state variables of a stump"""
        
        #The feature we are splitting on
        self.idx = None
        
        #The split value
        self.val = None
        
        #Mean of all objects in left leaf
        self.left = None
        
        #Mean of all objects in left leaf
        self.right = None
    
    def fit(self, data, targets):
        """ Fit a decision stump to data
        
        Find the best way to split the data, minimizing the least squares loss of the tree after the split 
    
        Args:
           data: np.array (n, d)  features
           targets: np.array (n, ) targets
    
        sets self.idx, self.val, self.left, self.right
        """
        # update these three
        self.idx = 0
        self.val = None
        self.left = None
        self.right = None
        
        #Get the number of points (n) and number of features (d)
        num_points, num_features = data.shape
        
        #Print the number of points
        print(num_points)
        
        #Initialize a variable to keep track of the best score
        best_score = None
        
        #Iterate through each data feature, named feat_idx
        for feat_idx in range(num_features):
            #Access the data belonging only to that feature in the data set
            feat = data[:, feat_idx]
            
            #Iterate through each possible split value by taking each value in the
            #index interval (0,n) for that feature
            for split_idx in range(num_points):
                split_value = feat[split_idx]
                
                #If the feature vector is less than the split value then that data point
                #is going to the left. Make two boolean vectors containing this information
                goes_left = feat < split_value
                goes_right = feat >= split_value

                #Why this check?
                if np.sum(goes_left)==0 or np.sum(goes_right)==0:
                    continue
                
                #Compute the mean of left and right side by only expressing
                #the entries from the targets that we want
                left_mean = np.mean(targets[goes_left])
                right_mean = np.mean(targets[goes_right])
                
                #Compute the scores by taking the least squares loss
                left_score = np.sum((targets[goes_left] - left_mean)**2)
                right_score = np.sum((targets[goes_right] - right_mean)**2)
                total_score = left_score + right_score
                
                #If the total score is better than the current best score update the fields of the model
                if best_score == None or total_score < best_score:
                    best_score = total_score
                    self.idx = feat_idx
                    self.val = split_value
                    self.left = left_mean
                    self.right = right_mean
        

    def predict(self, X):
        """ Regression tree prediction algorithm
        Args
            X: np.array, shape n,d
        returns pred: np.array shape n,  model prediction on X
        """
        pred = None
        
        #Retrieve the colum consisting of the dermining attribute
        determining_attribute = X[:, self.idx]
        print("determining attribute is:", determining_attribute)
        
        #Make a vector containing zeroes with rows equal to data set
        pred = np.zeros_like(determining_attribute)
        
        #Compares each value of determining attributes against the split value
        #Returns a decision vector containing 1 if the statement was true, otherwise 0
        decision = (determining_attribute < self.val)
        print("decision is", decision)
        #If decision had an entry of 1, that means we are in the left node
        #Therefore the following works for generating the correct values
        pred = decision * self.left + (1-decision) * self.right
        print("pred is", pred)
        
        ### END CODE
        return pred
    
    def score(self, X, y):
        """ Compute mean least squares loss of the model

        
        Args
            X: np.array, shape n,d
            y: np.array, shape n, 

        returns out: scalar - mean least squares loss.
        """
        out = None
        
        #First make the prediction on the data set returns a n row vector
        prediction = self.predict(X)
        
        #Compute the difference between prediction and true label
        difference = prediction-y
        
        #Compute the MEAN least squares loss and return
        out = (difference**2).mean()      
        return out
        


def main():
    """ Simple method testing """
    housing = fetch_california_housing()
    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(housing.data,
                                                        housing.target,
                                                        test_size=0.2)

    baseline_accuracy = np.mean((y_test-np.mean(y_train))**2)
    print('Least Squares Cost of learning mean of training data:', baseline_accuracy) 
    print('Lets see if we can do better with just one question')
    D = RegressionStump()
    D.fit(X_train, y_train)
    print('idx, val, left, right', D.idx, D.val, D.left, D.right)
    print('Feature name of idx', housing.feature_names[D.idx])
    print('Score of model', D.score(X_test, y_test))
    print('lets compare with sklearn decision tree')
    dc = DecisionTreeRegressor(max_depth=1)
    dc.fit(X_train, y_train)
    dc_score = ((dc.predict(X_test)-y_test)**2).mean()
    print('dc score', dc_score)
    print('feature names - for comparison', list(enumerate(housing.feature_names)))
    plot_tree(dc, housing.feature_names)

if __name__ == '__main__':
    main()