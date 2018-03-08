# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:43:34 2017

@author: Trisaratops
"""
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import math

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.plot(X[:,i],y,'.')
        plt.xlabel(features[i])
        plt.ylabel("Cost")
    
    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    #X_transpose = X.T 
    a = np.matmul(X.T,X)
    b = np.matmul(X.T,Y)
    # Remember to use np.linalg.solve instead of inverting!
    #XXt = np.matmul(X,X_transpose)
    #XtY = np.matmul(X_transpose,Y)
    w= np.linalg.solve(a,b)
    return w
    #raise NotImplementedError()
    
def seperate_data(X,y):
        #TODO: Split data into train and test
    trainI = np.random.choice(len(X), math.floor(0.8*len(X)), replace = False)
    
    #seperate into train and test data
    trainX = X[trainI]
    trainY = y[trainI]
    testI = []
    for i in range(len(y)):
        if i not in trainI:
            testI.append(i)
    testX = X[testI]
    testY = y[testI] 
    
    return trainX,trainY,testX,testY

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)
    #add bias to x
    X = np.concatenate((np.ones((506,1)),X),axis=1)
 
    trainX,trainY,testX,testY = seperate_data(X,y)
     
      
    # Fit regression model
    w = fit_regression(trainX, trainY)
    print("Bias = {}".format(w[0]))
    for i in range(len(features)):
        print("{}  = {}".format(features[i],w[i+1]))
    # Compute fitted values, MSE, etc.
    yhat = np.matmul(testX,w)
    
    #MSE
    summation = 0
    for j in range(len(testY)):
        summation += (yhat[j] - testY[j])**2
        
    MSE = (1/len(testY))*summation
    print("Mean Squared Error:")
    print(MSE)
    
    #calcualte mean average error
    mean_avg_error = MAE(testY,yhat)
    print(mean_avg_error)
    
    #calculate the R2 error
    R_2 = R2(testY,yhat)
    print(R_2)
    
    
def MAE(y,y_hat):
    """
    Return the mean absolute error
    """
    
    N = len(y)
    MAE = (1/N)*np.sum(np.abs(y-y_hat))
    return MAE

def R2(y,y_hat):
    """
    Calculate the R^2 error
    """
    
    avg_y = np.mean(y)
    total_ss = np.sum((y-avg_y)**2)
    resid_ss = np.sum((y-y_hat)**2)
    R_2 = 1- (resid_ss/total_ss)
    return R_2
    
    
if __name__ == "__main__":
    main()
    

