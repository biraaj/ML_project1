# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 23:54:13 2023

@author: biraaj
"""

import numpy as np
import math

def replace_parenthesis(_string):
    """
        Function to filter out data and split by removing parenthesis.
        Note: This implmentation was from hw1
    """
    return _string.replace('(','').replace(')','').replace(' ','').strip().split(",")

def loadTrain_data_1(train_data_path,k,d):
    """
        This functions takes in the training data along with the frequency k and depth d.
        It returns the feature matrix with nonlinear features, targets and the input feature array.
    """
    _feat = []
    _targ = []
    linear_feature_array = None
    with open(train_data_path) as _file:
        for _row in _file:
            temp_data = replace_parenthesis(_row)
            # mapping is done below to convert floating points and string to their respective format from a all string text.
            _feat.append(float(temp_data[0]))
            _targ.append(float(temp_data[1]))
    
    #This is to check if depth is not equal to 0 so that we can compute the non linear parameters.
    if(d!=0):
        non_linear_features = 2*d+2
        
        linear_feature_array = np.zeros((len(_feat),non_linear_features))
        
        linear_feature_array[:,0] = 1
        linear_feature_array[:,1] = np.array(_feat)
        index = 2
        for _d in range(1,d+1):
            third_term_non_linear_values = []
            fourth_term_non_linear_values = []
            for _val in _feat:
                third_term_non_linear_values.append((math.sin(_d*k*_val)**(_d*k))*math.cos(_val))
                fourth_term_non_linear_values.append((math.cos(_d*k*_val)**(_d*k))*math.sin(_val))
            linear_feature_array[:,index] = np.array(third_term_non_linear_values)
            linear_feature_array[:,index+1] = np.array(fourth_term_non_linear_values)
            index+=2
    else:
        linear_feature_array = np.zeros((len(_feat),2))
        linear_feature_array[:,0] = 1
        linear_feature_array[:,1] = np.array(_feat)
                
    return linear_feature_array, np.array(_feat), np.array(_targ)

##### Linear Regression Functions

def linear_regression_fit(features,target):
    """
        This function provides the coefficients with respect to analytic optimization
    """
    _tarnsposed_feat = features.transpose()
    squared_feature = np.dot(_tarnsposed_feat,features)
    assert (np.linalg.det(squared_feature) != 0.0), 'Coeff is not possible as the squared feature matrix is singular matrix'
    _coeff = np.dot(np.dot(np.linalg.inv(squared_feature),_tarnsposed_feat),target)
    
    return _coeff

def linear_regression_predict(features,coefficient):
    """
        This function provides the predictions for each input.
    """
    return np.array(np.dot(features,coefficient))

def mse(_predict,actual):
    """
        This function calculates the mse of two arrays having same shape
    """
    return np.square(np.subtract(_predict,actual)).mean()

##### Locally Weighted Linear Regression Functions

def compute_weights(features,x,gamma):
    """
        This function computes the weight for each input as per the formula given in the project pdf
    """
    _weights = []
    for _index,_feat in enumerate(features):
        _weights.append(np.exp(-(np.square(_feat-x))/(2*np.square(gamma))))
    return np.array(_weights)
        

def weighted_regression_fit(features,target,weights):
    """
        This function computes the coefficients after multiplying the weights to the analytic optimization for linear regression to get weighted coefficients.
    """
    transposed_feature_with_weight = np.dot(features.transpose(),np.diag(weights))
    squared_feature = np.dot(transposed_feature_with_weight,features)
    _coeff = np.dot(np.linalg.inv(squared_feature),np.dot(transposed_feature_with_weight,target))
    return _coeff


##### Softmax Regression


def compute_softmax(_weighted_feature_array,_axis=1):
    """
        This function computes the softmax probablities as per the softmax formula involving exponenets.
    """
    exponent_weighted_feature_array = np.exp(_weighted_feature_array - np.max(_weighted_feature_array,axis=_axis,keepdims=True)) 
    return exponent_weighted_feature_array/np.sum(exponent_weighted_feature_array,axis=_axis,keepdims=True)

def load_data_softmax_train(train_data_path,remove_from_end=0):
    """
        This is a function to load the training data for classification problems required for softmax regression.
        This returns the feature array with bias and target array by hot encoding it.
    """
    _feat = []
    _target = []
    with open(train_data_path) as _file:
        for _row in _file:
            temp_data = replace_parenthesis(_row)
            # mapping is done below to convert floating points and string to their respective format from a all string text.
            _feat.append([float(i) for i in temp_data[:(len(temp_data)-1-remove_from_end)]])
            _target.append(temp_data[len(temp_data)-1])

    unique_labels = list(set(_target))
    target_array = np.array(_target)
    single_hot_array = np.zeros((len(_target),len(unique_labels)))
    for _index,_val in enumerate(unique_labels):
        single_hot_array[:,_index] =  (target_array == _val).astype(int)
        
    #adding bias term to feature array
    _feat = np.insert(np.array(_feat),0,1,axis=1) 
    
    
    return _feat,single_hot_array,unique_labels

def load_data_softmax_test(test_data_path,remove_from_end=0):
    """
        This function provides the test data features for softmax regression
    """
    _feat = []
    with open(test_data_path) as _file:
        for _row in _file:
            temp_data = replace_parenthesis(_row)
            # mapping is done below to convert floating points and string to their respective format from a all string text.
            _feat.append([float(i) for i in temp_data[:(len(temp_data)-remove_from_end)]])
        
    #adding bias term to feature array
    _feat = np.insert(np.array(_feat),0,1,axis=1) 
    
    
    return _feat

def softmax_predict(feature_array,weights,_axis=1):
    #predicting probbality for classes when input is given
    feature_dot_weight = np.dot(feature_array,weights)
    return np.argmax(compute_softmax(feature_dot_weight,_axis=_axis),axis=_axis)
    


            
    
