# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 00:35:59 2023

@author: biraaj
"""

import matplotlib.pyplot as plt
import numpy as np
from functions import loadTrain_data_1,linear_regression_fit,linear_regression_predict,mse
data_path_train = "../data/data_train_1b_c_2.txt"
data_path_test = "../data/data_test_1c.txt"
d = 6
k=6
for _d in range(d+1):
    modified_linear_array_features, _input_features,_targets = loadTrain_data_1(data_path_train,k,_d)
    modified_linear_array_features_test, _input_features_test,_targets_test = loadTrain_data_1(data_path_test,k,_d)
    _coefficients = linear_regression_fit(modified_linear_array_features,_targets)
    predictions_train = linear_regression_predict(modified_linear_array_features,_coefficients)
    predictions_test = linear_regression_predict(modified_linear_array_features_test,_coefficients)
    sorted_input_features_test = np.argsort(_input_features_test)
    
    print("MSE for training data with depth "+str(_d)+" = "+str(mse(predictions_train,_targets)))
    print("MSE for test data with depth "+str(_d)+" = "+str(mse(predictions_test,_targets_test)))
    
    
    plt.plot(_input_features_test[sorted_input_features_test], predictions_test[sorted_input_features_test], '-c')
    plt.plot(_input_features_test[sorted_input_features_test], _targets_test[sorted_input_features_test], '+r')
    plt.xlabel('input')
    plt.ylabel('targets')
    plt.legend(['bestfit','original target'])
    plt.title('test targets and best fit with depth = ' + str(_d))
    plt.savefig('../results/1_c_bestfit_test_depth_' + str(_d) + '.jpeg',dpi=200)
    plt.show()
    plt.clf()