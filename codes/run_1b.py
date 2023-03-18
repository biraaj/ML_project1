# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 21:08:59 2023

@author: biraaj
"""

import matplotlib.pyplot as plt
import numpy as np
from functions import loadTrain_data_1,linear_regression_fit,linear_regression_predict
data_path = "../data/data_train_1b_c_2.txt"
d = 6
k=9
for _d in range(d+1):
    modified_linear_array_features, _input_features,_targets = loadTrain_data_1(data_path,k,_d)
    _coefficients = linear_regression_fit(modified_linear_array_features,_targets)
    predictions = linear_regression_predict(modified_linear_array_features,_coefficients)
    sorted_input_features = np.argsort(_input_features)    
    
    plt.plot(_input_features[sorted_input_features], predictions[sorted_input_features], '-c')
    plt.plot(_input_features[sorted_input_features], _targets[sorted_input_features], '+r')
    plt.xlabel('input')
    plt.ylabel('targets')
    plt.legend([ 'bestfit','original target'])
    plt.title('targets and best fit with depth = ' + str(_d))
    plt.savefig('../results/1_b_bestfit_depth_' + str(_d) + '.jpeg',dpi=200)
    plt.show()
    plt.clf()