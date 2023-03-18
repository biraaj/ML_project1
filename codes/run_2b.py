# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 03:29:08 2023

@author: biraaj
"""

import matplotlib.pyplot as plt
import numpy as np
from functions import loadTrain_data_1,weighted_regression_fit,compute_weights

data_path = "../data/data_train_1b_c_2.txt"

linear_feature_array_with_bias, _input_features,_targets = loadTrain_data_1(data_path,0,0)
gamma = 0.124

sorted_features = np.argsort(_input_features)
predicted_values = []

for _index,_data in enumerate(_input_features):
    x = _data
    calculated_Weights = compute_weights(_input_features,x,gamma)
    
    _coeff = weighted_regression_fit(linear_feature_array_with_bias,_targets,calculated_Weights)
    predicted_values.append(np.dot(_coeff,linear_feature_array_with_bias[_index]))
predicted_values = np.array(predicted_values)

plt.plot(_input_features[sorted_features], predicted_values[sorted_features], '-c')
plt.plot(_input_features[sorted_features], _targets[sorted_features], '+r')
plt.xlabel('input')
plt.ylabel('targets')
plt.legend([ 'weighted_regression','original target'])
plt.title('weighted linear regression with original data')
plt.savefig('../results/2_b_locally_weighted_regression.jpeg',dpi=200)
plt.show()
plt.clf()
    
 

