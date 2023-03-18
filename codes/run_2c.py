# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 21:58:39 2023

@author: biraaj
"""

import matplotlib.pyplot as plt
import numpy as np
from functions import loadTrain_data_1,weighted_regression_fit,compute_weights,mse

data_path = "../data/data_train_1b_c_2.txt"
data_path_test = "../data/data_test_1c.txt"

linear_feature_array_with_bias, _input_features,_targets = loadTrain_data_1(data_path,0,0)
test_linear_feature_array_with_bias, test_input_features,test_targets = loadTrain_data_1(data_path_test,0,0)
gamma = 0.124

sorted_input_features_test = np.argsort(test_input_features)
predicted_values_train = []
predicted_values_test = []

# training target predictions
for _index,_data in enumerate(_input_features):
    x = _data
    calculated_Weights = compute_weights(_input_features,x,gamma)
    
    _coeff = weighted_regression_fit(linear_feature_array_with_bias,_targets,calculated_Weights)
    predicted_values_train.append(np.dot(_coeff,linear_feature_array_with_bias[_index]))
predicted_values_train = np.array(predicted_values_train)

# testing target predictions
for _index,_data in enumerate(test_input_features):
    x = _data
    calculated_Weights = compute_weights(_input_features,x,gamma)
    
    _coeff = weighted_regression_fit(linear_feature_array_with_bias,_targets,calculated_Weights)
    predicted_values_test.append(np.dot(_coeff,test_linear_feature_array_with_bias[_index]))
predicted_values_test = np.array(predicted_values_test)

print("MSE for training data = "+str(mse(predicted_values_train,_targets)))
print("MSE for test data = "+str(mse(predicted_values_test,test_targets)))

plt.plot(test_input_features[sorted_input_features_test], predicted_values_test[sorted_input_features_test], '-c')
plt.plot(test_input_features[sorted_input_features_test], test_targets[sorted_input_features_test], '+r')
plt.xlabel('input')
plt.ylabel('targets')
plt.legend(['bestfit','original target'])
plt.title("fit of weighted linear regrssion with test data")
plt.savefig('../results/2_c_weighted_regression_with_test_data.jpeg',dpi=200)
plt.show()
plt.clf()