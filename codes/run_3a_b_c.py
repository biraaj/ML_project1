# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:47:36 2023

@author: biraaj
"""

from functions import load_data_softmax_train,compute_softmax,softmax_predict,load_data_softmax_test
import numpy as np
train_data_path = "../data/train_data_2b_hw1.txt"
test_data_path = "../data/test_data_2b_hw1.txt"
program_data_main = "../data/program_data_2c_2d_2e_hw1.txt"
alpha = 0.01
epoch = 10000
#implementing leave out one method
def softmax_regression(train_features,train_targets,weights):
    for _epoch in range(epoch):
        feature_mul_weight = np.dot(train_features,weights)
        train_probab_pred =  compute_softmax(feature_mul_weight)
        
        #Calculating gradient ascent
        _error = train_targets-train_probab_pred
        change_in_weight = alpha*np.dot(train_features.transpose(),_error)
        weights = weights+change_in_weight
    return weights
    
            
    
def leave_out_one_feature(train_features,train_targets,unique_classes,no_of_features=4):
    values_predicted = []
    for _index,_element in enumerate(train_features):
        value_to_be_predicted = train_features[_index]
        target_to_be_used = train_targets[_index]
        #Deleting one row from train and target arrays
        new_train_features = np.delete(train_features,_index,0)
        new_train_target = np.delete(train_targets,_index,0)
        
        #Intiliazing weights to 0
        weights = np.zeros((train_features.shape[1],len(unique_classes)))
        
        weights = softmax_regression(new_train_features,new_train_target,weights)
        
        predictions = softmax_predict(value_to_be_predicted,weights,_axis=0)
        values_predicted.append(target_to_be_used[predictions])

    print("Prediction accuracy with "+str(no_of_features)+" features=",(sum(values_predicted)/len(values_predicted))*100,"%")
    

## 3a Implementing softmax regression to classify the test dataset
train_features,train_targets,unique_classes = load_data_softmax_train(train_data_path,0)
test_features = load_data_softmax_test(test_data_path,0)
weights = np.zeros((train_features.shape[1],len(unique_classes)))
weights = softmax_regression(train_features,train_targets,weights)
predictions = softmax_predict(test_features,weights)
print("Testing the softmax regression function...")
print("Predictions on test data",predictions)
print("####################################")
print()
print()

print("Implementing leave out one method with 4 features")
## 3b Implementing the leave out one method with 4 features
train_features,train_targets,unique_classes = load_data_softmax_train(program_data_main,0)
leave_out_one_feature(train_features,train_targets,unique_classes,4)
print("####################################")
print()
print()


print("Implementing leave out one method with 3 features")
## 3c implementing the leave out one method with 3 features
train_features,train_targets,unique_classes = load_data_softmax_train(program_data_main,1)
leave_out_one_feature(train_features,train_targets,unique_classes,3)
print("####################################")