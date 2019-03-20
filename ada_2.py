# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:29:03 2019

@author: hw
"""

import numpy as np

# ada boost

# function to calculate error
def get_err(instance_weights, misclass):
    '''
    instance_weights： 1d array, weight of each instance
    misclassification: 1d array, prediction result of ith classifier, 1 for 
                        misclassification, 0 for correct
    '''
    
    sum_w = np.sum(instance_weights)
    weighted = instance_weights * misclass
    sum_weighted = np.sum(weighted)
    error = sum_weighted / sum_w
    return error

def update_weight(prev_weights, alpha, misclass):
    '''
    prev_weights: 1d array, weights of instances in last round
    alpha: weight of selected classifier
    misclass: misclassification of selected classifier
    '''
    misclass_edit = []
    for i in misclass:
        if i == 1:
            misclass_edit.append(1)
        else:
            misclass_edit.append(-1)
    alpha_y_h = alpha * np.array(misclass_edit)
    exp = np.exp(alpha_y_h)
    d_times_exp = prev_weights * exp
    
    #normalize weights
    sum_w = np.sum(d_times_exp)
    
    return d_times_exp/sum_w

#ignore this
def predict(alpha, misclass, instance_id):
    '''
    alpha: 1d array, weights of classfiers
    misclass: 2d array[classifier_idx, instance_idx]
    instance_idx: instance to be classified
    '''
    classifier_num = len(alpha)
    predict_sum = 0
    for classifier_idx in range(classifier_num):
        classifier_result = misclass[classifier_idx, instance_id]
        if classifier_result == 1:
            classifier_result = -1
        else:
            classifier_result = 1
        predict_sum += classifier_result * alpha[classifier_idx]
    
    return np.sign(predict_sum)

def predict_2(alpha, misclass, classifier_idx, instance_id):
    '''
    alpha: 1d array, weights of classfiers
    misclass: 2d array[classifier_idx, instance_idx]
    classifier_idx: 1d array, selected classifier index
    instance_idx: instance to be classified
    '''

    score = 0
    classifier_num = len(alpha)
    #1 for misclassify, -1 for correct
    binary = misclass.copy()
    binary[np.where(binary == 0)] = -1
    
    for idx in range(classifier_num):
        classifier_result = misclass[classifier_idx[idx], instance_id]
        score += classifier_result*alpha[idx]
    
    # 1 for misclassification, -1 for correct, 0: cannot determine
    return np.sign(score)
    

instance_num = 8
classifier_num = 4

#1 for misclassification
misclassification = np.array([[1,0,0,1,0,1,0,1],
                     [0,0,0,1,1,0,0,0],
                     [0,1,1,0,0,0,1,0],
                     [1,1,0,0,0,1,0,0]
                     ])


# initialize instance weights to 1/instance_num
init_inst_weights = np.ones(instance_num)
init_inst_weights = init_inst_weights /instance_num

################## round 0 ########################
err_0 = np.zeros(classifier_num)

#calculate error of each classifier
#err_0 = get_err(init_inst_weights, misclassification[0])
for i in range(classifier_num):
    err_0[i] = get_err(init_inst_weights, misclassification[i])

# select the classifier with min error
min_err_classifier_0 = np.argmin(err_0)

#calculate alpha, i.e. the weight of selected classifier
alpha_0 = 0.5 * np.log((1 - err_0[min_err_classifier_0]) / err_0[min_err_classifier_0])
print(alpha_0)
#update weights of instances
weight_0 = update_weight(init_inst_weights, alpha_0, misclassification[min_err_classifier_0])
print(weight_0)
###################### round 1 ######################
#calculate error

err_1 = np.zeros(classifier_num)
for i in range(classifier_num):
    err_1[i] = get_err(weight_0, misclassification[i])

min_err_classifier_1 = np.argmin(err_1)

#calculate alpha, i.e. the weight of classifier
alpha_1 = 0.5 * np.log((1 - err_1[min_err_classifier_1]) / err_1[min_err_classifier_1])
print(alpha_1)

#update weights of instances
weight_1 = update_weight(weight_0, alpha_1, misclassification[min_err_classifier_1])
print(weight_1)

##################### predict after 2 rounds #####################

voting_weights = np.array([alpha_0, alpha_1])
classifier_idx = np.array([min_err_classifier_0, min_err_classifier_1])

predict_result = predict_2(voting_weights, misclassification, classifier_idx, 7)
print(predict_result)

