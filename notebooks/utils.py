#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################
# PROGRAMMER: Pierre-Antoine Ksinant                  #
# DATE CREATED: 22/07/2019                            #
# REVISED DATE: -                                     #
# PURPOSE: General framework for modeling the problem #
#######################################################


##################
# Needed imports #
##################

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


############################################
# Create datasets for training and testing #
############################################

def create_training_testing_datasets(consolidated_dataset):
    """ Create shuffled and stratified datasets """
    
    # Separate target from features in the dataset:
    target = consolidated_dataset['country_destination']
    features = consolidated_dataset.drop('country_destination', axis=1)
    
    # Extract features:
    extracted_features = features.values
    
    # Encode target variable:
    le = LabelEncoder()
    le_countries = le.fit(target)
    encoding_dict = dict(zip(list(le_countries.classes_),
                             range(len(list(le_countries.classes_)))))
    encoded_target = le_countries.transform(target)
    
    # Shuffle, stratify and split the data into training and testing datasets:
    X_train, X_test, y_train, y_test = train_test_split(extracted_features,
                                                        encoded_target,
                                                        test_size=0.2,
                                                        stratify=encoded_target,
                                                        random_state=42)
    
    # Return results:
    return X_train, X_test, y_train, y_test, encoding_dict


###################################
# Calculate Normalized DCG metric #
###################################

def calculate_dcg(predictions, country_destination, rank=5):
    """ Calcultate Discounted Cumulative Gain """
    
    # Initialize DCG score and iteration variable:
    dcg_score = 0.
    i = 0
    
    # Determine largest rank to consider:
    max_rank = np.minimum(rank, len(predictions))
    
    # Calculate DCG score:
    while i < max_rank:
        rel = int(predictions[i] == country_destination)
        i += 1
        dcg_score += (np.power(2., rel) - 1.)/np.log2(i + 1.)
    
    # Return result:
    return dcg_score

def calculate_ndcg(predictions, country_destination, rank=5):
    """ Calculate Normalized DCG """
    
    # Determine DCG score:
    dcg_score = calculate_dcg(predictions, country_destination, rank)
    
    # Determine ideal DCG score (in project particular context):
    idcg_score = 1.
    
    # Calculate nDCG score:
    ndcg_score = dcg_score/idcg_score
    
    # Return result:
    return ndcg_score


###########################################
# Calculate prediction on one sample data #
###########################################

def clf_prediction(clf, sample_data):
    """ Perform predictions thanks to classifier """
    
    # Perform predictions:
    preds_prob = clf.predict_proba(sample_data.reshape(1, -1)).tolist()[0]
    
    # Build predictive ordered list of first booking destination country:
    preds_list = [x[1] for x in sorted(zip(preds_prob, range(12)), reverse=True) if x[0] != 0.]
    
    # Return result:
    return preds_list


##################################################
# Calculate nDCG mean score on a labeled dataset #
##################################################

def ndcg_mean_score_calc(clf, X, y):
    """ Calculate nDCG mean score on a labeled dataset """
    
    # Set nDCG scores list:
    ndcg_scores_list = []
    
    # Loop on labeled dataset:
    for i in range(len(y)):
        ndcg_score = calculate_ndcg(clf_prediction(clf, X[i]), y[i])
        ndcg_scores_list.append(ndcg_score)
        
    # Determine nDCG mean score:
    ndcg_mean_score = np.mean(ndcg_scores_list)
    
    # Return result:
    return ndcg_mean_score

def detailed_ndcg_mean_score_calc(clf, X, y, encoding_dict):
    """ Calculate nDCG mean score on a labeled dataset for each class """
    
    # Reverse encoding dictionary:
    decoding_dict = dict(map(reversed, encoding_dict.items()))
    
    # Set nDCG scores objects:
    ndcg_scores_dict = {country_dest: [] for country_dest in range(12)}
    ndcg_mean_scores_list = []
    
    # Loop on labeled dataset:
    for i in range(len(y)):
        ndcg_score = calculate_ndcg(clf_prediction(clf, X[i]), y[i])
        ndcg_scores_dict[y[i]].append(ndcg_score)
        
    # Loop on country destinations:
    for country_dest in range(12):
        ndcg_mean_scores_list.append(np.mean(ndcg_scores_dict[country_dest]))
        
    # Return result:
    return ndcg_mean_scores_list