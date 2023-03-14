import json
import matplotlib.pyplot as plt
import os
import math
import random
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
import numpy as np
import time
save_path = './Outputs/Latency/' +  time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) # used for storing trained model
# resume_path = save_path # used for predictor
is_train=False
resume_path = '/home/ryguan/TileSparsity_TVM/DataProcess/Outputs/Latency/2023-03-14-05_28_06'
def convert_keys_to_int(d):
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            new_dict[int(key)] = convert_keys_to_int(value)
        else:

            
            new_dict[int(key)] = value
    return new_dict

with open('/home/ryguan/TileSparsity_TVM/AutoSchedule/Outputs/latency_lookup_table.json', 'r') as f:
    granularity_sparisty_latency_dict = convert_keys_to_int(json.load(f))

granularity_sparsity_latency = []
for granularity, spar2lat in granularity_sparisty_latency_dict.items():
    for sparsity, [pack_and_gemm, scatter] in spar2lat.items():
        totalExecuteTime = pack_and_gemm+scatter

        granularity_sparsity_latency.append([granularity, sparsity, totalExecuteTime])
granularity_sparsity_latency = np.array(granularity_sparsity_latency)

train_data, test_data = train_test_split(granularity_sparsity_latency, test_size=0.2, random_state=42)

train_data = TabularDataset(train_data, columns=['granularity', 'sparsity', 'latency'])
test_data = TabularDataset(test_data, columns=['granularity', 'sparsity', 'latency'])

label = 'latency'
if is_train:
    print("Summary of class variable: \n", train_data[label].describe())
    predictor = TabularPredictor(label=label, path=save_path).fit(train_data)

y_test = test_data[label]  # values to predict
test_data_nolab = test_data.drop(columns=[label])  # delete label column to prove we're not cheating

predictor = TabularPredictor.load(resume_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file

y_pred = predictor.predict(test_data_nolab)
print("Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)


print(predictor.leaderboard(test_data, silent=True))