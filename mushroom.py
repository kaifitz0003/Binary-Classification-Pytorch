"""
This is a binary classification problem that uses the mushroom dataset to predict whether a mushroom is poisonous or not.

The data is from kaggle https://www.kaggle.com/datasets/prishasawhney/mushroom-dataset. It has 54035 rows and 9 columns.

We used a neural network with 1 hidden layer to create the model.
The input layer has 8 neurons, the hidden layer has 2 neurons and the output layer has 1 neuron.
The model is trained for 250 epochs.
BCELoss made the algorithm learn better than MSELoss.

On a Nvidia T4 GPU on Google Colab it takes around 1 minute per epoch. 250 epochs got an accuracy of around 75%

TODO:
Improve our results by doing hyperparameter tuning/optimization.
This means we are checking how many layers, how many neurons per layer, the learning rate and the number of epochs produce  the best results.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

df = pd.read_csv('mushroom_cleaned.csv')
X = df.drop('class', axis = 1).values
y = df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
model = nn.Sequential(nn.Linear(8,2),nn.Sigmoid(),nn.Linear(2,1),nn.Sigmoid())
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
for epoch in range(250):
  for i in range(len(X_train)):
    out = model(X_train[i])
    loss = loss_fn(out[0], y_train[i])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  print(epoch)
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_pred = model(X_test)
y_pred
accuracy_score(y_test, y_pred.detach().numpy().round())
