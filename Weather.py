"""
This is a binary classification problem that uses a weather dataset to predict 
the humidity and tempature to determine if the weather will be good or bad.

The data is made using a time series. 

I generated the data over all 366 days of the leap year 2024. It records humidity and tempature every 3 hours.
 
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

# Data
readings = 8
half_year = int(366/2)
days_year = 366

temp = np.concatenate([np.random.rand(half_year*readings)*10+70,np.random.rand(half_year*readings)*10+90]) # Good & Bad Tempature
humidity = np.concatenate([np.random.rand(half_year*readings)*10+30,np.random.rand(half_year*readings)*10+60]) # Good & Bad Humidity

weather = np.concatenate([np.ones(half_year*readings,dtype = np.int32),np.zeros(half_year*readings,dtype = np.int32)]) # Good & Bad Weather
idx = pd.date_range("2024-01-01T12:00AM", periods = readings*days_year, freq='3h')

df = pd.DataFrame(range(readings*days_year), index = idx, columns = ['Tempature'])
df['Humidity'] = humidity # Good Tempature
df['Tempature'] = temp # Bad Tempature
df['Weather'] = weather
X = df.drop('Weather', axis = 1).values
y = df['Weather'].values
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Algo


model = nn.Sequential(nn.Linear(2,2),nn.Sigmoid(),nn.Linear(2,1),nn.Sigmoid())

# Training
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
for epoch in range(10):
  for i in range(len(X_train)):
    out = model(X_train[i])
    loss = loss_fn(out[0], y_train[i])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  print(epoch)

# Testing/Prediction
y_pred = model(X_test)
y_pred
accuracy_score(y_test, y_pred.detach().numpy().round())
