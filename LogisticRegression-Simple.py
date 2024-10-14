"""
Logistic Regression, but used for classification.

This is a binary classifiaction problem because it only has 2 classes, 0 and 1.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = np.array([[0],[1],[2],[3],[10],[11],[12]], dtype = np.float32)
y = np.array([[0],[0],[0],[0],[1],[1],[1]], dtype = np.float32)

X = torch.from_numpy(X)
y = torch.from_numpy(y)
X_train,X_test,y_train,y_test = train_test_split(X,y)
model = nn.Sequential(nn.Linear(1,1), nn.Sigmoid())
loss_fn = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = torch.optim.SGD(params=model.parameters(),lr = 1e-2)
for epoch in range (200):
  for i in range (len(X_train)):
    out = model(X_train[i])
    loss = loss_fn(out, y_train[i])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

y_pred = model(X_test)
print(y_test)
print(y_pred)

accuracy_score(y_test, y_pred.detach().numpy().round())
