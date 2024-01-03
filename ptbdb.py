import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# constants
NUM_EPOCHS = 5000
HIDDEN1 = 100
HIDDEN2 = 50

# generate data from csv files and split into training/testing
print("preparing data...")
normal_data = np.genfromtxt("ptbdb_normal.csv", delimiter=",")
abnormal_data = np.genfromtxt("ptbdb_abnormal.csv", delimiter=",")

data = np.concatenate((normal_data, abnormal_data))
X = data[:,:-1]
y = data[:,-1].reshape(-1, 1)

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.15, random_state=104, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)