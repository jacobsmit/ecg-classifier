import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("Preparing data...")
train_data = np.genfromtxt("mitbih_train.csv", delimiter=",")
X_train = torch.tensor(train_data[:,:-1], dtype=torch.float32)
y_train = torch.tensor(train_data[:,-1], dtype=torch.long)

test_data = np.genfromtxt("mitbih_test.csv", delimiter=",")
X_test = torch.tensor(test_data[:,:-1], dtype=torch.float32)
y_test = torch.tensor(test_data[:,-1], dtype=torch.long)