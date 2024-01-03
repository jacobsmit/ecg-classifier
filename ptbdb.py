import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# hyperparameters
NUM_EPOCHS = 1000
HIDDEN1 = 100
HIDDEN2 = 50
LR = 0.001

# generate data from csv files and split into training/testing
print("Preparing data...")
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

# define the class with layers and activation functions
print("Initializing model...")
class PTBDBClassifier(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.hidden1 = nn.Linear(187, HIDDEN1)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(HIDDEN2, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x) :
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

# initalize the model, loss function, and optimizer
model = PTBDBClassifier()
loss_fn = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# train model on training data for NUM_EPOCHS
print("Training model...")
for epoch in range(NUM_EPOCHS) :
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')


