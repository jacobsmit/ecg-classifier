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

print("Initalizing model...")
class MITBIHClassifier(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.hidden1 = nn.Linear(187, 100)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(100, 50)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(50, 5)

    def forward(self, x) :
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.out(x)
        return x
    
model = MITBIHClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

