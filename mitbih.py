import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# number of cycles through all training data
NUM_EPOCHS = 1000

# process data into training and test; X and y
print("Preparing data...")
train_data = np.genfromtxt("mitbih_train.csv", delimiter=",")
X_train = torch.tensor(train_data[:,:-1], dtype=torch.float32)
y_train = torch.tensor(train_data[:,-1], dtype=torch.long)

test_data = np.genfromtxt("mitbih_test.csv", delimiter=",")
X_test = torch.tensor(test_data[:,:-1], dtype=torch.float32)
y_test = torch.tensor(test_data[:,-1], dtype=torch.long)

# define the class with layers and activation functions
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
    
# initalize the model, loss function, and optimizer
model = MITBIHClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train model on training data for NUM_EPOCHS
print("Training model...")
for epoch in range(NUM_EPOCHS) :
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

# determine accuracy via test data
with torch.no_grad :
    model.eval()
    y_pred = model(X_test)
    model.train()

y_pred = torch.argmax(y_pred, dim=1)
accuracy = (y_pred == y_test).float().mean()
print(f"Test Accuracy: {accuracy_test}")