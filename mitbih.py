import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# hyperparameter
NUM_EPOCHS = 5000
HIDDEN1 = 100
HIDDEN2 = 50
LR = 0.001

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
        self.hidden1 = nn.Linear(187, HIDDEN1)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(HIDDEN2, 5)

    def forward(self, x) :
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.out(x)
        return x
    
# initalize the model, loss function, and optimizer
model = MITBIHClassifier()
loss_fn = nn.CrossEntropyLoss()
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

# determine accuracy via test data
with torch.no_grad() :
    model.eval()
    y_pred = model(X_test)
    model.train()

print(classification_report(y_test.numpy(), y_pred.numpy()))

current_index = 0
conclusion = {0: "Normal beat", 1: "Supraventricular premature beat", 2: "Premature ventricular contraction", 3: "Fusion of ventricular and normal beat", 4: "Unclassifiable beat"}

# create a figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# plot the initial data as a line graph
line, = ax.plot(X_test[current_index], label=f'Array {current_index + 1}')
ax.set_title(f'Predicted: {conclusion[y_pred[0].item()]}, Actual: {conclusion[y_test[0].item()]}')

# add a slider for scrolling through the arrays
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Array Index', 0, len(X_test) - 1, valinit=current_index, valstep=1)

# function to update the plot based on the slider value
def update(val) :
    index = int(slider.val)
    line.set_ydata(X_test[index])
    line.set_label(f'Array {index + 1}')
    ax.set_title(f'Predicted: {conclusion[y_pred[index].item()]}, Actual: {conclusion[y_test[index].item()]}')
    ax.legend()
    fig.canvas.draw_idle()

# connect the slider to the update function
slider.on_changed(update)

plt.show()
