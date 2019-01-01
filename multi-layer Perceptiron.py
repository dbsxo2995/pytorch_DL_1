import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

#step1. data road
wine = load_wine()
train_X, test_X, train_Y, test_Y = train_test_split(wine.data[0:178], wine.target[0:178], test_size=0.3, random_state= 123 )

#step2. numpy to Tensor
train_X = torch.from_numpy(train_X).float(); train_Y = torch.from_numpy(train_Y).long()
test_X = torch.from_numpy(test_X).float(); test_Y = torch.from_numpy(test_Y).long()
print(train_X.shape)
train = torch.utils.data.TensorDataset(train_X, train_Y)
train_loader = torch.utils.data.DataLoader(train, batch_size= 32, shuffle= True)

#step3. Make Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 124)
        self.fc2 = nn.Linear(124, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.nn.functional.log_softmax(x, dim = 1)
    #parameter of log_sfotmax
    #if dim = 0, calculates softmax across the rows, so each column sums to 1
    #if dim = 1, calculates softmax across the colmns, so each row sums to 1

model = Net()


#step4. train model
criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.05)

for epoch in range(500):
    total_loss = 0
    for train_x, train_y in train_loader:
        train_x, train_y = Variable(train_X), Variable(train_Y)

        optimizer.zero_grad()

        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()

        optimizer.step()

        total_loss += loss.data

    if (epoch + 1) % 50 ==0:
       print(epoch + 1, total_loss)

#step5. model test
test_x, test_y = Variable(test_X), Variable(test_Y)
result = torch.max(model(test_X).data, dim = 1)[1]

acc = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())
print(acc)


#Note. Autograd
