import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.model_selection import train_test_split


# random.seed(2016010165)
# np.random.seed(2016010165)


# parameters
batch_size = 16
epochs = 200
lr = 0.0015
weight_decay = 0.0030


# load data
data = np.load('data/train_data.npy')
norm = np.load('data/norm.npy')
labels = data[:, 0]
inputs = data[:, 1:]
inputs = (inputs - norm[1]) / (norm[0] - norm[1])

# split train and valid
train_inputs, valid_inputs, train_labels, valid_labels \
    = train_test_split(inputs, labels, test_size=0.10)

# data loader
train_set = Data.TensorDataset(torch.FloatTensor(train_inputs), torch.LongTensor(train_labels))
valid_set = Data.TensorDataset(torch.FloatTensor(valid_inputs), torch.LongTensor(valid_labels))
train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = Data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.fc(x)
        return x


def train(net, train_loader, valid_loader):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    max_acc = 0
    min_rmse = 10

    net.train()
    for epoch in range(epochs):
        begin = time.time()
        running_loss = 0.0
        for index, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_acc, train_rmse = validate(net, train_loader)
        valid_acc, valid_rmse = validate(net, valid_loader)
        if valid_acc > max_acc and valid_rmse < min_rmse:
            max_acc = valid_acc
            min_rmse = valid_rmse
            torch.save(net.state_dict(),'model/net.pkl')
        end = time.time()
        print('[%3d] loss: %.4f' % (epoch + 1, running_loss / len(train_loader.dataset)), end='\t')
        print('train acc: %.4f mrse: %.4f' % (train_acc, train_rmse), end='\t')
        print('valid acc: %.4f mrse: %.4f' % (valid_acc, valid_rmse), end='\t')
        print('max acc: %.4f min mrse: %.4f' % (max_acc, min_rmse), end='\t')
        print('')
    print('finish training')


def validate(net, loader, detail=False):
    mse = 0
    correct = 0
    total = 0
    labels_all = []
    preds_all = []

    net.eval()
    for data in loader:
        inputs, labels = data
        outputs = net(inputs)
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        mse += ((preds - labels) ** 2.0).sum().item()
        if detail:
            labels_all += labels.numpy().tolist()
            preds_all += preds.numpy().tolist()
    acc = correct / total
    rmse = np.sqrt(mse / total)
    if detail:
        labels_all = np.array(labels_all)
        preds_all = np.array(preds_all)
        upper = np.max(np.abs(labels_all - preds_all))
        print(labels_all.tolist())
        print(preds_all.tolist())
        print('acc: %.4f\trmse: %.4f\tmax: %d' % (acc, rmse, upper))
        torch.save(net.state_dict(),'model/net_%.4f_%.4f_%d.pkl' % (acc, rmse, upper))
    return acc, rmse


net = MLP()
train(net, train_loader, valid_loader)
best_net = MLP()
best_net.load_state_dict(torch.load('model/net.pkl'))
validate(best_net, valid_loader, True)
