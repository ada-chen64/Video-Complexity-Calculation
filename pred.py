import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data


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


def predict_complexity(data):
    '''
    Param:
        data: features
    Return:
        preds: complexity from 0 to 9
    '''

    # network
    net = MLP()
    net.load_state_dict(torch.load('model/net.pkl'))

    # prediction
    net.eval()
    inputs = torch.FloatTensor(data)
    outputs = net(inputs)
    _, preds = torch.max(outputs.data, 1)

    return preds


def performance(preds, labels, clips):
    '''
    If test set has labels, calculate acc and rmse

    Print:
        acc = correct / total
        rmse = sqrt(sum((labels - preds)^2))
    '''

    # get preds and labels
    preds = torch.squeeze(preds)
    lables = torch.squeeze(labels)
    print(preds)
    print(labels)

    # statistics
    total = labels.size(0)
    diff = (preds - labels).abs()
    acc0 = (diff == 0).sum().item() / total
    acc1 = (diff <= 1).sum().item() / total
    acc2 = (diff <= 2).sum().item() / total
    mse = (diff ** 2.0).sum().item()
    rmse = np.sqrt(mse / total)
    worst = diff.max().item()
    print('acc ±0: %.4f\tacc ±1: %.4f\tacc ±2: %.4f\nrmse: %.4f\tworst: %d'
        % (acc0, acc1, acc2, rmse, worst))
    print()

    print('Bad samples:')
    worse = torch.nonzero(diff > 1)
    for w in worse:
        print('[%3d] %s:\tlabel %d\tpred %d\tdiff %d' % (w, clips[w], labels[w], preds[w], diff[w]))
    print()
    return


if __name__ == "__main__":
    preds = torch.Tensor([3, 2, 3, 3, 7, 2, 4, 7, 1, 3, 4, 2, 0, 0, 0, 1, 3, 0, 1, 0, 0, 0, 1, 1,
        2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 3, 1, 5, 0, 0, 9, 0, 0, 1, 9, 3,
        1, 1, 0, 1, 1, 2, 2, 3, 2, 0, 2, 1, 1, 1, 6, 9, 2, 1, 2, 2, 9, 2, 1, 1,
        2, 0, 3, 3, 5, 7, 3, 6, 6, 3, 4, 2])
    labels = torch.Tensor([3, 2, 3, 3, 4, 2, 5, 5, 1, 3, 3, 2, 1, 1, 1, 1, 3, 1, 2, 0, 0, 0,
       1, 0, 3, 2, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 4, 1, 4, 0, 0, 7, 0,
       0, 1, 9, 2, 2, 1, 0, 1, 1, 2, 2, 3, 2, 1, 3, 2, 1, 2, 5, 9, 2, 2,
       2, 2, 7, 2, 1, 2, 2, 2, 5, 5, 5, 6, 4, 5, 6, 4, 3, 3])
    clips = np.loadtxt('clips.txt', dtype=np.str)
    performance(preds, labels, clips)
