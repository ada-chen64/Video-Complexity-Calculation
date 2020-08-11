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


def performance(data, labels, clips):
    '''
    If test set has labels, calculate acc and rmse

    Print:
        acc = correct / total
        rmse = sqrt(sum((labels - preds)^2))
    '''

    # get preds and labels
    preds = predict_complexity(data)
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
    worse = torch.nonzero(diff > 2)
    for w in worse:
        print('[%3d] %s:  label %d\tpred %d\tdiff %d' % (w, clips[w], labels[w], preds[w], diff[w]))
    print()
    return