import torch
import numpy as np
from torch.nn import functional as F

class DomainCNN(torch.nn.Module):
    # input is vector in R^64

    def __init__(self, domains):
        super(DomainCNN, self).__init__()

        # 3 1D convolution layers
        self.conv1 = torch.nn.Conv1d(1, 32, kernel_size=5)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2)
        self.conv2 = torch.nn.Conv1d(32, 16, kernel_size=5, stride=2)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv1d(16, 8, kernel_size=2, stride=2)
        self.pool3 = torch.nn.MaxPool1d(kernel_size=2, stride=1)

        # last layer projects vectors to dimension k
        # average the vectors to obtain a single vector of dim k
        self.fc1 = torch.nn.Linear(8*2, domains)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.pool1(x)
        x = F.elu(self.conv2(x))
        x = self.pool2(x)
        x = F.elu(self.conv3(x))
        x = self.pool3(x)
        # reshape
        x = x.view(-1, 8*2)
        m = torch.nn.Softmax(1)
        x = m(self.fc1(x))
        return x

class DomainLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(DomainLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        """param outputs: k dimension outputs from confusion network """
        # TODO FIX THIS HAHA
        loss = torch.nn.CrossEntropyLoss(outputs, targets)
        loss = loss.sum / loss.shape[1]



