import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    def __init__(
            self,
            epochs=10,
            learning_rate=0.01,
            batch_size=64,
            weight_decay=0,
            momentum=0,
            device=None
    ):
        super(MNISTCNN, self).__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.criterion = nn.CrossEntropyLoss()
        self.device = device if not device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.35)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def make_optimizer(self):
        return optim.SGD(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum
        )
