import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class MyCNN(nn.Module):
    def __init__(
            self,
            epochs=10,
            learning_rate=0.01,
            batch_size=64,
            weight_decay=0,
            momentum=0,
            gamma=0.1,
            device=None
    ):
        super(MyCNN, self).__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gamma = gamma
        self.device = device if device != None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.criterion = nn.CrossEntropyLoss()

        # Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # output = F.log_softmax(x, dim=1)
        return self.output(x)

    def make_optimizer(self):
        return optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=True,
        )

    def make_schedule(self, optimizer):
        return StepLR(optimizer, step_size=self.epochs // 4, gamma=self.gamma)
