import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from base_cnn import BaseCNN


class CIFAR10CNN(BaseCNN):
    def __init__(self, gamma=0.1, **kwargs):
        if "additional_transforms" not in kwargs:
            kwargs["additional_transforms"] = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=4),
            ])
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        super(CIFAR10CNN, self).__init__(classes=classes, **kwargs)
        self.gamma = gamma

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