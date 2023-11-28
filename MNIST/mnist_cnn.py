import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from base_cnn import BaseCNN


class MNISTCNN(BaseCNN):
    def __init__(self, **kwargs):
        if "additional_transforms" not in kwargs:
            kwargs["additional_transforms"] = transforms.Compose([
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.7, 1.1)),
            ])
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        super(MNISTCNN, self).__init__(classes=classes, **kwargs)

        # Layers
        # self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        # self.pool1 = nn.MaxPool2d(2, 2)
        # self.pool2 = nn.MaxPool2d(2, 2)
        # self.dropout1 = nn.Dropout(0.15)
        # self.dropout2 = nn.Dropout(0.35)
        # self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # self.fc2 = nn.Linear(128, 10)
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def make_optimizer(self):
        return optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )

    def do_additional_transforms(self, inputs):
        if self.additional_transforms:
            return self.additional_transforms(inputs)
        else:
            return input
