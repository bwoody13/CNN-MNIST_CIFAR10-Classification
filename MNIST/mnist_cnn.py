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
                transforms.RandomRotation(degrees=15)
            ])
        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        super(MNISTCNN, self).__init__(classes=classes, **kwargs)

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

    def do_additional_transforms(self, inputs):
        if self.additional_transforms:
            return self.additional_transforms(inputs)
        else:
            return input