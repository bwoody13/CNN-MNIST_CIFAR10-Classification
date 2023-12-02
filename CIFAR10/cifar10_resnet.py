import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from base_cnn import BaseCNN
from scheduled import Scheduled


# Option A shortcut connection
class ShortcutConnection(nn.Module):
    def __init__(self, channels, stride):
        super(ShortcutConnection, self).__init__()
        self.channels = channels
        self.stride = stride

    def forward(self, x):
        return F.pad(x[:, :, ::self.stride, ::self.stride],
                     (0, 0, 0, 0, self.channels // 4, self.channels // 4),
                     "constant", 0)


class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutConnection(out_channels, stride)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFAR10ResNet(BaseCNN, Scheduled):
    def __init__(self, num_blocks, gamma=0.1, epochs=120, **kwargs):
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        BaseCNN.__init__(self, classes=classes, epochs=epochs, **kwargs)
        step = int(self.epochs * 0.4)
        Scheduled.__init__(self, gamma=gamma, step=step)

        if "additional_transforms" not in kwargs:
            self.additional_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
            ])

        self.in_channels = 16
        self.dataset_size = 50000

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 10)

    def _make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def make_optimizer(self):
        return optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True
        )

