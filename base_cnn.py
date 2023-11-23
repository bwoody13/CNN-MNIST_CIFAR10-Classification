import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms

class BaseCNN(nn.Module):
    def __init__(
            self,
            epochs=10,
            learning_rate=0.01,
            batch_size=64,
            weight_decay=0,
            momentum=0,
            device=None,
            additional_transforms=None,
            criterion=nn.CrossEntropyLoss()
            classes=None
    ):
        super(BaseCNN, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = device if not device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.additional_transforms = additional_transforms
        self.criterion = criterion
        self.classes = classes

    def make_optimizer(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def do_additional_transforms(self, inputs):
        return self.additional_transforms(inputs) if self.additional_transforms else inputs