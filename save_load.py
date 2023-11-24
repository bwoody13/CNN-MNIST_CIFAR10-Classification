import torch

MNIST_type = "MNIST"
CIFAR10_type = "CIFAR10"


def save_model(model, name, model_type):
    torch.save(model, f"{model_type}/models/{name}.pth")


def load_model(name, model_type):
    model = torch.load(f"{model_type}/models/{name}.pth")
    model.eval()
    return model
