import torch

MNIST_type = "MNIST"
CIFAR10_type = "CIFAR10"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def save_model(model, name, model_type):
    torch.save(model, f"{model_type}/models/{name}.pth")


def save_state_dict(model, name, model_type):
    torch.save(model.state_dict(), f"{model_type}/models/{name}_sd.pth")


def load_model(name, model_type):
    model = torch.load(f"{model_type}/models/{name}.pth",
                       map_location=torch.device(device))
    model.eval()
    return model


def load_state_dict(model, name, model_type):
    state_dict = torch.load(f"{model_type}/models/{name}_sd.pth",
                            map_location=torch.device(device))
    model.load_state_dict(state_dict)
