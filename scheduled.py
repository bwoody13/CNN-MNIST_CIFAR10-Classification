from torch.optim.lr_scheduler import StepLR
class Scheduled:
    def __init__(self, gamma=0.7, step=3):
        self.gamma = gamma
        self.step = step

    def make_scheduler(self, optimizer):
        return StepLR(optimizer, gamma=self.gamma, step_size=self.step)
