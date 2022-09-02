from .cnn import Net
import torch.nn as nn

from .models import LeNetComplete
from .resnet import ResNet


class model_factory():
    def __init__(self, parse):
        self.model = self.model_complete(parse)
        self.split_layer = parse['split_layer']

    def model_complete(self, parse):
        model_name = parse['model']

        if model_name == "LeNet":
            return LeNetComplete()
        elif model_name == "ResNet56":
            return ResNet()

    def create(self):
        return nn.Sequential(*nn.ModuleList(self.model.children())[:self.split_layer]), \
               nn.Sequential(*nn.ModuleList(self.model.children())[self.split_layer:])
