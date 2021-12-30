import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, models, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.classifier = nn.Sequential()

        num_fcs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_fcs, 256),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model.forward(x)