import torch
import torch.nn as nn

# from .utils import create_fc
from torchvision import models

# Script to return all pretrained models in torchvision.models module
def resnet18(pretrained, out_dim, remove_last_layer=True):
    encoder = models.__dict__['resnet18'](pretrained = pretrained)
    encoder.fc = nn.Identity()

    return encoder