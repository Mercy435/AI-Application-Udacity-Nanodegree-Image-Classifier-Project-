"""
functions and classes relating to the model 
"""

import torch
import torch.nn.functional as F
import json
import utilities
import matplotlib.pyplot as plt

from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict


def pretrain(network):
    # Load a pre-trained network

    if network == "resnet152":
        pretrained_model = models.resnet152(pretrained=True)
    elif network == "vgg16":
        pretrained_model = models.vgg16(pretrained=True)
    else:
        raise Exception("Architecture not supported!")
    return pretrained_model


def build_model(arch, hidden_layers, output_size=102):
    '''
    Define network architecture, build our model using the loaded pre-trained model
    '''
    # Load a pre-trained model
    model = pretrain(arch)

    # Freeze our parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    if arch == "resnet152":
        classifier = nn.Sequential(nn.Linear(2048, hidden_layers),
                                   nn.ReLU(),
                                   nn.Linear(hidden_layers, output_size),
                                   nn.LogSoftmax(dim=1))

        model.fc = classifier

    elif arch == "vgg16":
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_layers)),
            ('relu1', nn.ReLU()),
            ('drop_out1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(hidden_layers, output_size)),
            ('output', nn.LogSoftmax(dim=1))]))

        model.classifier = classifier

    return model


# function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    # Load the checkpoint
    checkpoint = torch.load(filepath)

    # Load the pre-trained model and rebuild our model
    model = pretrain(checkpoint['architecture'])

    if checkpoint["architecture"] == "resnet152":
        model.fc = checkpoint['classifier']
    elif checkpoint["architecture"] == "vgg16":
        model.classifier = checkpoint['classifier']

    model.load_state_dict(checkpoint['state_dict'])

    print('Checkpoint loaded successfully!')

    return model, checkpoint
