import torch.nn as nn
from torchvision import models

def get_resnet18_for_cl(num_classes, pretrained=True):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def get_resnet34_for_cl(num_classes, pretrained=True):
    if pretrained:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet34(weights=None)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def get_resnet50_for_cl(num_classes, pretrained=True):
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=None)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model