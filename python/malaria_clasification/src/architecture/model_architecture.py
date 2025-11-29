import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes=2):
    #model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = models.resnet18(weights= None)
    
    # Replace the fully connected layer to fit our number of classes
    num_ftrs = model.fc.in_features # 512 size vector representing the image in the form of learned numerical features.
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
