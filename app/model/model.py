import torch.nn as nn
from torchvision import models

def get_model():
    model = models.efficientnet_b0(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 4)
    return model
