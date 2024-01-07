import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def get_model(model_name):
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)

        preprocess = weights.transforms()

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(in_features=512, out_features=1)

        return model, preprocess
    else:
        raise Exception("Model not found")
