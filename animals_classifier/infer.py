import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.autonotebook import tqdm


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def infer(cfg: DictConfig):
    model = resnet18()
    model.fc = nn.Linear(in_features=512, out_features=1)
    model.load_state_dict(torch.load("models/resnet18.pt"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    test_dataset = torchvision.datasets.ImageFolder(
        cfg["train_path"], transform=preprocess
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=cfg["num_workers"],
    )

    model.eval()
    correct = 0
    total = 0
    answers = []

    with torch.inference_mode():
        for data, target in tqdm(test_dataloader, leave=False):
            data = data.to(device)
            target = target.unsqueeze(1).float().to(device)
            output = model(data)

            predicted = (torch.sigmoid(output) > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
            answers.append(predicted.cpu().tolist())

    test_accuracy = correct / total

    tqdm.write(f"Test accuracy: {test_accuracy}")

    df = pd.DataFrame(test_dataset.imgs, columns=["image_path", "true_label"])

    df["predicted"] = np.concatenate(answers)
    df.to_csv("pred.csv", index=False)


if __name__ == "__main__":
    infer()
