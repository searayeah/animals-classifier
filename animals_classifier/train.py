import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torchvision
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.autonotebook import tqdm, trange


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")

    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment("Animals classifier")

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    preprocess = weights.transforms()

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(in_features=512, out_features=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_dataset = torchvision.datasets.ImageFolder(
        cfg["train_path"], transform=preprocess
    )
    val_dataset = torchvision.datasets.ImageFolder(
        cfg["val_path"], transform=preprocess
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=cfg["num_workers"],
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=cfg["num_workers"],
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    with mlflow.start_run(run_name="resnet18"):
        mlflow.log_params(dict(cfg))

        for epoch in trange(cfg["num_epochs"]):
            model.train()

            losses = []
            for data, target in tqdm(train_dataloader, leave=False):
                data = data.to(device)
                target = target.unsqueeze(1).float().to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            tqdm.write(f"Train loss: {np.mean(losses)}")
            mlflow.log_metric("Train loss", np.mean(losses), step=epoch)

            model.eval()
            val_losses = []
            correct = 0
            total = 0

            with torch.inference_mode():
                for data, target in tqdm(val_dataloader, leave=False):
                    data = data.to(device)
                    target = target.unsqueeze(1).float().to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_losses.append(loss.item())

                    predicted = (torch.sigmoid(output) > 0.5).float()
                    correct += (predicted == target).sum().item()
                    total += target.size(0)

            avg_val_loss = np.mean(val_losses)
            val_accuracy = correct / total

            tqdm.write(f"Validation Loss: {avg_val_loss} val acc: {val_accuracy}")
            mlflow.log_metric("Validation Loss", avg_val_loss, step=epoch)
            mlflow.log_metric("Validation Accuracy", val_accuracy, step=epoch)

    torch.save(model.state_dict(), "models/resnet18.pt")


if __name__ == "__main__":
    train()
