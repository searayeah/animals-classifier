import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from dataset import get_train_data, get_val_data
from model import get_model
from omegaconf import DictConfig
from torch import optim
from tqdm.autonotebook import tqdm, trange


def train_epoch(model, train_dataloader, device, optimizer, criterion):
    model.train()

    train_losses = []
    for data, target in tqdm(train_dataloader, leave=False):
        data = data.to(device)
        target = target.unsqueeze(1).float().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
    avg_train_loss = np.mean(train_losses)
    tqdm.write(f"Train loss: {avg_train_loss}")
    return avg_train_loss


def eval_epoch(model, val_dataloader, device, criterion):
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
    return avg_val_loss, val_accuracy


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")

    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment("Animals classifier")

    model, preprocess = get_model(cfg["model_name"])

    if cfg["device"].startswith("cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    model.to(device)

    train_dataset, train_dataloader = get_train_data(
        cfg["train_path"], preprocess, cfg["batch_size"], cfg["num_workers"]
    )
    val_dataset, val_dataloader = get_val_data(
        cfg["val_path"], preprocess, cfg["batch_size"], cfg["num_workers"]
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    with mlflow.start_run(run_name="resnet18"):
        mlflow.log_params(dict(cfg))

        for epoch in trange(cfg["num_epochs"]):
            avg_train_loss = train_epoch(
                model, train_dataloader, device, optimizer, criterion
            )
            mlflow.log_metric("Train loss", avg_train_loss, step=epoch)

            avg_val_loss, val_accuracy = eval_epoch(
                model, val_dataloader, device, criterion
            )

            mlflow.log_metric("Validation Loss", avg_val_loss, step=epoch)
            mlflow.log_metric("Validation Accuracy", val_accuracy, step=epoch)

    torch.save(model.state_dict(), "models/resnet18.pt")


if __name__ == "__main__":
    train()
