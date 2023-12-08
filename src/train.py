import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader

# from torchvision.datasets import ImageFolder
from torchvision.models import ResNet152_Weights, resnet152
from tqdm.autonotebook import tqdm


def train():
    train_path = "data/train_11k"
    val_path = "data/val"
    # test_path = "data/test_labeled"
    # test_path = "data/test_labeled"

    batch_size = 256
    # shuffle = False
    # drop_last = False
    num_workers = 4
    learning_rate = 0.00001
    num_epochs = 20

    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)

    preprocess = weights.transforms()

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(in_features=2048, out_features=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_dataset = torchvision.datasets.ImageFolder(train_path, transform=preprocess)
    val_dataset = torchvision.datasets.ImageFolder(val_path, transform=preprocess)
    # test_dataset = torchvision.datasets.ImageFolder(test_path, transform=preprocess)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=num_workers,
    # )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for _epoch in range(num_epochs):
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

    torch.save(model.state_dict(), "models/resnet152.pt")


if __name__ == "__main__":
    train()
