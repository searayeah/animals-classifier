import torchvision
from torch.utils.data import DataLoader


def get_train_data(path, preprocess, batch_size, num_workers):
    train_dataset = torchvision.datasets.ImageFolder(path, transform=preprocess)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_dataset, train_dataloader


def get_val_data(path, preprocess, batch_size, num_workers):
    val_dataset = torchvision.datasets.ImageFolder(path, transform=preprocess)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return val_dataset, val_dataloader


def get_test_data(path, preprocess, batch_size, num_workers):
    test_dataset = torchvision.datasets.ImageFolder(path, transform=preprocess)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return test_dataset, test_dataloader
