import hydra
import numpy as np
import pandas as pd
import torch
from dataset import get_test_data
from model import get_model
from omegaconf import DictConfig
from tqdm.autonotebook import tqdm


def predict(model, test_dataloader, device):
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
    return answers


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def infer(cfg: DictConfig):
    model, preprocess = get_model(cfg["model_name"])

    if cfg["device"].startswith("cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    model.load_state_dict(torch.load("models/resnet18.pt", map_location=device))
    model.to(device)

    test_dataset, test_dataloader = get_test_data(
        cfg["test_path"], preprocess, cfg["batch_size"], cfg["num_workers"]
    )

    answers = predict(model, test_dataloader, device)

    answers_df = pd.DataFrame(test_dataset.imgs, columns=["image_path", "true_label"])

    answers_df["predicted"] = np.concatenate(answers)
    answers_df.to_csv("pred.csv", index=False)


if __name__ == "__main__":
    infer()
