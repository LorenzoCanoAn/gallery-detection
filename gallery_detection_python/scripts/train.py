import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import os
import json
from pprint import pprint
import numpy as np
from tqdm import tqdm
from gallery_detection_models.models import GalleryDetectorV3
import neptune
from neptune.types import File

PARAMETERS = {
    "dataset_folder_path": "/media/lorenzo/SAM500/datasets/temp_dataset",
    "n_samples": None,
    "batch_size": 32,
    "n_epochs": 1024,
    "lr": 0.00002,
    "lr_decay": 0.999,
}


class GalleryDetectionDataset(Dataset):
    def __init__(self, index, n_desired_samples=None):
        self.index = index
        self.get_total_datapoints()
        if n_desired_samples is None:
            self.n_desired_samples = self.n_available_samples
        else:
            if self.n_available_samples > n_desired_samples:
                self.n_desired_samples = n_desired_samples
            else:
                self.n_desired_samples = self.n_available_samples
        self.set_n_samples_per_world()
        self.load()

    def get_total_datapoints(self):
        data = self.index["data"]
        self.n_available_samples = 0
        for world_name in data.keys():
            self.n_available_samples += data[world_name]["n_datapoints"]

    def set_n_samples_per_world(self):
        self.n_samples_per_world = {}
        for world_name in self.index["data"].keys():
            n_samples_in_world = self.index["data"][world_name]["n_datapoints"]
            self.n_samples_per_world[world_name] = int(
                np.round(n_samples_in_world * self.n_desired_samples / self.n_available_samples)
            )
        self.final_n_datapoints = sum(
            self.n_samples_per_world[k] for k in self.n_samples_per_world.keys()
        )
        print(self.n_samples_per_world)

    def load(self):
        print("Allocating memory")
        self.images = np.zeros((self.final_n_datapoints, 1, 16, self.index["info"]["image_width"]))
        self.labels = np.zeros((self.final_n_datapoints, 360))
        global_index = 0
        with tqdm(total=self.final_n_datapoints) as pbar:
            for world_name in self.index["data"].keys():
                folder_name = self.index["data"][world_name]["images_folder"]
                samples_to_load = self.n_samples_per_world[world_name]
                path_to_world_folder = os.path.join(
                    self.index["info"]["path_to_dataset"], folder_name
                )
                assert os.path.exists(path_to_world_folder)
                raw_idxs = np.arange(0, self.index["data"][world_name]["n_datapoints"])
                np.random.shuffle(raw_idxs)
                idxs = raw_idxs[:samples_to_load]
                for idx in idxs:
                    file_name = f"{idx:010d}.npz"
                    path_to_file = os.path.join(path_to_world_folder, file_name)
                    data = np.load(path_to_file)
                    self.images[global_index, 0, :, :] = torch.tensor(data["image"])
                    self.labels[global_index] = torch.tensor(data["label"])
                    global_index += 1
                    pbar.update(1)
        self.images = torch.tensor(self.images)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return self.final_n_datapoints

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def main(P):
    run = neptune.init_run(
        project="lcano/gallery-detection",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiYjcxZGU4OC00ZjVkLTRmMDAtYjBlMi0wYzkzNDQwOGJkNWUifQ==",
    )  # your credentials
    run["parameters"] = P
    dataset_folder_path = P["dataset_folder_path"]
    n_samples = P["n_samples"]
    batch_size = P["batch_size"]
    n_epochs = P["n_epochs"]
    lr = P["lr"]
    lr_decay = P["lr_decay"]
    path_to_index_file = os.path.join(dataset_folder_path, "index.json")
    assert os.path.exists(path_to_index_file)
    with open(path_to_index_file, "r") as f:
        index = json.load(f)
    print("Dataset info: ")
    pprint(index["info"])
    dataset = GalleryDetectionDataset(index, n_desired_samples=n_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = GalleryDetectorV3()
    model = model.type(torch.float)
    model = model.to("cuda")
    optimizer = Adam(model.parameters(), lr=lr)
    lr_scheduler = ExponentialLR(optimizer, lr_decay)
    criterion = MSELoss(reduction="mean")
    for n_epoch in tqdm(range(n_epochs)):
        for data in tqdm(dataloader, leave=False):
            img, lbl = data
            img = img.to("cuda").type(torch.float)
            lbl = lbl.to("cuda").type(torch.float)
            pred = model(img)
            optimizer.zero_grad()
            loss = criterion(lbl, pred)
            run["train/loss"].append(loss.item())
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        fig = plt.figure()
        plt.plot(pred[0].detach().cpu().numpy())
        plt.plot(lbl[0].detach().cpu().numpy())
        file = File.as_html(fig)
        run["plots/predictions"].upload(file)
    run.stop()


if __name__ == "__main__":
    main(PARAMETERS)
