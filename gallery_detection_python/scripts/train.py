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
from PIL import Image

PARAMETERS = {
    "dataset_folder_path": "/media/lorenzo/SAM500/datasets/temp_dataset",
    "n_samples": None,
    "batch_size": 64,
    "n_epochs": 512,
    "lr": 0.00004,
    "lr_decay": 0.999,
    "save_folder": "/media/lorenzo/SAM500/models/gallery-detection/",
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
        self.images = torch.zeros(
            (self.final_n_datapoints, 1, 16, self.index["info"]["image_width"])
        )
        self.labels = torch.zeros((self.final_n_datapoints, 360))
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
        print("Dataset loaded")

    def __len__(self):
        return self.final_n_datapoints

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def main(P):
    print("Starting neptune run")
    run = neptune.init_run(
        project="lcano/gallery-detection",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiYjcxZGU4OC00ZjVkLTRmMDAtYjBlMi0wYzkzNDQwOGJkNWUifQ==",
        capture_stderr=False,
        capture_stdout=False,
    )  # your credentials
    print("Getting parameters")
    dataset_folder_path = P["dataset_folder_path"]
    n_samples = P["n_samples"]
    batch_size = P["batch_size"]
    n_epochs = P["n_epochs"]
    lr = P["lr"]
    lr_decay = P["lr_decay"]
    save_folder = P["save_folder"]
    path_to_index_file = os.path.join(dataset_folder_path, "index.json")
    assert os.path.exists(path_to_index_file)
    with open(path_to_index_file, "r") as f:
        index = json.load(f)
    os.makedirs(save_folder, exist_ok=True)
    print("Dataset info: ")
    pprint(index["info"])
    dataset = GalleryDetectionDataset(index, n_desired_samples=n_samples)
    print("Creating dataloader")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Instanciating model")
    model = GalleryDetectorV3()
    model = model.type(torch.float)
    model = model.to("cuda")
    print("Instanciating optimizer")
    optimizer = Adam(model.parameters(), lr=lr)
    print("Instanciating lr scheduler")
    lr_scheduler = ExponentialLR(optimizer, lr_decay)
    print("Instanciating loss")
    criterion = MSELoss(reduction="mean")
    print("Start training")
    for n_epoch in tqdm(range(n_epochs)):
        epoch_avg_loss = 0
        n_batches = 0
        for batch_data in tqdm(dataloader, leave=False):
            n_batches += 1
            img, lbl = batch_data
            img = img.to("cuda").type(torch.float)
            lbl = lbl.to("cuda").type(torch.float)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(lbl, pred)
            loss.backward()
            optimizer.step()
            epoch_avg_loss += loss.item()
            run["train/loss"].append(loss.item())
        lr_scheduler.step()
        epoch_avg_loss /= n_batches
        if epoch_avg_loss < 0.002:
            save_file_path = os.path.join(save_folder, f"{model.__class__.__name__}.torch")
            torch.save(model.state_dict(), save_file_path)
        fig = plt.figure()
        axes1 = fig.add_axes([0, 0, 1, 1])
        axes1.imshow(torch.clone(img[0][0]).detach().cpu().numpy())
        axes2 = fig.add_axes([0, 1, 1, 1])
        axes2.plot(torch.clone(pred[0].detach().cpu()).numpy())
        axes2.plot(torch.clone(lbl[0].detach().cpu()).numpy())
        run["predictions"].append(fig)
        plt.close()

    run.stop()


if __name__ == "__main__":
    main(PARAMETERS)
