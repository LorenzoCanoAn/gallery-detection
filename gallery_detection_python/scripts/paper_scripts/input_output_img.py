from gallery_detection_models.models import GalleryDetectorV2
import torch
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.npyio import NpzFile
import os

PATH_TO_DATASET = "/media/lorenzo/SAM500/datasets/gallery_detection_dataset/env_001"
PATH_TO_MODEL = "/media/lorenzo/SAM500/models/gallery-detection/GalleryDetectorV2.v4_128_epochs.torch"
model = GalleryDetectorV2()
model.load_state_dict(torch.load(PATH_TO_MODEL))
model = model.eval().cpu().type(torch.float32)
for filename in os.listdir(PATH_TO_DATASET):
    path = os.path.join(PATH_TO_DATASET, filename)
    data = np.load(path)
    assert isinstance(data, NpzFile)
    image = data["image"]
    label = data["label"]
    image_torch = torch.tensor(image).unsqueeze(0).unsqueeze(0).type(torch.float32)
    prediction = model(image_torch).detach().numpy()
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 1, 2)
    plt.plot(prediction[0])
    plt.xticks([0, 90, 180, 270, 360], labels=["$0$", "$\\pi/2$", "$\\pi$", "$3\\pi/2$", "$2\\pi$"])
    plt.show()
