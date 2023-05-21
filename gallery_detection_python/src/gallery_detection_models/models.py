import torch
from torch import nn


class gallery_detector_v3(nn.Module):
    def __init__(self):
        self.input_height = 16
        self.input_width = 360
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, [3, 3], padding=(0, 1), padding_mode="circular"),
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Dropout(p=0.05),
            nn.MaxPool2d([2, 2]),
            nn.ReLU(),
            nn.Conv2d(8, 16, [3, 3], padding=(0, 1), padding_mode="circular"),
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Dropout(p=0.05),
            nn.Conv2d(16, 16, [3, 3], padding=(0, 1), padding_mode="circular"),
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Dropout(p=0.05),
            nn.MaxPool2d([2, 2]),
            nn.ReLU(),
            nn.Conv2d(16, 32, [3, 3], padding=(0, 1), padding_mode="circular"),
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Conv2d(32, 32, [3, 3], padding=(0, 1), padding_mode="circular"),
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Dropout(p=0.05),
            nn.MaxPool2d([2, 2]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2880, 2880),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(2880, 1440),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(1440, 720),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(720, 360),
            nn.ReLU(),
        )

    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits
