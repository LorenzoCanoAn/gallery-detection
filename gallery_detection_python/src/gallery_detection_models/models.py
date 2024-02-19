import torch
from torch import nn


class GalleryDetectorV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, [3, 5], padding=(0, 2), padding_mode="circular", stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(8, 16, [3, 5], padding=(0, 2), padding_mode="circular", stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, [3, 5], padding=(0, 1), padding_mode="circular", stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 16, [3, 3], padding=(0, 1), padding_mode="circular", stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 8, [3, 3], padding=(0, 1), padding_mode="circular", stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(8, 1, [3, 3], padding=(0, 1), padding_mode="circular", stride=(1, 1)),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1016, 1024),
            nn.ReLU(512),
            nn.Linear(1024, 512),
            nn.ReLU(512),
            nn.Linear(512, 360),
        )

    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        logits = self.conv(x)
        logits = torch.flatten(logits, 1)
        return self.fc(logits)


if __name__ == "__main__":
    from torchinfo.torchinfo import summary

    model = GalleryDetectorV3()
    summary(model, input_size=(1, 1, 16, 1024))
