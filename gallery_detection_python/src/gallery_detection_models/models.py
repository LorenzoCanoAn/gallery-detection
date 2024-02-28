from torch import nn
import torch


class GalleryDetectorV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Sequential(
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
            nn.Conv2d(8, 1, [6, 3], padding=(0, 1), padding_mode="circular", stride=(2, 1)),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(254, 360),
            nn.ReLU(),
        )
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(32, 16, 3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(16, 1, 3, padding=1, padding_mode="circular"),
            nn.ReLU(),
        )

    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        x = self.conv2d(x)
        # print("Conv2d")
        # print(torch.mean(x))
        x = torch.flatten(x, 1)
        # print("Flatten")
        # print(torch.mean(x))
        x = self.fc(x)
        # print("Fc")
        # print(torch.mean(x))
        x = torch.unsqueeze(x, 1)
        # print("unsqz")
        # print(torch.mean(x))
        x = self.conv1d(x)
        # print("conv1d")
        # print(torch.mean(x))
        x = torch.flatten(x, 1)
        # print("flatten")
        # print(torch.mean(x))
        return x


if __name__ == "__main__":
    from torchinfo.torchinfo import summary

    model = GalleryDetectorV3()
    summary(model, input_size=(1, 1, 16, 1024))
