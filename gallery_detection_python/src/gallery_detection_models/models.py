from torch import nn
import torch


class GalleryDetectorV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_height = 16
        self.input_width = 360
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


class GalleryDetectorSkipConn(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 8, [3, 5], padding=(0, 2), padding_mode="circular", stride=(1, 2)),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(178, 360),
        )

    def init_weights(self):
        for name, W in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(W)

    @classmethod
    def is_2d(cls):
        return False

    def dbpt(self, x, tensor_name=""):  # Debug print
        if self.debug:
            print(
                f"{tensor_name}: {torch.mean(x).item(), torch.max(x).item(), torch.min(x).item()}"
            )
        return x

    def forward(self, x):
        x = self.dbpt(self.conv2d(x))
        x = self.dbpt(torch.flatten(x, 1))
        x = self.dbpt(self.fc(x))
        x = self.dbpt(torch.unsqueeze(x, 1))
        x = self.dbpt(self.conv1d(x))
        x = self.dbpt(torch.flatten(x, 1))
        return x


class Conv2DBnDropReLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, padding_mode, stride, p=0.05
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                padding_mode=padding_mode,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class GalleryDetectorV3(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self.conv2d = nn.Sequential(
            Conv2DBnDropReLU(1, 8, [3, 5], (0, 2), "circular", (1, 2)),
            Conv2DBnDropReLU(8, 16, [3, 5], (0, 2), "circular", (1, 2)),
            Conv2DBnDropReLU(16, 32, [3, 5], (0, 1), "circular", (1, 1)),
            Conv2DBnDropReLU(32, 16, [3, 3], (0, 1), "circular", (1, 1)),
            Conv2DBnDropReLU(16, 8, [3, 3], (0, 1), "circular", (1, 1)),
            nn.Conv2d(8, 1, [6, 3], padding=(0, 1), padding_mode="circular", stride=(2, 1)),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(178, 360),
            nn.Dropout(p=0.05),
        )
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2, padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1, padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Conv1d(32, 16, 3, padding=1, padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Conv1d(16, 1, 3, padding=1, padding_mode="circular"),
            nn.Sigmoid(),
        )

    def init_weights(self):
        for name, W in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(W)

    @classmethod
    def is_2d(cls):
        return False

    def dbpt(self, x, tensor_name=""):  # Debug print
        if self.debug:
            print(
                f"{tensor_name}: {torch.mean(x).item(), torch.max(x).item(), torch.min(x).item()}"
            )
        return x

    def forward(self, x):
        x = self.dbpt(self.conv2d(x))
        x = self.dbpt(torch.flatten(x, 1))
        x = self.dbpt(self.fc(x))
        x = self.dbpt(torch.unsqueeze(x, 1))
        x = self.dbpt(self.conv1d(x))
        x = self.dbpt(torch.flatten(x, 1))
        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = GalleryDetectorV3(debug=True)
    model.init_weights()
    summary(model, (1, 1, 16, 720))
