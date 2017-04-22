from torch import nn

from utils import N_CLASSES


class BaselineCNN(nn.Module):
    def __init__(self, patch_size: int, is_fcn=False):
        super().__init__()
        assert patch_size % 8 == 0
        self.is_fcn = is_fcn
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AvgPool2d(patch_size // 8, stride=1),
            nn.Conv2d(64, N_CLASSES + 1, 1),
        )

    def forward(self, x):
        x = self.cnn(x)
        if not self.is_fcn:
            x = x.view(x.size(0), -1)
        return x
