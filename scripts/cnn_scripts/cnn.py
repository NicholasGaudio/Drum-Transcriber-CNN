import torch.nn as nn
import torch.nn.functional as F

class DrumCNN(nn.Module):
    def __init__(self):
        super(DrumCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(32 * 32 * 32, 64) # reduce to 64 features
        self.fc2 = nn.Linear(64, 3)  # reduce to 3 classes: kick, snare, hihat

    def forward(self, x): # x has shape of (batch_size, 1, 128, 128) (Batch, 1 channel, 128 height, 128 width)
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, 32, 32)
        x = x.view(x.size(0), -1)  # flattens each sample dynamically
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x