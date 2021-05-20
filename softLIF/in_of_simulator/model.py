import torch.nn as nn
from softLIF_activation import soft_LIF


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        # Initialize the softLIF activtion
        self.softLIF = soft_LIF().apply

    def forward(self, x):
        x= F.avg_pool2d(self.softLIF(self.conv1(x)), (2,2))
        x= F.avg_pool2d(self.softLIF(self.conv2(x)), (2,2))
        x = self.softLIF(self.conv3(x))
        x = self.softLIF(self.conv4(x))
        x = F.avg_pool2d(self.softLIF(self.conv5(x)), (2, 2))
        x = x.view(-1, 256)
        x = F.dropout(self.softLIF(self.fc1(x)), p=0.5)
        x = F.dropout(self.softLIF(self.fc2(x)), p=0.5)
        x = self.fc3(x)
        return x


def alexnet(**kwargs):
    return AlexNet(**kwargs)
