import torch
import torch.nn as nn

class CIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10Net, self).__init__()
        
        # C1: First Convolutional Block (3 layers)
        # Initial RF: 3x3 -> 5x5 -> 7x7
        self.c1 = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Layer 2
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Layer 3
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
        # Transition 1: 1x1 conv
        self.t1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # C2: Second Conv Block with Dilated Convolutions (3 layers)
        # RF: 11x11 -> 15x15 -> 19x19
        self.c2 = nn.Sequential(
            # Layer 1
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Layer 2
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Layer 3
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # Transition 2: 1x1 conv
        self.t2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        # C3: Third Block with Depthwise Separable Convolutions (3 layers)
        # RF: 23x23 -> 27x27 -> 31x31
        self.c3 = nn.Sequential(
            # Layer 1
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48),
            nn.Conv2d(48, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # Layer 2
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48),
            nn.Conv2d(48, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # Layer 3
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48),
            nn.Conv2d(48, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        
        # Transition 3: 1x1 conv
        self.t3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # C4: Final Conv Block with Dilated Convolutions (3 layers)
        # RF: 35x35 -> 43x43 -> 51x51
        self.c4 = nn.Sequential(
            # Layer 1
            nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Layer 2
            nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Layer 3
            nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Global Average Pooling and Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Main blocks
        x = self.c1(x)
        x = self.t1(x)
        x = self.c2(x)
        x = self.t2(x)
        x = self.c3(x)
        x = self.t3(x)
        x = self.c4(x)
        
        # Classifier
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x 