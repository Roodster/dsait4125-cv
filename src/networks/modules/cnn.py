import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)  # Convolutional layer 1
        self.relu1 = nn.ReLU()  # ReLU activation function
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer 1

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1)  # Convolutional layer 2
        self.relu2 = nn.ReLU()  # ReLU activation function
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer 2

        # Adaptive average pooling to handle varying input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(in_features=32, out_features=num_classes)  # Fully connected layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.avgpool(x)  # Output shape: (batch_size, 32, 1, 1)
        x = torch.flatten(x, 1) # Output shape: (batch_size, 32)

        x = self.fc(x)

        return x
