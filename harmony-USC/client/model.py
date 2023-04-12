import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np

class cnn_layers_1(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size):
        super().__init__()

        # Extract features, 2D conv layers
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16, -1)#[bsz, 16, 1, 198]

        return x


class cnn_layers_2(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size):
        super().__init__()

        # Extract features, 2D conv layers
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16, -1)

        return x


class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.acc_cnn_layers = cnn_layers_1(input_size)
        self.gyr_cnn_layers = cnn_layers_2(input_size)

    def forward(self, x1, x2):

        acc_output = self.acc_cnn_layers(x1)
        gyro_output = self.gyr_cnn_layers(x2)

        return acc_output, gyro_output



class MyMMModel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.encoder = Encoder(input_size)

        self.gru = nn.GRU(198, 120, 2, batch_first=True)

        # Classify output, fully connected layers
        self.classifier = nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes),
            )

    def forward(self, x1, x2):

        acc_output, gyro_output = self.encoder(x1, x2)

        fused_feature = (acc_output + gyro_output) / 2 # weighted sum
        # fused_feature = torch.cat((acc_output,gyro_output), dim=1) #concate
        # print(fused_feature.shape)

        fused_feature, _ = self.gru(fused_feature)
        fused_feature = fused_feature.contiguous().view(fused_feature.size(0), 1920)

        output = self.classifier(fused_feature)

        return output


class MySingleModel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes, modality):
        super().__init__()

        if modality == 'acc':
            self.encoder = cnn_layers_1(input_size)
        else:
            self.encoder = cnn_layers_2(input_size)


        self.gru = nn.GRU(198, 120, 2, batch_first=True)
        
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes),
            )

    def forward(self, x):

        x = self.encoder(x)

        x = x.view(x.size(0), 16, -1)
        x, _ = self.gru(x)

        x = x.contiguous().view(x.size(0), 1920)
        output = self.classifier(x)


        return output