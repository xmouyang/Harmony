import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):
    def __init__(
            self,
            input_dim=20,
            output_dim=512,
            context_size=5,
            stride=1,
            dilation=1,
            batch_norm=True,
            dropout_p=0.0
    ):
        """
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        """
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        """
        input : size (batch, seq_len, input_features)
        output: size (batch, new_seq_len, output_features)
        """
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.input_dim),
            stride=(1, self.input_dim),
            dilation=(self.dilation, 1)
        )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x)
        x = self.nonlinearity(x)

        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x


## audio input: [bsz, 20, 87]
class audio_encoder(nn.Module):
    """
    model for audio data
    """

    def __init__(self):
        super().__init__()

        self.tdnn1 = TDNN(input_dim=20, output_dim=256, context_size=5, dilation=5)
        self.tdnn2 = TDNN(input_dim=256, output_dim=512, context_size=5, dilation=5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=256, context_size=5, dilation=5)
        self.tdnn4 = TDNN(input_dim=256, output_dim=128, context_size=3, dilation=3)
        self.tdnn5 = TDNN(input_dim=128, output_dim=128, context_size=3, dilation=3)

        self.gru = nn.GRU(128, 16, 2, batch_first=True)

    def forward(self, x):

        # self.gru.flatten_parameters()
        x = x.transpose(1,2)

        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        
        # print("original audio feature:", x.shape)#[8, 15, 128]

        x = x.reshape(x.size(0), -1, 128)#[bsz, 15, 128]
        x, _ = self.gru(x)

        # print("audio feature after gru:", x.shape)#[bsz, 15, 16]

        out = x.reshape(x.size(0), -1)#[bsz, 240]

        return out


## depth input: [bsz, 1, 16, 112, 112]
class depth_encoder(nn.Module):
    """
    model for depth video
    """

    def __init__(self):
        super().__init__()

        # conv1 input (n*1*16*112*112), conv5 output (n*512*1*4*4)
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        )

        self.gru = nn.GRU(64, 16, 2, batch_first=True)

    def forward(self, x):

        # self.gru.flatten_parameters()

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # print("original depth feature:", x.shape)#[bsz, 64, 1, 4, 4]

        x = x.view(x.size(0), 16, -1)#[bsz, 16, 64]
        x, _ = self.gru(x)

        out = x.reshape(x.size(0), -1)#[bsz, 256]

        # print("depth feature after gru:", out.shape)

        return out


## depth input: [bsz, 20, 2, 16, 32, 16]
class radar_encoder(nn.Module):
    """
    For radar: input size (20*16*32*16)
    """

    def __init__(self):
        super().__init__()

        # conv1 input (n*20)*2*16*32*16, conv4 output (n*20)*256*2*4*2
        self.conv1 = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64)
        )
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=1024, hidden_size=16, num_layers=2, bidirectional=False, batch_first=True),
        )


    def forward(self, x):
        bsz = x.size(0)
        x = x.view(-1, 2, 16, 32, 16)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # print("original radar feature:", x.shape)#[160, 64, 2, 4, 2]
        x = x.view(bsz, 20, -1)  # [bsz, 20, 1024]

        out, _ = self.lstm(x)  # [bsz, 20, 32]
        # print("radar feature after lstm:", out.shape)# [bsz, 20, 16]

        out = out.reshape(out.size(0), -1)#[bsz, 320]

        return out


class MySingleModel(nn.Module):

    def __init__(self, num_classes, modality):
        super().__init__()

        if modality == 'audio':#[1498907]
            self.encoder = audio_encoder()#[1496256]
            self.classifier = nn.Sequential(
                nn.Linear(240, num_classes),
                nn.Softmax()
                )#[2651]
        elif modality == 'depth':#[2223883]
            self.encoder = depth_encoder()#[2221056]
            self.classifier = nn.Sequential(
                nn.Linear(256, num_classes),
                nn.Softmax()
                )#[2827]        
        elif modality == 'radar':#[629771]
            self.encoder = radar_encoder()#[626240]
            self.classifier = nn.Sequential(
            nn.Linear(320, num_classes),
            nn.Softmax()
            )#[3531]

    def forward(self, x):
        # print(x.shape)
        feature = self.encoder(x)
        output = self.classifier(feature)

        return output


class Encoder3(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = audio_encoder()
        self.encoder_2 = depth_encoder()
        self.encoder_3 = radar_encoder()

    def forward(self, x1, x2, x3):

        feature_1 = self.encoder_1(x1)
        feature_2 = self.encoder_2(x2)
        feature_3 = self.encoder_3(x3)

        return feature_1, feature_2, feature_3


class My3Model(nn.Module):

    def __init__(self, num_classes):#[4352539]
        super().__init__()

        self.encoder = Encoder3()#[4343552]

        self.classifier = nn.Sequential(
        nn.Linear(816, num_classes),
        nn.Softmax()
        )#[8987]
     
    def forward(self, x1, x2, x3):

        feature_1, feature_2, feature_3 = self.encoder(x1, x2, x3)

        feature = torch.cat((feature_1, feature_2, feature_3), dim=1)
        output = self.classifier(feature)

        return output


class Encoder2_AD(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = audio_encoder()
        self.encoder_2 = depth_encoder()

    def forward(self, x1, x2):

        feature_1 = self.encoder_1(x1)
        feature_2 = self.encoder_2(x2)

        return feature_1, feature_2

class Encoder2_DR(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = depth_encoder()
        self.encoder_2 = radar_encoder()

    def forward(self, x1, x2):

        feature_1 = self.encoder_1(x1)
        feature_2 = self.encoder_2(x2)

        return feature_1, feature_2

class Encoder2_AR(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = audio_encoder()
        self.encoder_2 = radar_encoder()

    def forward(self, x1, x2):

        feature_1 = self.encoder_1(x1)
        feature_2 = self.encoder_2(x2)

        return feature_1, feature_2


class My2Model(nn.Module):

    def __init__(self, num_classes, modality):#[4352539]
        super().__init__()

        #A, D , R: 240, 256, 320; 1496256, 2221056; 626240
        if modality == "AD":#[3722779]
            self.encoder = Encoder2_AD()#[3717312]
            self.classifier = nn.Sequential(
            nn.Linear(496, num_classes),
            nn.Softmax()
            )#[5467]
        elif modality == "DR":#[2853643]
            self.encoder = Encoder2_DR()#[2847296]
            self.classifier = nn.Sequential(
            nn.Linear(576, num_classes),
            nn.Softmax()
            )#[6347]
        elif modality == "AR":#[2128667]
            self.encoder = Encoder2_AR()#[2122496]
            self.classifier = nn.Sequential(
            nn.Linear(560, num_classes),
            nn.Softmax()
            )#[6171]

    def forward(self, x1, x2):

        feature_1, feature_2 = self.encoder(x1, x2)

        feature = torch.cat((feature_1, feature_2), dim=1)
        output = self.classifier(feature)

        return output
