import torch
import torch.nn as nn 


# ## gps input: [bsz, 2, 1]
# class gps_encoder(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.layer1 = nn.Sequential(
#         nn.Conv1d(2, 20, 2, padding = 'same'),
#         nn.ReLU(inplace=True)
#         )

#         self.layer2 = nn.Sequential(
#         nn.Conv1d(20, 40, 2, padding = 'same'),
#         nn.ReLU(inplace=True),
#         nn.MaxPool1d(2,padding = 1)
#         )

#         self.layer3 = nn.Sequential(
#         nn.Conv1d(40, 80, 2, padding = 'same'),
#         nn.ReLU(inplace=True)
#         )

#         self.layer4 = nn.Sequential(
#         nn.Conv1d(80, 40, 2, padding = 'same'),
#         nn.ReLU(inplace=True),
#         nn.MaxPool1d(2,padding = 1),
#         nn.Flatten()
#         )

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         return x


## gps input: [bsz, 2, 1]
class gps_encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
        nn.Conv1d(2, 20, 3, padding = 1),
        nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
        nn.Conv1d(20, 40, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2,padding = 1)
        )

        self.layer3 = nn.Sequential(
        nn.Conv1d(40, 80, 3, padding = 1),
        nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
        nn.Conv1d(80, 40, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2,padding = 1),
        nn.Flatten()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

## lidar input: [bsz, 20, 20, 20]
class lidar_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.channel = 32
        self.dropProb = 0.3


        self.layer1 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size = (3,3), padding = (1,1)),
            nn.ReLU(inplace = True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = (3,3), padding = (1,1)),
            nn.ReLU(inplace = True),
        
            nn.Conv2d(64, 128, kernel_size = (3,3), padding = (1,1)),
            nn.ReLU(inplace = True),

            nn.Conv2d(128, 32, kernel_size = (3,3), padding = (1,1)),
            nn.ReLU(inplace = True)

        )

        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 3)),
            nn.Dropout(p = self.dropProb)
        )

        self.maxpool_ = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p = self.dropProb)
        )

        self.flatten_layer = nn.Sequential(
            nn.Flatten()
        )



    def forward(self, x):
        a = self.layer1(x)
        x = a + self.layer2(a)
        x = self.maxpool(x) # b

        b = x
        x = self.layer2(x) + b
        x = self.maxpool(x) #c

        c = x 
        x = self.layer2(x) + c 
        x = self.maxpool_(x) #d

        d = x 
        x = self.layer2(x) + d 

        x = self.flatten_layer(x)

        return x


## image input: [bsz, 3, 112, 112]
class image_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.channel = 32
        self.dropProb = 0.25


        self.layer0 = nn.Sequential(
            nn.Conv2d(3, self.channel, kernel_size = (7,7), padding = (1,1)),
            nn.ReLU(inplace = True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.channel, 32, kernel_size = (3,3), padding = (1,1)),
            nn.ReLU(inplace = True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = (3,3), padding = (1,1)),
            nn.ReLU(inplace = True),
        
            nn.Conv2d(64, 128, kernel_size = (3,3), padding = (1,1)),
            nn.ReLU(inplace = True),

            nn.Conv2d(128, 64, kernel_size = (3,3), padding = (1,1)),
            nn.ReLU(inplace = True),

            nn.Conv2d(64, 32, kernel_size = (3,3), padding = (1,1)),
            nn.ReLU(inplace = True),

        )

        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(6, 6)),
            nn.Dropout(p = self.dropProb)
        )

        self.maxpool_ = nn.Sequential(
            nn.MaxPool2d(kernel_size=(6, 6)),
            nn.Dropout(p = self.dropProb),
            nn.Flatten()
        )


    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        b = x 
        x = self.layer2(x) + b
        x = self.maxpool(x)
        c = x 
        x = self.layer2(x) + c 
        x = self.maxpool_(x)


        return x
    
              

class MySingleModel(nn.Module):

    def __init__(self, num_classes, modality):
        super().__init__()

        # print("DEBUG: modality is: ", modality)

        if modality == 'lidar':
            self.encoder = lidar_encoder()
            self.classifier = nn.Sequential(
                nn.Linear(160, num_classes),
                nn.Softmax()
                )
        elif modality == 'image':
            self.encoder = image_encoder()
            self.classifier = nn.Sequential(
                nn.Linear(288, num_classes),
                nn.Softmax()
                )        
        elif modality == 'gps':
            self.encoder = gps_encoder()
            self.classifier = nn.Sequential(
            nn.Linear(40, num_classes),
            nn.Softmax()
            )

    def forward(self, x):
        # print(x.shape)
        feature = self.encoder(x)
        output = self.classifier(feature)

        return output


class Encoder2_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = gps_encoder()
        self.encoder_2 = lidar_encoder()

    def forward(self, x1, x2):

        feature_1 = self.encoder_1(x1)
        feature_2 = self.encoder_2(x2)

        return feature_1, feature_2

class Encoder2_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = gps_encoder()
        self.encoder_2 = image_encoder()

    def forward(self, x1, x2):

        feature_1 = self.encoder_1(x1)
        feature_2 = self.encoder_2(x2)

        return feature_1, feature_2

class Encoder2_3(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = lidar_encoder()
        self.encoder_2 = image_encoder()

    def forward(self, x1, x2):

        feature_1 = self.encoder_1(x1)
        feature_2 = self.encoder_2(x2)

        return feature_1, feature_2


class My2Model(nn.Module):

    def __init__(self, num_classes, modality):
        super().__init__()

        # print("DEBUG: modality is: ", modality)

        if modality == 'g+l':
            self.encoder = Encoder2_1()
            self.classifier = nn.Sequential(
            nn.Linear(200, num_classes),
            nn.Softmax()
            )

        elif modality == 'g+i':
            self.encoder = Encoder2_2()
            self.classifier = nn.Sequential(
                nn.Linear(328, num_classes),
                nn.Softmax()
                ) 

        elif modality == 'l+i':
            self.encoder = Encoder2_3()
            self.classifier = nn.Sequential(
                nn.Linear(448, num_classes),
                nn.Softmax()
                )
     


    def forward(self, x1, x2):
        # print(x.shape)

        feature_1, feature_2 = self.encoder(x1, x2)

        feature = torch.cat((feature_1, feature_2), dim=1)
        output = self.classifier(feature)

        return output


class Encoder3(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = gps_encoder()
        self.encoder_2 = lidar_encoder()
        self.encoder_3 = image_encoder()

    def forward(self, x1, x2, x3):

        feature_1 = self.encoder_1(x1)
        feature_2 = self.encoder_2(x2)
        feature_3 = self.encoder_3(x3)

        return feature_1, feature_2, feature_3


class My3Model(nn.Module):

    def __init__(self, num_classes, modality):
        super().__init__()

        self.encoder = Encoder3()

        self.classifier = nn.Sequential(
        nn.Linear(488, num_classes),
        nn.Softmax()
        )
     
    def forward(self, x1, x2, x3):

        feature_1, feature_2, feature_3 = self.encoder(x1, x2, x3)

        feature = torch.cat((feature_1, feature_2, feature_3), dim=1)
        output = self.classifier(feature)

        return output
