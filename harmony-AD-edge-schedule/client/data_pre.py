import math
import numpy as np
import torch
import os



class Unimodal_dataset():
    """Build dataset from audio data."""

    def __init__(self, node_id, modality_str):

        # user_id = node2user[node_id]
        self.folder_path = "../../AD-data/node_{}/".format(node_id)
        self.modality = modality_str
        self.data_path = self.folder_path + "{}/".format(self.modality)

        y = np.load(self.folder_path + "label.npy")

        self.labels = y.tolist()
        self.labels = torch.tensor(self.labels)
        self.labels = (self.labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x = np.load(self.data_path + "{}.npy".format(idx))

        self.data = x.tolist()
        self.data = torch.tensor(self.data)

        sensor_data = self.data
        if self.modality == 'depth':
            sensor_data = torch.unsqueeze(sensor_data, 0)

        activity_label = self.labels[idx]

        return sensor_data, activity_label


class Multimodal_dataset():
    """Build dataset from motion sensor data."""
    def __init__(self, node_id):

        # user_id = node2user[node_id]
        self.folder_path = "../../AD-data/node_{}/".format(node_id)
        y = np.load(self.folder_path + "label.npy")

        self.labels = y.tolist() #tolist
        self.labels = torch.tensor(self.labels)
        self.labels = (self.labels).long()


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x1 = np.load(self.folder_path + "audio/" + "{}.npy".format(idx))
        x2 = np.load(self.folder_path + "depth/" + "{}.npy".format(idx))
        x3 = np.load(self.folder_path + "radar/" + "{}.npy".format(idx))

        self.data1 = x1.tolist() #concate and tolist
        self.data2 = x2.tolist() #concate and tolist
        self.data3 = x3.tolist()
        
        sensor_data1 = torch.tensor(self.data1) # to tensor
        sensor_data2 = torch.tensor(self.data2) # to tensor
        sensor_data3 = torch.tensor(self.data3) # to tensor

        sensor_data2 = torch.unsqueeze(sensor_data2, 0)

        activity_label = self.labels[idx]

        return sensor_data1, sensor_data2, sensor_data3, activity_label



