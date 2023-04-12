import numpy as np
import torch
import random
np.set_printoptions(threshold=np.inf)

random.seed(0)


class Multimodal_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, x1, x2, x3, y):

		self.data1 = x1.tolist() #concate and tolist
		self.data2 = x2.tolist() #concate and tolist
		self.data3 = x3.tolist()
		self.labels = y.tolist() #tolist

		self.data1 = torch.tensor(self.data1) # to tensor
		self.data2 = torch.tensor(self.data2) # to tensor
		self.data3 = torch.tensor(self.data3) # to tensor
		self.labels = torch.tensor(self.labels)
		self.labels = (self.labels).long()


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):

		sensor_data1 = self.data1[idx]
		sensor_data2 = self.data2[idx]
		sensor_data3 = self.data3[idx]

		activity_label = self.labels[idx]

		return sensor_data1, sensor_data2, sensor_data3, activity_label


class Unimodal_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, x, y):

		self.data = x.tolist() #concate and tolist
		self.labels = y.tolist() #tolist

		self.data = torch.tensor(self.data) # to tensor
		self.labels = torch.tensor(self.labels)
		self.labels = (self.labels).long()


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):

		sensor_data = self.data[idx]
		# sensor_data = torch.unsqueeze(sensor_data, 0)

		activity_label = self.labels[idx]
		# print(sensor_data.shape, activity_label.shape)

		return sensor_data, activity_label



def load_single_modality_from_one_node(modality, node_id, test_flag):
	
	node_path = ("../../FLASH-data/")

	if test_flag == 0:
		temp_path = node_path + "train/node_" + str(node_id) + "/" + modality + '.npz'
	else:
		group_id = int(int(node_id / 10) * 10)
		temp_path = node_path + "test_{}/".format(group_id) +  modality + '.npz'

	temp_data = np.load(temp_path)[modality]

	return temp_data



def load_data(modality, node_id, strategy, test_flag):

	x = load_single_modality_from_one_node(modality, node_id, test_flag)
	y = load_single_modality_from_one_node("rf", node_id, test_flag)


	return x, y
