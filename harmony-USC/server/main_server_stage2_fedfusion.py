import socketserver
import pickle, struct
import os
import sys
import argparse
import time
from threading import Lock, Thread
import threading
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity


# np.set_printoptions(threshold=np.inf)


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    ## FL
    parser.add_argument('--fl_round', type=int, default=21,
                    help='communication to server after the epoch of local training')
    parser.add_argument('--num_of_users', type=int, default=4,
                    help='num of users in FL')

    ## model
    parser.add_argument('--num_of_modality', type=int, default=2,
                    help='num of modality in model')
    parser.add_argument('--dim_enc_multi', type=int, default=39456,
                    help='dimension of encoders in multimodal model')
    parser.add_argument('--dim_enc_single', type=int, default=int(39456/2),
                    help='dimension of encoder in singlemodal model')
    parser.add_argument('--dim_cls_multi', type=int, default=2829532,
                    help='dimension of encoders in multimodal model')
    parser.add_argument('--dim_cls_single', type=int, default=2829532,
                    help='dimension of encoder in singlemodal model')
    parser.add_argument('--dim_multi', type=int, default=int(39456 + 2829532),
                    help='dimension of encoders in multimodal model')
    parser.add_argument('--dim_single', type=int, default=int(39456/2 + 2829532),
                    help='dimension of encoder in singlemodal model')
    parser.add_argument('--server_folder', type=str, default='save_enc_cen_fusion_fl_all/save_server_enc_cen',
                        # choices=['save_server_enc_cen', 'save_server_enc_fl'], 
                        help='server_folder')


    opt = parser.parse_args()

    return opt


opt = parse_option()

torch.manual_seed(42)

iteration_count = 0
trial_count = 0
NUM_OF_WAIT = opt.num_of_users

## recieved all model weights
ENC = np.zeros((opt.num_of_users, opt.dim_enc_multi))
CLS = np.zeros((opt.num_of_users, opt.dim_cls_multi))

Update_Flag = np.ones(opt.num_of_users)
Local_Modality = np.zeros(opt.num_of_users).astype(int)

conver_indicator = 1e5

mean_encoder_0 = np.zeros(opt.dim_enc_single)
mean_encoder_1 = np.zeros(opt.dim_enc_single)
mean_encoder_all = np.zeros(opt.dim_enc_multi)

mean_classifier_0 = np.zeros(opt.dim_cls_single)
mean_classifier_1 = np.zeros(opt.dim_cls_single)
mean_classifier_multi = np.zeros(opt.dim_cls_multi)


# for multimodal nodes
def temp_to_user(temp_id, local_modality):

	reorder_array = np.loadtxt("reorder_id_stage2.txt").astype(int)
	user_id = reorder_array[temp_id, 1]

	return user_id


def group_index(user_id):

	num_of_group = 2

	## iid, 4,5,5
	group_set = [[0,2], [1,3]]

	for group_id in range(num_of_group):

		if user_id in group_set[group_id]:
			return group_id



def mmFedavg_encoder(opt, encoder):

	num_of_group = 2

	mean_enc_all = np.zeros((num_of_group, opt.dim_enc_multi))
	count_user = np.zeros(num_of_group)

	for user_id in range(opt.num_of_users):

		group_id = group_index(user_id)
		mean_enc_all[group_id] += encoder[user_id]
		count_user[group_id] += 1

	for index_group in range(num_of_group):
		mean_enc_all[index_group] = mean_enc_all[index_group] / count_user[index_group]


	return mean_enc_all


def mmFedavg_classifier(opt, classifier):

	num_of_group = 2

	mean_classifier_all = np.zeros((num_of_group, opt.dim_cls_multi))
	count_user = np.zeros(num_of_group)

	for user_id in range(opt.num_of_users):

		group_id = group_index(user_id)
		mean_classifier_all[group_id] += classifier[user_id]
		count_user[group_id] += 1

	for index_group in range(num_of_group):
		mean_classifier_all[index_group] = mean_classifier_all[index_group] / count_user[index_group]


	return mean_classifier_all


def weight_dis_matrix(opt, weight):

	dis_matrix = np.zeros((opt.num_of_users, opt.num_of_users))

	dis_matrix = cosine_similarity(weight)


	return dis_matrix



def calculate_dis(opt, encoder, classifier):

	# print("debug here 2")

	enc_1 = encoder[:, 0:opt.dim_enc_single]
	enc_2 = encoder[:, opt.dim_enc_single:]

	# print("debug here 3")
	enc_1_dis = weight_dis_matrix(opt, enc_1)
	enc_2_dis = weight_dis_matrix(opt, enc_2)
	cls_dis = weight_dis_matrix(opt, classifier)

	return enc_1_dis, enc_2_dis, cls_dis



def server_update():
	
	global opt, ENC, CLS, Local_Modality, iteration_count
	global mean_encoder_all
	global mean_classifier_multi

	global trial_count, opt
	opt.save_folder = "./" + opt.server_folder + "/trial_" + str(trial_count) + "/"
	if not os.path.isdir(opt.save_folder):
	    os.makedirs(opt.save_folder)

	# print("debug here 1")

	encoder_1_dis, encoder_2_dis, classifier_dis = calculate_dis(opt, ENC, CLS)
	np.savetxt(opt.save_folder + "enc_1_dis_{}.txt".format(iteration_count), encoder_1_dis)
	np.savetxt(opt.save_folder + "enc_2_dis_{}.txt".format(iteration_count), encoder_2_dis)
	np.savetxt(opt.save_folder + "cls_dis_{}.txt".format(iteration_count), classifier_dis)
	print("Iteration {}: distance of encoder 1".format(iteration_count), encoder_1_dis)
	print("Iteration {}: distance of encoder 2".format(iteration_count), encoder_2_dis)
	print("Iteration {}: distance of classifiers".format(iteration_count), classifier_dis)


	## mmFedavg for model encoders
	print("Iteration {}: mmFedavg of encoders".format(iteration_count))
	mean_encoder_all = mmFedavg_encoder(opt, ENC)

	## mmFedavg for classifiers
	print("Iteration {}: mmFedavg of classifiers".format(iteration_count))
	mean_classifier_multi = mmFedavg_classifier(opt, CLS)


	iteration_count = iteration_count + 1
	print("iteration_count: ", iteration_count)

	
def reinitialize():

	global iteration_count, trial_count
	trial_count += 1
	iteration_count = 0
	print("Trial: ", trial_count)

	global opt, NUM_OF_WAIT
	opt = parse_option()
	NUM_OF_WAIT = opt.num_of_users

	global ENC, CLS, Update_Flag, Local_Modality

	## recieved all model weights
	ENC = np.zeros((opt.num_of_users, opt.dim_enc_multi))
	CLS = np.zeros((opt.num_of_users, opt.dim_cls_multi))

	Update_Flag = np.ones(opt.num_of_users)
	Local_Modality = np.zeros(opt.num_of_users).astype(int)

	global mean_encoder_all, mean_classifier_multi


	mean_encoder_all = np.zeros(opt.dim_enc_multi)
	mean_classifier_multi = np.zeros(opt.dim_cls_multi)

	barrier_update()



barrier_start = threading.Barrier(NUM_OF_WAIT,action = None, timeout = None)
barrier_W = threading.Barrier(NUM_OF_WAIT,action = server_update, timeout = None)
barrier_end = threading.Barrier(NUM_OF_WAIT, action = reinitialize, timeout = None)

def barrier_update():
	global NUM_OF_WAIT
	print("update the barriers to NUM_OF_WAIT: ",NUM_OF_WAIT)
	global barrier_W
	barrier_W = threading.Barrier(NUM_OF_WAIT,action = server_update, timeout = None)
	global barrier_end
	barrier_end = threading.Barrier(NUM_OF_WAIT, action = reinitialize, timeout = None)


class MyTCPHandler(socketserver.BaseRequestHandler):

	def send2node(self, var):

		var_data = pickle.dumps(var, protocol = 0)
		var_size = sys.getsizeof(var_data)
		var_header = struct.pack("i",var_size)
		self.request.sendall(var_header)
		self.request.sendall(var_data)

		return var_size


	def handle(self):
		while True:
			try:
				#receive the size of content
				header = self.request.recv(4)
				size = struct.unpack('i', header)

				#receive the id of client
				u_id = self.request.recv(4)
				user_id = struct.unpack('i',u_id)

				# receive the type of message, defination in communication.py
				# mess_type = self.request.recv(4)
				# mess_type = struct.unpack('i',mess_type)[0]

				#receive the id of client
				u_id = self.request.recv(4)
				temp_id = struct.unpack('i',u_id)
				user_id = temp_to_user(int(temp_id[0]), 3)
				# print("user_id:", user_id)

				#print("This is the {}th node with message type {}".format(user_id[0],mess_type))

				#receive the body of message
				recv_data = b""
				
				while sys.getsizeof(recv_data)<size[0]:
					recv_data += self.request.recv(size[0]-sys.getsizeof(recv_data))
				
				#if hello message, barrier until all clients arrive and send a message to start
				if mess_type == -1:
					try:
						barrier_start.wait(120)
					except Exception as e:
						print("start wait timeout...")

					start_message = 'start'
					mess_size = self.send2node(start_message)

				# if modality message, record the local modality
				elif mess_type == 1:

					try:
						barrier_start.wait(10)
					except Exception as e:
						print("wait W timeout...")

					temp_modality = pickle.loads(recv_data)

					if temp_modality == 'both':
						Local_Modality[user_id] = 2
					elif temp_modality == 'acc':
						Local_Modality[user_id] = 0
					elif temp_modality == 'gyr':
						Local_Modality[user_id] = 1
					print("client {} has modality {}".format(user_id[0], Local_Modality[user_id]))


				#if W message, server update for model aggregation
				elif mess_type == 0:

					weights = pickle.loads(recv_data)
					print(weights)

					if Local_Modality[user_id] == 2:
						ENC[user_id] = weights[0:opt.dim_enc_multi]
						CLS[user_id] = weights[opt.dim_enc_multi:]

					try:
						barrier_W.wait(120)
					except Exception as e:
						print("wait W timeout...")

					if Local_Modality[user_id] == 2:
						group_id = group_index(int(user_id[0]))
						print("group_id:", group_id)
						send_model = np.append(mean_encoder_all[group_id], mean_classifier_multi[group_id])
						print("send_model:", send_model.shape)
						mess_size = self.send2node(send_model)#self.send2node(New_ALL[user_id])


					print("send New_W to client {} with the size of {}".format(user_id[0],mess_size))


					#if Update_Flag=0, stop the specific client
					if Update_Flag[user_id]==0:
						
						sig_stop = struct.pack("i",2)

						global NUM_OF_WAIT
						NUM_OF_WAIT-=1
						barrier_update()
						self.finish()

					#if convergence, stop all the clients
					elif(np.abs(conver_indicator)<1e-2 or (iteration_count == opt.fl_round)):
						sig_stop = struct.pack("i",1)
					else:
						sig_stop = struct.pack("i",0)
					self.request.sendall(sig_stop)


				elif mess_type == 9:
					break

				elif mess_type == 10:
					try:
						barrier_end.wait(5)
					except Exception as e:
						print("finish timeout...")
					break


			except Exception as e:
				print('err',e)
				break



if __name__ == "__main__":
	HOST, PORT = "0.0.0.0", 9998 
	server = socketserver.ThreadingTCPServer((HOST,PORT),MyTCPHandler)
	# server.server_close()
	server.serve_forever(poll_interval = 0.5)
