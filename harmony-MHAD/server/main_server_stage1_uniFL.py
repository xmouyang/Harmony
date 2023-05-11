import socketserver
import pickle, struct
import os
import sys
import argparse
import time
from threading import Lock, Thread
import threading
import numpy as np

# np.set_printoptions(threshold=np.inf)


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    ## FL
    parser.add_argument('--fl_round', type=int, default=21,
                    help='communication to server after the epoch of local training')
    parser.add_argument('--num_of_users', type=int, default=9,
                    help='num of users in FL')

    ## model
    parser.add_argument('--num_of_modality', type=int, default=2,
                    help='num of modality in model')
    parser.add_argument('--modality_group', type=str, default="acc",
                    help='modality_group')
    parser.add_argument('--dim_enc_0', type=int, default = 30752)
    parser.add_argument('--dim_enc_1', type=int, default = 357792)
    parser.add_argument('--dim_enc_multi', type=int, default = 388544)
    parser.add_argument('--dim_cls_0', type=int, default = 2827)
    parser.add_argument('--dim_cls_1', type=int, default = 2827)
    parser.add_argument('--dim_cls_multi', type=int, default = 17291)
    parser.add_argument('--dim_all_1', type=int, default = 33579)
    parser.add_argument('--dim_all_2', type=int, default = 360619)
    parser.add_argument('--dim_all_multi', type=int, default=405835)

    opt = parser.parse_args()

    return opt


opt = parse_option()


iteration_count = 0
trial_count = 0
NUM_OF_WAIT = opt.num_of_users

## recieved all model weights
ENC = np.zeros((opt.num_of_users, opt.dim_enc_multi))
CLS = np.zeros((opt.num_of_users, opt.dim_cls_multi))

Update_Flag = np.ones(opt.num_of_users)
Local_Modality = np.zeros(opt.num_of_users).astype(int)

conver_indicator = 1e5

mean_encoder_0 = np.zeros(opt.dim_enc_0)
mean_encoder_1 = np.zeros(opt.dim_enc_1)
mean_encoder_all = np.zeros(opt.dim_enc_multi)

mean_classifier_0 = np.zeros(opt.dim_cls_0)
mean_classifier_1 = np.zeros(opt.dim_cls_1)
mean_classifier_multi = np.zeros(opt.dim_cls_multi)


# for uniFL
def temp_to_user(temp_id, local_modality):

	if local_modality == "acc":
		reorder_array = np.loadtxt("reorder_id_stage1_acc.txt").astype(int)
	elif local_modality == "skeleton":
		reorder_array = np.loadtxt("reorder_id_stage1_skeleton.txt").astype(int)

	user_id = reorder_array[temp_id, 1]

	return user_id


def mmFedavg_encoder(opt, encoder, local_modality):

	count_modality = np.zeros(opt.num_of_modality)

	encoder_0_record = np.zeros(opt.dim_enc_0)
	encoder_1_record = np.zeros(opt.dim_enc_1)

	for user_id in range(opt.num_of_users):

		if local_modality[user_id] == 0:
			encoder_0_record += encoder[user_id][0:opt.dim_enc_0]## the encoder weights are saved on the head for single modality nodes
			count_modality[0] += 1
		elif local_modality[user_id] == 1:
			encoder_1_record += encoder[user_id][0:opt.dim_enc_1]
			count_modality[1] += 1

	if count_modality[0] != 0:
		encoder_0_record = encoder_0_record / count_modality[0]
		print("count_modality encoder 0: {}".format(count_modality[0]))

	if count_modality[1] != 0:
		encoder_1_record = encoder_1_record / count_modality[1]
		print("count_modality encoder 1: {}".format(count_modality[1]))


	return encoder_0_record, encoder_1_record


def mmFedavg_classifier(opt, classifier, local_modality):

	count_modality = np.zeros(opt.num_of_modality)

	classifier_0_record = np.zeros(opt.dim_cls_0)
	classifier_1_record = np.zeros(opt.dim_cls_1)


	for user_id in range(opt.num_of_users):

		if local_modality[user_id] == 0:
			classifier_0_record += classifier[user_id][0:opt.dim_cls_0]## the classifier weights are saved on the head for single modality nodes
		elif local_modality[user_id] == 1:
			classifier_1_record += classifier[user_id][0:opt.dim_cls_1]
			
		count_modality[local_modality[user_id]] += 1


	if count_modality[0] != 0:
		classifier_0_record = classifier_0_record / count_modality[0]
		print("count_modality classifier 0: {}".format(count_modality[0]))

	if count_modality[1] != 0:
		classifier_1_record = classifier_1_record / count_modality[1]
		print("count_modality classifier 1: {}".format(count_modality[1]))


	return classifier_0_record, classifier_1_record


def server_update():
	
	global opt, ENC, CLS, Local_Modality, iteration_count
	global mean_encoder_0, mean_encoder_1, mean_encoder_all
	global mean_classifier_0, mean_classifier_1, mean_classifier_multi

	## mmFedavg for model encoders
	print("Iteration {}: mmFedavg of encoders".format(iteration_count))
	mean_encoder_0, mean_encoder_1 = mmFedavg_encoder(opt, ENC, Local_Modality)

	## mmFedavg for classifiers
	print("Iteration {}: mmFedavg of classifiers".format(iteration_count))
	mean_classifier_0, mean_classifier_1 = mmFedavg_classifier(opt, CLS, Local_Modality)


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

	global mean_encoder_0, mean_encoder_1, mean_encoder_all, mean_classifier_0, mean_classifier_1, mean_classifier_multi

	mean_encoder_0 = np.zeros(opt.dim_enc_0)
	mean_encoder_1 = np.zeros(opt.dim_enc_1)
	mean_encoder_all = np.zeros(opt.dim_enc_multi)

	mean_classifier_0 = np.zeros(opt.dim_cls_0)
	mean_classifier_1 = np.zeros(opt.dim_cls_1)
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

				# #receive the id of client
				# u_id = self.request.recv(4)
				# user_id = struct.unpack('i',u_id)

				#receive the id of client
				u_id = self.request.recv(4)
				temp_id = struct.unpack('i',u_id)

				user_id = temp_to_user(int(temp_id[0]), opt.modality_group)
				# print("user_id:", user_id)

				# receive the type of message, defination in communication.py
				mess_type = self.request.recv(4)
				mess_type = struct.unpack('i',mess_type)[0]

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
					elif temp_modality == 'skeleton':
						Local_Modality[user_id] = 1
					print("client {} has modality {}".format(user_id[0], Local_Modality[user_id]))


				#if W message, server update for model aggregation
				elif mess_type == 0:

					weights = pickle.loads(recv_data)
					print(weights)

					if Local_Modality[user_id] == 0:
						ENC[user_id][0:opt.dim_enc_0] = weights[0:opt.dim_enc_0]## the encoder weights are saved on the head for single modality nodes
						CLS[user_id][0:opt.dim_cls_0] = weights[opt.dim_enc_0:]## the classifier weights are saved on the head for single modality nodes
					elif Local_Modality[user_id] == 1:
						ENC[user_id][0:opt.dim_enc_1] = weights[0:opt.dim_enc_1]## the classifier weights are saved on the head for single modality nodes
						CLS[user_id][0:opt.dim_cls_1] = weights[opt.dim_enc_1:]## the classifier weights are saved on the head for single modality nodes

					try:
						barrier_W.wait(120)
					except Exception as e:
						print("wait W timeout...")

					if Local_Modality[user_id] == 0:
						send_weight = np.append(mean_encoder_0, mean_classifier_0)
					elif Local_Modality[user_id] == 1:
						send_weight = np.append(mean_encoder_1, mean_classifier_1)
					
					mess_size = self.send2node(send_weight)

					# print(send_weight)
					print("send New_W to client {} with the size of {}, and shape of {}".format(user_id, mess_size, send_weight.shape))


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
						barrier_end.wait(500)
					except Exception as e:
						print("finish timeout...")
					break


			except Exception as e:
				print('err',e)
				break



if __name__ == "__main__":
	
	HOST = "0.0.0.0"

	opt = parse_option()
	if opt.modality_group == "acc":
		PORT = 9998
	elif opt.modality_group == "skeleton":
		PORT = 9997
	print(PORT)

	server = socketserver.ThreadingTCPServer((HOST,PORT),MyTCPHandler)
	# server.server_close()
	server.serve_forever(poll_interval = 0.5)
