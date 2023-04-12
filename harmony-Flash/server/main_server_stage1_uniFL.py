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
    parser.add_argument('--num_of_users', type=int, default=18,
                    help='num of users in FL')

    ## model
    parser.add_argument('--num_of_modality', type=int, default=3,
                    help='num of modality in model')
    parser.add_argument('--modality_group', type=int, default=0,
                    help='modality')
    parser.add_argument('--dim_enc_0', type=int, default = 21900)
    parser.add_argument('--dim_enc_1', type=int, default = 135040)
    parser.add_argument('--dim_enc_2', type=int, default = 198592)
    parser.add_argument('--dim_enc_multi', type=int, default = 355532)
    parser.add_argument('--dim_cls_0', type=int, default = 2624)
    parser.add_argument('--dim_cls_1', type=int, default = 10304)
    parser.add_argument('--dim_cls_2', type=int, default = 18496)
    parser.add_argument('--dim_cls_multi', type=int, default = 31296)
    parser.add_argument('--dim_all_0', type=int, default = 24524)
    parser.add_argument('--dim_all_1', type=int, default = 145344)
    parser.add_argument('--dim_all_2', type=int, default = 217088)
    parser.add_argument('--dim_all_multi', type=int, default=386828)

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
mean_encoder_2 = np.zeros(opt.dim_enc_2)
mean_encoder_all = np.zeros(opt.dim_enc_multi)

mean_classifier_0 = np.zeros(opt.dim_cls_0)
mean_classifier_1 = np.zeros(opt.dim_cls_1)
mean_classifier_2 = np.zeros(opt.dim_cls_2)
mean_classifier_multi = np.zeros(opt.dim_cls_multi)

wait_time_record = np.zeros(opt.fl_round)
aggregation_time_record = np.zeros(opt.fl_round)
server_start_time_record = np.zeros((opt.num_of_users, opt.fl_round))

# for uniFL
def temp_to_user(temp_id, local_modality):

	# [0, 1, 6, 7, 8, 9,  10, 11, 16, 17, 18, 19,  20, 21,  30, 31] -> [0, 1, 2, 3, 4, 5,  6, 7, 8, 9, 10, 11, 12, 13]
	if local_modality == 0:
		if temp_id%10 < 6:
			user_id = int(temp_id/10)*6 + temp_id%10
		else:
			user_id = int(temp_id/10)*6 + temp_id%10 - 4


	# [2, 3, 6, 7, 8, 9,  12, 13,16, 17, 18, 19,  23, 23,  33, 33] -> [0, 1, 2, 3, 4, 5,  6, 7, 8, 9, 10, 11, ]
	elif local_modality == 1:

		if temp_id%10 < 6:
			user_id = int(temp_id/10)*6 + temp_id%10 - 2
		else:
			user_id = int(temp_id/10)*6 + temp_id%10 - 4

	# [4, 5, 6, 7, 8, 9,  14, 15, 16, 17, 18, 19,  24, 25,  34, 35] -> [0, 1,  2, 3,  4, 5,  6,7]
	elif local_modality == 2:
		user_id = int(temp_id/10)*6 + temp_id%10 - 4

	return user_id


def mmFedavg_encoder(opt, encoder, local_modality):

	count_modality = np.zeros(opt.num_of_modality)

	encoder_0_record = np.zeros(opt.dim_enc_0)
	encoder_1_record = np.zeros(opt.dim_enc_1)
	encoder_2_record = np.zeros(opt.dim_enc_2)
	encoder_multi_record = np.zeros(opt.dim_enc_multi)

	for user_id in range(opt.num_of_users):

		if local_modality[user_id] == 0:
			encoder_0_record += encoder[user_id][0:opt.dim_enc_0]## the encoder weights are saved on the head for single modality nodes
			count_modality[0] += 1
		elif local_modality[user_id] == 1:
			encoder_1_record += encoder[user_id][0:opt.dim_enc_1]
			count_modality[1] += 1
		elif local_modality[user_id] == 2:
			encoder_2_record += encoder[user_id][0:opt.dim_enc_2]
			count_modality[2] += 1
		else:
			encoder_0_record += encoder[user_id][0:opt.dim_enc_0]
			encoder_1_record += encoder[user_id][opt.dim_enc_0 : opt.dim_enc_0 + opt.dim_enc_1]
			encoder_2_record += encoder[user_id][opt.dim_enc_0 + opt.dim_enc_1:]
			count_modality[0] += 1
			count_modality[1] += 1
			count_modality[2] += 1

	if count_modality[0] != 0:
		encoder_0_record = encoder_0_record / count_modality[0]
		print("count_modality encoder 0: {}".format(count_modality[0]))

	if count_modality[1] != 0:
		encoder_1_record = encoder_1_record / count_modality[1]
		print("count_modality encoder 1: {}".format(count_modality[1]))

	if count_modality[2] != 0:
		encoder_2_record = encoder_2_record / count_modality[2]
		print("count_modality encoder 2: {}".format(count_modality[2]))

	encoder_multi_record[0: opt.dim_enc_0] = encoder_0_record
	encoder_multi_record[opt.dim_enc_0 : opt.dim_enc_0 + opt.dim_enc_1] = encoder_1_record
	encoder_multi_record[opt.dim_enc_0 + opt.dim_enc_1:] = encoder_2_record

	return encoder_0_record, encoder_1_record, encoder_2_record, encoder_multi_record


def mmFedavg_classifier(opt, classifier, local_modality):

	count_modality = np.zeros( int(opt.num_of_modality + 1) )

	classifier_0_record = np.zeros(opt.dim_cls_0)
	classifier_1_record = np.zeros(opt.dim_cls_1)
	classifier_2_record = np.zeros(opt.dim_cls_2)
	classifier_multi_record = np.zeros(opt.dim_cls_multi)

	for user_id in range(opt.num_of_users):

		if local_modality[user_id] == 0:
			classifier_0_record += classifier[user_id][0:opt.dim_cls_0]## the classifier weights are saved on the head for single modality nodes
		elif local_modality[user_id] == 1:
			classifier_1_record += classifier[user_id][0:opt.dim_cls_1]
		elif local_modality[user_id] == 2:
			classifier_2_record += classifier[user_id][0:opt.dim_cls_2]
		else:
			classifier_multi_record += classifier[user_id][0:opt.dim_cls_multi]
			
		count_modality[local_modality[user_id]] += 1


	if count_modality[0] != 0:
		classifier_0_record = classifier_0_record / count_modality[0]
		print("count_modality classifier 0: {}".format(count_modality[0]))

	if count_modality[1] != 0:
		classifier_1_record = classifier_1_record / count_modality[1]
		print("count_modality classifier 1: {}".format(count_modality[1]))

	if count_modality[2] != 0:
		classifier_2_record = classifier_2_record / count_modality[2]
		print("count_modality classifier 2: {}".format(count_modality[2]))

	if count_modality[3] != 0:
		classifier_multi_record = classifier_multi_record / count_modality[3]
		print("count_modality classifier multi: {}".format(count_modality[3]))

	return classifier_0_record, classifier_1_record, classifier_2_record, classifier_multi_record


def server_update():
	
	global opt, ENC, CLS, Local_Modality, iteration_count
	global mean_encoder_0, mean_encoder_1, mean_encoder_2, mean_encoder_all
	global mean_classifier_0, mean_classifier_1, mean_classifier_2, mean_classifier_multi
	global aggregation_time_record, wait_time_record, server_start_time_record

	aggregate_time1 = time.time()
	wait_time_record[iteration_count] = aggregate_time1 - np.min(server_start_time_record[:, iteration_count])
	print("server wait time:", wait_time_record[iteration_count])

	print("Local_Modality:", Local_Modality)

	## mmFedavg for model encoders
	print("Iteration {}: mmFedavg of encoders".format(iteration_count))
	mean_encoder_0, mean_encoder_1, mean_encoder_2, mean_encoder_all = mmFedavg_encoder(opt, ENC, Local_Modality)

	## mmFedavg for classifiers
	print("Iteration {}: mmFedavg of classifiers".format(iteration_count))
	mean_classifier_0, mean_classifier_1, mean_classifier_2, mean_classifier_multi = mmFedavg_classifier(opt, CLS, Local_Modality)
	# print("error 3")

	aggregate_time2 = time.time()
	aggregation_time_record[iteration_count] = aggregate_time2 - aggregate_time1
	print("server aggregation time:", aggregation_time_record[iteration_count])

	iteration_count = iteration_count + 1
	print("iteration_count: ", iteration_count)

	
def reinitialize():

	global iteration_count, trial_count
	trial_count += 1
	iteration_count = 0
	print("Trial: ", trial_count)

	global opt, NUM_OF_WAIT, wait_time_record, aggregation_time_record, server_start_time_record
	print("All of Server Wait Time:", np.sum(wait_time_record))
	print("All of Server Aggregate Time:", np.sum(aggregation_time_record))

	save_model_path = "./save_server_time/"
	if not os.path.isdir(save_model_path):
		os.makedirs(save_model_path)

	np.savetxt(os.path.join(save_model_path, "{}_aggregation_time_record.txt").format(opt.modality_group), aggregation_time_record)
	np.savetxt(os.path.join(save_model_path, "{}_wait_time_record.txt").format(opt.modality_group), wait_time_record)
	np.savetxt(os.path.join(save_model_path, "{}_server_start_time_record.txt").format(opt.modality_group), server_start_time_record)

	wait_time_record = np.zeros(opt.fl_round)
	aggregation_time_record = np.zeros(opt.fl_round)
	server_start_time_record = np.zeros((opt.num_of_users, opt.fl_round))

	opt = parse_option()
	NUM_OF_WAIT = opt.num_of_users

	global ENC, CLS, Update_Flag, Local_Modality

	## recieved all model weights
	ENC = np.zeros((opt.num_of_users, opt.dim_enc_multi))
	CLS = np.zeros((opt.num_of_users, opt.dim_cls_multi))

	Update_Flag = np.ones(opt.num_of_users)
	Local_Modality = np.zeros(opt.num_of_users).astype(int)

	global mean_encoder_0, mean_encoder_1, mean_encoder_2, mean_encoder_all, mean_classifier_0, mean_classifier_1, mean_classifier_2, mean_classifier_multi
	
	mean_encoder_0 = np.zeros(opt.dim_enc_0)
	mean_encoder_1 = np.zeros(opt.dim_enc_1)
	mean_encoder_2 = np.zeros(opt.dim_enc_2)
	mean_encoder_all = np.zeros(opt.dim_enc_multi)

	mean_classifier_0 = np.zeros(opt.dim_cls_0)
	mean_classifier_1 = np.zeros(opt.dim_cls_1)
	mean_classifier_2 = np.zeros(opt.dim_cls_2)
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

					if temp_modality == 'all':
						Local_Modality[user_id] = 3
					elif temp_modality == 'gps':
						Local_Modality[user_id] = 0
					elif temp_modality == 'lidar':
						Local_Modality[user_id] = 1
					elif temp_modality == 'image':
						Local_Modality[user_id] = 2
					print("client {} has modality {}".format(user_id, Local_Modality[user_id]))


				#if W message, server update for model aggregation
				elif mess_type == 0:

					server_start_time_record[user_id, iteration_count] = time.time()

					weights = pickle.loads(recv_data)
					print(weights.shape)

					if Local_Modality[user_id] == 3:
						ENC[user_id] = weights[0:opt.dim_enc_multi]
						CLS[user_id] = weights[opt.dim_enc_multi:]
					elif Local_Modality[user_id] == 0:
						ENC[user_id][0:opt.dim_enc_0] = weights[0:opt.dim_enc_0]## the encoder weights are saved on the head for single modality nodes
						CLS[user_id][0:opt.dim_cls_0] = weights[opt.dim_enc_0:]## the classifier weights are saved on the head for single modality nodes
					elif Local_Modality[user_id] == 1:
						ENC[user_id][0:opt.dim_enc_1] = weights[0:opt.dim_enc_1]## the classifier weights are saved on the head for single modality nodes
						CLS[user_id][0:opt.dim_cls_1] = weights[opt.dim_enc_1:]## the classifier weights are saved on the head for single modality nodes
					elif Local_Modality[user_id] == 2:
						ENC[user_id][0:opt.dim_enc_2] = weights[0:opt.dim_enc_2]## the classifier weights are saved on the head for single modality nodes
						CLS[user_id][0:opt.dim_cls_2] = weights[opt.dim_enc_2:]## the classifier weights are saved on the head for single modality nodes

						# print("error 1")
					try:
						barrier_W.wait(120)
					except Exception as e:
						print("wait W timeout...")

					if Local_Modality[user_id] == 3:
						send_weight = np.append(mean_encoder_all, mean_classifier_multi)
					elif Local_Modality[user_id] == 0:
						send_weight = np.append(mean_encoder_0, mean_classifier_0)
					elif Local_Modality[user_id] == 1:
						send_weight = np.append(mean_encoder_1, mean_classifier_1)
					elif Local_Modality[user_id] == 2:
						send_weight = np.append(mean_encoder_2, mean_classifier_2)

					mess_size = self.send2node(send_weight)

					# print(send_weight.shape)
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

	# HOST, PORT = "0.0.0.0", 9998
	HOST = "0.0.0.0"
	port_num = 9998


	if opt.modality_group == 0:
		port_num = 9999
	elif opt.modality_group == 1:
		port_num = 9998
	elif opt.modality_group == 2:
		port_num = 9997

	print(opt.modality_group, port_num)

	server = socketserver.ThreadingTCPServer((HOST,port_num),MyTCPHandler)
	# server.server_close()
	server.serve_forever(poll_interval = 0.5)
