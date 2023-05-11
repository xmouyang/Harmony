import socketserver
import pickle, struct
import os
import sys
import argparse
import time
from threading import Lock, Thread
import threading
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# np.set_printoptions(threshold=np.inf)


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    ## FL
    parser.add_argument('--fl_round', type=int, default=11,
                    help='communication to server after the epoch of local training')
    parser.add_argument('--num_of_users', type=int, default=6,
                    help='num of users in FL')
    parser.add_argument('--encoder', type=str, default='fedavg',
                        choices=['fedprox', 'fedavg', 'PerAvg', 'pFedMe'], help='encoder')

    ## model
    parser.add_argument('--num_of_modality', type=int, default=2,
                    help='num of modality in model')
    parser.add_argument('--num_of_group', type=int, default=2,
                    help='num of group for nodes')
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
CLS = np.zeros((opt.num_of_users, opt.dim_cls_multi))

Update_Flag = np.ones(opt.num_of_users)
Local_Modality = np.zeros(opt.num_of_users).astype(int)

conver_indicator = 1e5

mean_classifier_multi = np.zeros((opt.num_of_group, opt.dim_cls_multi))
encoder_dis_record = np.zeros((opt.num_of_users, opt.num_of_modality))
group_index = np.zeros(opt.num_of_users)

group_record = np.zeros((opt.fl_round, opt.num_of_users))
nor_enc_dis = np.zeros((opt.fl_round, opt.num_of_users, opt.num_of_modality))

# def group_index(opt, user_id):

# 	## iid, 4,5,5
# 	group_set = [[3], [0,1, 2, 4,5]]

# 	for group_id in range(opt.num_of_group):

# 		if user_id in group_set[group_id]:
# 			return group_id


# for multimodal nodes
def temp_to_user(temp_id, local_modality):

	reorder_array = np.loadtxt("reorder_id_stage2.txt").astype(int)
	user_id = reorder_array[temp_id, 1]

	return user_id


def encoder_bias_group(opt, encoder_dis):

	print("original encoder distance:", encoder_dis)

	for modality_id in range(opt.num_of_modality):
		max_dis = np.max(encoder_dis[:, modality_id])
		encoder_dis[:, modality_id] = encoder_dis[:, modality_id] / max_dis

	print("normalized encoder distance:", encoder_dis)

	kmeans = KMeans(n_clusters=opt.num_of_group, random_state=0).fit(encoder_dis)
	print(kmeans.labels_)

	return kmeans.labels_

def mmFedavg_classifier(opt, classifier, encoder_dis):

	global group_index

	group_index = encoder_bias_group(opt, encoder_dis)

	mean_classifier_all = np.zeros((opt.num_of_group, opt.dim_cls_multi))
	count_user = np.zeros(opt.num_of_group)

	for user_id in range(opt.num_of_users):

		group_id = group_index[user_id]#group_index(opt, user_id)
		mean_classifier_all[group_id] += classifier[user_id]
		count_user[group_id] += 1

	for group_id in range(opt.num_of_group):

		if count_user[group_id] != 0:
			mean_classifier_all[group_id] = mean_classifier_all[group_id] / count_user[group_id]

	print("count_modality {}".format(count_user))
	# print("mean_classifiers {}".format(mean_classifier_all))
	# print("averaged mean_classifiers: ", np.mean(mean_classifier_all))


	return mean_classifier_all, encoder_dis


def weight_dis_matrix(opt, weight):

	dis_matrix = np.zeros((opt.num_of_users, opt.num_of_users))

	# print("debug here 4")
	dis_matrix = cosine_similarity(weight)
	# print("debug here 5")


	return dis_matrix


def server_update():
	
	global opt, CLS, iteration_count
	global mean_classifier_multi, encoder_dis_record

	global opt, group_index, group_record, nor_enc_dis

	# classifier_dis = weight_dis_matrix(opt, CLS)
	# np.savetxt(opt.save_folder + "cls_dis_{}.txt".format(iteration_count), classifier_dis)
	# print("Iteration {}: distance of classifiers".format(iteration_count), classifier_dis)

	## mmFedavg for classifiers
	print("Iteration {}: mmFedavg of classifiers".format(iteration_count))
	mean_classifier_multi, encoder_dis = mmFedavg_classifier(opt, CLS, encoder_dis_record)

	group_record[iteration_count] = group_index
	nor_enc_dis[iteration_count] = encoder_dis
	# np.savetxt(opt.save_folder + "cls_dis_{}.txt".format(iteration_count), classifier_dis)

	iteration_count = iteration_count + 1
	print("iteration_count: ", iteration_count)

	
def reinitialize():

	global iteration_count, trial_count
	# trial_count += 1
	iteration_count = 0
	print("Trial: ", trial_count)

	global opt, NUM_OF_WAIT
	opt = parse_option()
	NUM_OF_WAIT = opt.num_of_users

	global CLS, Update_Flag, Local_Modality

	## recieved all model weights
	CLS = np.zeros((opt.num_of_users, opt.dim_cls_multi))

	Update_Flag = np.ones(opt.num_of_users)
	Local_Modality = np.zeros(opt.num_of_users).astype(int)

	global mean_classifier_multi, encoder_dis_record, group_index, group_record, nor_enc_dis

	opt.save_folder = "./save_server_{}_group_{}/".format(opt.encoder, opt.num_of_group)
	if not os.path.isdir(opt.save_folder):
	    os.makedirs(opt.save_folder)

	np.savetxt(opt.save_folder + "group_record.txt", group_record)
	np.savetxt(opt.save_folder + "nor_enc_dis.txt", nor_enc_dis)

	mean_classifier_multi = np.zeros((opt.num_of_group, opt.dim_cls_multi))
	encoder_dis_record = np.zeros((opt.num_of_users, opt.num_of_modality))
	group_index = np.zeros(opt.num_of_users)

	group_record = np.zeros((opt.fl_round, opt.num_of_users))
	nor_enc_dis = np.zeros((opt.fl_round, opt.num_of_users, opt.num_of_modality))

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
				user_id = temp_to_user(int(temp_id[0]), 3)

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
					elif temp_modality == 'gyr':
						Local_Modality[user_id] = 1
					print("client {} has modality {}".format(user_id[0], Local_Modality[user_id]))


				#if W message, server update for model aggregation

				elif mess_type == 2:
					temp_dis = pickle.loads(recv_data)

					encoder_dis_record[user_id] = temp_dis


				elif mess_type == 0:

					weights = pickle.loads(recv_data)
					print(weights.shape)

					if Local_Modality[user_id] == 2:
						CLS[user_id] = weights

					try:
						barrier_W.wait(120)
					except Exception as e:
						print("wait W timeout...")

					if Local_Modality[user_id] == 2:
						temp_group = group_index[user_id[0]]# group_index(opt, user_id[0])
						mess_size = self.send2node(mean_classifier_multi[temp_group])
					print("send New_W to client {} with the size of {}, and shape of {}".format(user_id[0],mess_size, mean_classifier_multi[temp_group].shape))


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
						# reinitialize()
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
