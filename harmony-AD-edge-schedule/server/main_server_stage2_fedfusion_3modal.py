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
    parser.add_argument('--num_of_users', type=int, default=10,
                    help='num of users in FL')

    ## system
    parser.add_argument('--start_wait_time', type=int, default=300,
                    help='start_wait_time')
    parser.add_argument('--W_wait_time', type=int, default=1200,
                    help='W_wait_time')
    parser.add_argument('--end_wait_time', type=int, default=1200,
                    help='end_wait_time')

    ## model
    parser.add_argument('--num_of_modality', type=int, default=3,
                    help='num of modality in model')
    parser.add_argument('--num_of_group', type=int, default=3,
                    help='num of group for nodes')
    parser.add_argument('--dim_enc_0', type=int, default = 1496256)
    parser.add_argument('--dim_enc_1', type=int, default = 2221056)
    parser.add_argument('--dim_enc_2', type=int, default = 626240)
    parser.add_argument('--dim_enc_multi', type=int, default = 4343552)
    parser.add_argument('--dim_cls_0', type=int, default = 2651)
    parser.add_argument('--dim_cls_1', type=int, default = 2827)
    parser.add_argument('--dim_cls_2', type=int, default = 3531)
    parser.add_argument('--dim_cls_multi', type=int, default = 8987)
    parser.add_argument('--dim_all_0', type=int, default = 1498907)
    parser.add_argument('--dim_all_1', type=int, default = 2223883)
    parser.add_argument('--dim_all_2', type=int, default = 629771)
    parser.add_argument('--dim_all_multi', type=int, default=4352539)


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
group_index = np.zeros(opt.num_of_users).astype(int)

group_record = np.zeros((opt.fl_round, opt.num_of_users))
nor_enc_dis = np.zeros((opt.fl_round, opt.num_of_users, opt.num_of_modality))

wait_time_record = np.zeros(opt.fl_round)
aggregation_time_record = np.zeros(opt.fl_round)
server_start_time_record = np.zeros((opt.num_of_users, opt.fl_round))


# for multimodal nodes
def temp_to_user(temp_id, local_modality):

	reorder_array = np.loadtxt("reorder_id_stage2.txt").astype(int)
	user_id = reorder_array[temp_id, 1]

	return user_id

def encoder_bias_group(opt, encoder_dis):

	print("original encoder distance:", encoder_dis)

	for modality_id in range(opt.num_of_modality):
		max_dis = np.max(encoder_dis[:, modality_id])
		if max_dis != 0:
			encoder_dis[:, modality_id] = encoder_dis[:, modality_id] / max_dis

	print("normalized encoder distance:", encoder_dis)

	kmeans = KMeans(n_clusters=opt.num_of_group, random_state=0).fit(encoder_dis)

	print(kmeans.labels_)

	return kmeans.labels_, encoder_dis


def mmFedavg_classifier(opt, classifier, encoder_dis):

	global group_index

	group_index, normalized_encoder_dis = encoder_bias_group(opt, encoder_dis)
	print("group_index:", group_index)

	mean_classifier_all = np.zeros((opt.num_of_group, opt.dim_cls_multi))
	count_user = np.zeros(opt.num_of_group)

	for user_id in range(opt.num_of_users):

		group_id = int(group_index[user_id])#group_index(opt, user_id)
		mean_classifier_all[group_id] += classifier[user_id]
		count_user[group_id] += 1

	for group_id in range(opt.num_of_group):

		if count_user[group_id] != 0:
			mean_classifier_all[group_id] = mean_classifier_all[group_id] / count_user[group_id]

	print("count group: {}".format(count_user))


	return mean_classifier_all, normalized_encoder_dis



def server_update():
	
	global opt, CLS, Local_Modality, iteration_count
	global mean_classifier_multi, encoder_dis_record

	global aggregation_time_record, wait_time_record, server_start_time_record
	global group_index, group_record, nor_enc_dis

	aggregate_time1 = time.time()
	wait_time_record[iteration_count] = aggregate_time1 - np.min(server_start_time_record[:, iteration_count])
	print("server wait time:", wait_time_record[iteration_count])

	## mmFedavg for classifiers
	print("Iteration {}: mmFedavg of classifiers".format(iteration_count))
	mean_classifier_multi, encoder_dis = mmFedavg_classifier(opt, CLS, encoder_dis_record)

	group_record[iteration_count] = group_index
	nor_enc_dis[iteration_count] = encoder_dis

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

	global mean_classifier_multi, encoder_dis_record, group_index, group_record, nor_enc_dis

	opt.save_folder = "./save_fedfuse_server_group_{}".format(opt.num_of_group) + "/"
	if not os.path.isdir(opt.save_folder):
	    os.makedirs(opt.save_folder)

	np.savetxt(opt.save_folder + "group_record.txt", group_record)
	np.savetxt(opt.save_folder + "nor_enc_dis.txt", nor_enc_dis)
	np.savetxt(os.path.join(save_folder, "aggregation_time_record.txt"), aggregation_time_record)
	np.savetxt(os.path.join(save_folder, "wait_time_record.txt"), wait_time_record)
	np.savetxt(os.path.join(save_folder, "server_start_time_record.txt"), server_start_time_record)
	
	opt = parse_option()
	NUM_OF_WAIT = opt.num_of_users

	mean_classifier_multi = np.zeros((opt.num_of_group, opt.dim_cls_multi))
	encoder_dis_record = np.zeros((opt.num_of_users, opt.num_of_modality))
	group_index = np.zeros(opt.num_of_users)

	group_record = np.zeros((opt.fl_round, opt.num_of_users))
	nor_enc_dis = np.zeros((opt.fl_round, opt.num_of_users, opt.num_of_modality))

	wait_time_record = np.zeros(opt.fl_round)
	aggregation_time_record = np.zeros(opt.fl_round)
	server_start_time_record = np.zeros((opt.num_of_users, opt.fl_round))

	CLS = np.zeros((opt.num_of_users, opt.dim_cls_multi))
	Update_Flag = np.ones(opt.num_of_users)
	Local_Modality = np.zeros(opt.num_of_users).astype(int)

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
				# u_id = self.request.recv(4)
				# user_id = int(struct.unpack('i',u_id)[0])

				#receive the id of client
				u_id = self.request.recv(4)
				temp_id = struct.unpack('i',u_id)
				user_id = temp_to_user(int(temp_id[0]), 3)
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
						barrier_start.wait(100)
					except Exception as e:
						print("wait W timeout...")
						print(e)

					temp_modality = pickle.loads(recv_data)

					if temp_modality == 'all':
						Local_Modality[user_id] = 3
					print("client {} has modality {}".format(user_id, Local_Modality[user_id]))

				elif mess_type == 2:

					temp_dis = pickle.loads(recv_data)

					encoder_dis_record[user_id] = temp_dis

					print("encoder distance of user {}: ".format(user_id), temp_dis)
					
				#if W message, server update for model aggregation
				elif mess_type == 0:

					server_start_time_record[user_id, iteration_count] = time.time()
					weights = pickle.loads(recv_data)
					print(weights.shape)

					if Local_Modality[user_id] == 3:
						CLS[user_id] = weights

					try:
						barrier_W.wait(1200)
					except Exception as e:
						print("wait W timeout...")
						print(e)

					if Local_Modality[user_id] == 3:
						temp_group = group_index[user_id]
						send_weight = mean_classifier_multi[temp_group]
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
						barrier_end.wait(50)
					except Exception as e:
						print("finish timeout...")
					break


			except Exception as e:
				print('err',e)
				break



if __name__ == "__main__":
	HOST, PORT = "0.0.0.0", 9995
	server = socketserver.ThreadingTCPServer((HOST,PORT),MyTCPHandler)
	# server.server_close()
	server.serve_forever(poll_interval = 0.5)
