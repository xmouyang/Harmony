import socket
import pickle, struct
import sys
from threading import Lock, Thread
import threading


class COMM:
    def __init__(self, host, port, user_id):
        self.host = host
        self.port = port
        self.id = user_id
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((host,port))

    # the mess_type defines the content sent to server
    # -1 means start request
    # 0 means W
    # 1 means loss
    # 9 means straggler end connection
    # 10 means end connection
    def send2server(self,content,mess_type):
        data = pickle.dumps(content, protocol = 0)
        size = sys.getsizeof(data)

        header = struct.pack("i",size)
        u_id = struct.pack("i",self.id)
        mess_type = struct.pack("i",mess_type)

        self.client.sendall(header)
        self.client.sendall(u_id)
        self.client.sendall(mess_type)
        self.client.sendall(data)

    def recvfserver(self):
        header = self.client.recv(4)
        size = struct.unpack('i',header)

        recv_data = b""
        while sys.getsizeof(recv_data)<size[0]:
            recv_data += self.client.recv(size[0]-sys.getsizeof(recv_data))

        data = recv_data
        # print("the received data size is: {}, while the size of notice is: {}".format(sys.getsizeof(data),size[0]))
        data = pickle.loads(data)

        return data

    def recvOUF(self):

        #receive new weight from server
        header = self.client.recv(4)
        size = struct.unpack('i',header)

        recv_data = b""
        while sys.getsizeof(recv_data)<size[0]:
            recv_data += self.client.recv(size[0]-sys.getsizeof(recv_data))

        # print(recv_data, size)
        data = recv_data
        new_w = pickle.loads(data)


        ## receive time and ratio
        header = self.client.recv(4)
        size = struct.unpack('i',header)

        recv_data = b""
        while sys.getsizeof(recv_data)<size[0]:
            recv_data += self.client.recv(size[0]-sys.getsizeof(recv_data))
        data = recv_data
        time_record2node = pickle.loads(data)

        print("time_record2node:", time_record2node)
        send_system_time = time_record2node[0]
        send_node_ratio = time_record2node[1]


        # receive stop signal
        sig_stop = self.client.recv(4)
        sig_stop = struct.unpack('i',sig_stop)[0]


        # send_system_time = self.client.recv(4)
        # send_system_time = struct.unpack('f',send_system_time)[0]
        # print("send_system_time:", send_system_time)

        # send_node_ratio = self.client.recv(4)
        # send_node_ratio = struct.unpack('f',send_node_ratio)[0]
        # print("send_node_ratio:", send_node_ratio)

        return new_w, sig_stop, send_system_time, send_node_ratio


    def disconnect(self, type):
        if type==1:
            self.send2server('end',10)
        elif type==0:
            self.send2server('end',9)
        self.client.close()
