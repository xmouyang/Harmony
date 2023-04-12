import logging
# multi processes instead of multi threads
import multiprocessing as mp
from multiprocessing import SimpleQueue, Process, Condition, Pipe
import psutil
import os
import pickle
import socket
import time
import argparse
import torch
import numpy as np 

from main_single_modality import exec_unimodal_training


def parse_global_option():
    parser = argparse.ArgumentParser('argument for harmony run')
    parser.add_argument('--usr_id', type=int, default=0,
                        help='user id')
    parser.add_argument('--modality', nargs='*', default=['audio'], choices = ['audio', 'depth', 'radar'],
                    help='choose the modalities you want to use')
    parser.add_argument('--time_slice', type=int, default = 100, help='set the total time slice')
    opt = parser.parse_args()

    # set the path according to the environment
    opt.result_path = './save_unifl_results/node_{}/{}_results/'.format(opt.usr_id, opt.local_modality)

    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)

    return opt

def normalize(a):
    return list(np.array(a)/sum(a))

def main():

    opt = parse_global_option()

    # the number of modality is the length of modality parameters
    num_of_modality = len(opt.modality)

    total_time_slice = opt.time_slice
    print("total_time_slice is: ", total_time_slice)

    # prepare the list of processes and necessary variables
    connWrite = []
    connRead = []

    process = []
    pid_list = []
    # time_slice = [1] * num_of_modality


    time_slice = [30, 20, 5]
    time_ratio = np.zeros((num_of_modality,2))

    FL_round = 22
    round_id = 0
    time_slice_record = np.zeros((FL_round, num_of_modality))
    time_slice_record[0] = time_slice

    # record which modality has received the ratio from server
    data_arrive = [0] * num_of_modality
    # record which modality has finished this epoch, 1 means not done, 0 means done
    epoch_done = [1] * num_of_modality
    # record which modality has finished all epochs, 1 means not done, 0 means done
    final_done = [1] * num_of_modality

    event = mp.Event()

    for i in range(num_of_modality):
        pwrite, pread = Pipe()
        connRead.append(pread)
        connWrite.append(pwrite)

    for i in range(num_of_modality):
        train_process = mp.Process(target=exec_unimodal_training, args=(opt.modality[i], connWrite[i], event, opt.usr_id))
        process.append(train_process)

    for i in range(num_of_modality):
        process[i].start()
        control_process = psutil.Process(process[i].pid)
        pid_list.append(control_process)


    # Start the scheduler
    while True:
        # listen to the pipe to get the ratio sent from server
        for i in range(num_of_modality):
            if connRead[i].poll():
                while connRead[i].poll():
                    ret = connRead[i].recv()
                    if type(ret)==list:
                        data_arrive[i] = 1
                        # epoch_done[i] = 1
                        print("DEBUG: the data arrive situation is: ", data_arrive)
                        time_ratio[i][0] = ret[0]
                        time_ratio[i][1] = ret[1]
                        print("Yeah! I got time and ratio from subprocess: ", ret)
                    elif type(ret)==str:
                        # print("DEBUG: the done signal is: ", ret)
                        epoch_done[i] = 0

                if sum(data_arrive) == sum(final_done):
                    beta1 = time_ratio[0][0] / (time_ratio[0][0] + time_ratio[1][0] + time_ratio[2][0]) * time_ratio[0][1]
                    beta2 = time_ratio[1][0] / (time_ratio[0][0] + time_ratio[1][0] + time_ratio[2][0]) * time_ratio[1][1]
                    beta3 = time_ratio[2][0] / (time_ratio[0][0] + time_ratio[1][0] + time_ratio[2][0]) * time_ratio[2][1]
                    [ratio1, ratio2, ratio3] = normalize([beta1, beta2, beta3])
                    print("ratio1: ", ratio1)
                    print("ratio2: ", ratio2)
                    print("ratio3: ", ratio3)
                    renew_time_slice_1 = time_slice[0] * ratio1
                    renew_time_slice_2 = time_slice[1] * ratio2
                    renew_time_slice_3 = time_slice[2] * ratio3
                    [time_slice[0], time_slice[1], time_slice[2]] = normalize([renew_time_slice_1, renew_time_slice_2, renew_time_slice_3])
                    print("time_slice[0]: ", time_slice[0])
                    print("time_slice[1]: ", time_slice[1])
                    print("time_slice[2]: ", time_slice[2])
                    round_id = round_id + 1
                    time_slice_record[round_id] = time_slice
                    data_arrive = [0] * num_of_modality
                    epoch_done = [1] * num_of_modality

                    np.savetxt(opt.result_path+"time_slice_record.txt", time_slice_record)

            

        # run process 1
        if epoch_done[0] == 1 and final_done[0] == 1:
            print("Run the first modality!!!!!!!!!!!!!!!!!!!!")
            if process[0].is_alive():
                pid_list[0].resume()
            if process[1].is_alive():
                pid_list[1].suspend()
            if process[2].is_alive():
                pid_list[2].suspend()
            # control_process3.suspend()
            event.wait(time_slice[0] * total_time_slice / 
                np.sum(np.multiply(np.multiply(np.array(time_slice), np.array(epoch_done)), np.array(final_done))))
            event.clear()
        # if has done training, while not receving the data
        if epoch_done[0] == 0 and data_arrive[0] == 0:
            print("Run the short version first modality!!!!!!!!!!!!!!!!!!!!")
            if process[0].is_alive():
                pid_list[0].resume()
            if process[1].is_alive():
                pid_list[1].suspend()
            if process[2].is_alive():
                pid_list[2].suspend()
            # control_process3.suspend()
            event.wait(1)
            event.clear()


        # run process 2
        if epoch_done[1] == 1 and final_done[1] == 1:
            print("Run the second modality!!!!!!!!!!!!!!!!!!!!")
            if process[0].is_alive():
                pid_list[0].suspend()
            if process[1].is_alive():
                pid_list[1].resume()
            if process[2].is_alive():
                pid_list[2].suspend()
            event.wait(time_slice[1] * total_time_slice / 
                np.sum(np.multiply(np.multiply(np.array(time_slice),np.array(epoch_done)),np.array(final_done))))
            event.clear()

        if epoch_done[1] == 0 and data_arrive[1] == 0:
            print("Run the short version second modality!!!!!!!!!!!!!!!!!!!!")
            if process[0].is_alive():
                pid_list[0].suspend()
            if process[1].is_alive():
                pid_list[1].resume()
            if process[2].is_alive():
                pid_list[2].suspend()
            event.wait(1)
            event.clear()

        # run process 3
        if epoch_done[2] == 1 and final_done[2] == 1:
            print("Run the third modality!!!!!!!!!!!!!!!!!!!!")
            if process[0].is_alive():
                pid_list[0].suspend()
            if process[1].is_alive():
                pid_list[1].suspend()
            if process[2].is_alive():
                pid_list[2].resume()

            event.wait(time_slice[2] * total_time_slice / 
                np.sum(np.multiply(np.multiply(np.array(time_slice),np.array(epoch_done)),np.array(final_done))))
            event.clear()
            
        if epoch_done[2] == 0 and data_arrive[2] == 0:
            print("Run the short version third modality!!!!!!!!!!!!!!!!!!!!")
            if process[0].is_alive():
                pid_list[0].suspend()
            if process[1].is_alive():
                pid_list[1].suspend()
            if process[2].is_alive():
                pid_list[2].resume()
                
            event.wait(1)
            event.clear()

        for i in range(num_of_modality):
            if process[i].is_alive() == False:
                final_done[i] = 0

        if sum(final_done) == 1:
            for i in range(num_of_modality):
                if process[i].is_alive():
                    process[i].resume()

        print("DEBUG: Current final_done is: ", final_done)

        if sum(final_done) == 0:
            break



if __name__ == "__main__":
    main()
