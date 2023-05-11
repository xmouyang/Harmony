from __future__ import print_function

import os
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model

from model import MySingleModel, My3Model, My2Model
import data_pre as data
import numpy as np
from sklearn.model_selection import train_test_split
from communication import COMM
import random

import copy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import torch.optim as optim

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--usr_id', type=int, default=0,
                        help='user id')
    parser.add_argument('--local_modality', type=str, default='audio',
                        choices=['audio', 'depth', 'radar', 'all', 'AD', 'DR', 'AR'], help='local_modality')
    parser.add_argument('--server_address', type=str, default='10.54.20.14',
                        help='server_address')
    parser.add_argument('--fl_epoch', type=int, default=10,
                    help='communication to server after the epoch of local training')
    
    
    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')

    # model dataset
    parser.add_argument('--model', type=str, default='MyMMmodel')
    parser.add_argument('--dataset', type=str, default='AD',
                        choices=['MHAD', 'FLASH', 'AD'], help='dataset')
    parser.add_argument('--num_class', type=int, default=11,
                        help='num_class')
    parser.add_argument('--num_of_train', type=int, default=50,
                        help='num_of_train')
    parser.add_argument('--num_of_test', type=int, default=200,
                        help='num_of_test')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

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

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.model_path = './save_uniFL/{}_models/'.format(opt.dataset)
    opt.result_path = './save_fedfuse_results/node_{}/{}_results/'.format(opt.usr_id, opt.local_modality)


    opt.load_folder = os.path.join(opt.model_path)

    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)


    return opt



def set_loader(opt):

    # load labeled train and test data
    print("train labeled data:")

    if opt.local_modality in ["all", 'AD', 'DR', 'AR']:
        total_dataset = data.Multimodal_dataset(opt.usr_id) 
    else:
        total_dataset = data.Unimodal_dataset(opt.usr_id, opt.local_modality)
    

    sample_index_path = "./sample_index/node_{}/".format(opt.usr_id)
    all_train_sample_index = np.loadtxt(sample_index_path + "train_sample_index.txt")
    all_test_sample_index = np.loadtxt(sample_index_path + "test_sample_index.txt")


    if len(all_train_sample_index) < opt.num_of_train:
        opt.num_of_train = len(all_train_sample_index)
    if len(all_test_sample_index) < opt.num_of_test:
        opt.num_of_test = len(all_test_sample_index)

    train_sample_index = all_train_sample_index[0:opt.num_of_train]
    test_sample_index = all_test_sample_index[0:opt.num_of_test]
    print(len(train_sample_index))
    print(len(test_sample_index))

    train_sample_index = list(map(int, train_sample_index))#np.array(train_sample_index).
    test_sample_index = list(map(int, test_sample_index))

    train_sample = torch.utils.data.SubsetRandomSampler(train_sample_index, generator=torch.Generator().manual_seed(42))
    test_sample = torch.utils.data.SubsetRandomSampler(test_sample_index, generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(
        total_dataset, batch_size=opt.batch_size, sampler=train_sample,
        num_workers=opt.num_workers, 
        pin_memory=True, 
        # shuffle=True, 
        drop_last = True)
    val_loader = torch.utils.data.DataLoader(
        total_dataset, batch_size=opt.batch_size, sampler=test_sample,
        num_workers=opt.num_workers, 
        pin_memory=True, 
        # shuffle=True, 
        drop_last = True)
    
    return train_loader, val_loader




def load_single_model(opt, modality):

    ckpt_path = opt.load_folder + "/last_" + modality + ".pth"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    # if torch.cuda.is_available():
    #     # if torch.cuda.device_count() <= 1:
    #     new_state_dict = {}
    #     for k, v in state_dict.items():
    #         k = k.replace("module.", "")
    #         new_state_dict[k] = v
    #     state_dict = new_state_dict

    return state_dict


def set_model(opt):

    criterion = torch.nn.CrossEntropyLoss()

    if opt.local_modality == "all":
        model = My3Model(num_classes=opt.num_class)
        state_dict_1 = load_single_model(opt, "audio")
        state_dict_2 = load_single_model(opt, "depth")
        state_dict_3 = load_single_model(opt, "radar")
        model.encoder.encoder_1.load_state_dict(state_dict_1)
        model.encoder.encoder_2.load_state_dict(state_dict_2)
        model.encoder.encoder_3.load_state_dict(state_dict_3)
    elif opt.local_modality in ['AD', 'DR', 'AR']:
        model = My2Model(num_classes = opt.num_class, modality = opt.local_modality)
        if opt.local_modality == "AD":
            state_dict_1 = load_single_model(opt, "audio")
            state_dict_2 = load_single_model(opt, "depth")    
        elif opt.local_modality == "DR":  
            state_dict_1 = load_single_model(opt, "depth")
            state_dict_2 = load_single_model(opt, "radar")
        elif opt.local_modality == "AR":  
            state_dict_1 = load_single_model(opt, "audio")
            state_dict_2 = load_single_model(opt, "radar")  
        model.encoder.encoder_1.load_state_dict(state_dict_1)
        model.encoder.encoder_2.load_state_dict(state_dict_2) 

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion



def train_multi(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    end = time.time()
    for idx, (input_data1, input_data2, input_data3, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            input_data2 = input_data2.cuda()
            input_data3 = input_data3.cuda()
            labels = labels.cuda()
        bsz = input_data1.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        if opt.local_modality == "all":
            output = model(input_data1, input_data2, input_data3)
        elif opt.local_modality == "AD":
            output = model(input_data1, input_data2)
        elif opt.local_modality == "DR":
            output = model(input_data2, input_data3)
        elif opt.local_modality == "AR":
            output = model(input_data1, input_data3)
        loss = criterion(output, labels)

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        # update metric
        losses.update(loss.item(), bsz)
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg



def validate_multi(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    confusion = np.zeros((opt.num_class, opt.num_class))

    with torch.no_grad():
        end = time.time()
        for idx, (input_data1, input_data2, input_data3, labels) in enumerate(val_loader):

            if torch.cuda.is_available():
                input_data1 = input_data1.float().cuda()
                input_data2 = input_data2.float().cuda()
                input_data3 = input_data3.float().cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            if opt.local_modality == "all":
                output = model(input_data1, input_data2, input_data3)
            elif opt.local_modality == "AD":
                output = model(input_data1, input_data2)
            elif opt.local_modality == "DR":
                output = model(input_data2, input_data3)
            elif opt.local_modality == "AR":
                output = model(input_data1, input_data3)
            loss = criterion(output, labels)

            # update metric
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), bsz)
            top1.update(acc1[0], bsz)

            # calculate and store confusion matrix
            rows = labels.cpu().numpy()
            cols = output.max(1)[1].cpu().numpy()
            for label_index in range(labels.shape[0]):
                confusion[rows[label_index], cols[label_index]] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    return losses.avg, top1.avg, confusion



def get_model_array(model):

    params = []
    for param in model.parameters():
        if torch.cuda.is_available():
            params.extend(param.view(-1).cpu().detach().numpy())
        else:
            params.extend(param.view(-1).detach().numpy())

    # model_params = params.cpu().numpy()
    model_params = np.array(params)
    print("Shape of model weight: ", model_params.shape)#39456

    return model_params


def reset_model_parameter(new_params, model):

    temp_index = 0

    with torch.no_grad():
        for param in model.parameters():

            # print(param.shape)

            if len(param.shape) == 2:

                para_len = int(param.shape[0] * param.shape[1])
                temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight.reshape(param.shape[0], param.shape[1])))
                temp_index += para_len

            elif len(param.shape) == 3:

                para_len = int(param.shape[0] * param.shape[1] * param.shape[2])
                temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2])))
                temp_index += para_len 

            elif len(param.shape) == 4:

                para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3])
                temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3])))
                temp_index += para_len  

            elif len(param.shape) == 5:

                para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3] * param.shape[4])
                temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3], param.shape[4])))
                temp_index += para_len  

            else:

                para_len = param.shape[0]
                temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight))
                temp_index += para_len



def set_commu(opt):

    #prepare the communication module
    server_addr = opt.server_address
    server_port = 9995

    ## set1
    current_user = opt.usr_id# - 6
    comm = COMM(server_addr, server_port, current_user)

    comm.send2server('hello',-1)

    print(comm.recvfserver())

    comm.send2server(opt.local_modality ,1)

    return comm

def calculate_dis(model, init_model, opt):

    enc_1_dis = 0.0
    enc_2_dis = 0.0
    cls_dis = 0.0
    len_idx = 0

    # iterate through the current and global model parameters
    for w, w_t in zip(init_model.parameters(), model.parameters()) :

        if len_idx < opt.dim_enc_0:
            enc_1_dis += (w-w_t).norm(2).cpu().detach().numpy()
        elif len_idx >= opt.dim_enc_0 and len_idx < opt.dim_enc_multi:
            enc_2_dis += (w-w_t).norm(2).cpu().detach().numpy()
        else:
            cls_dis += (w-w_t).norm(2).cpu().detach().numpy()

        temp_shape = w.view(-1).shape[0]
        len_idx += temp_shape

        # print("add distance:",temp_shape, len_idx)

    return enc_1_dis, enc_2_dis, cls_dis


def calculate_cos_dis(model, init_model, opt):

    enc_1_dis = 0.0
    enc_2_dis = 0.0
    enc_3_dis = 0.0
    cls_dis = 0.0

    model_weight = get_model_array(model)
    init_model_weight = get_model_array(init_model)

    enc_1_dis = distance.cosine(model_weight[0:opt.dim_enc_0].reshape(1, -1), init_model_weight[0:opt.dim_enc_0].reshape(1, -1))

    start_index = opt.dim_enc_0
    end_index = opt.dim_enc_0 + opt.dim_enc_1
    enc_2_dis = distance.cosine(model_weight[start_index:end_index].reshape(1, -1), init_model_weight[start_index:end_index].reshape(1, -1))

    start_index = opt.dim_enc_0 + opt.dim_enc_1
    end_index = opt.dim_enc_multi
    enc_3_dis = distance.cosine(model_weight[start_index:end_index].reshape(1, -1), init_model_weight[start_index:end_index].reshape(1, -1))

    cls_dis = distance.cosine(model_weight[opt.dim_enc_multi:].reshape(1, -1), init_model_weight[opt.dim_enc_multi:].reshape(1, -1))

    return enc_1_dis, enc_2_dis, enc_3_dis, cls_dis



def main():

    opt = parse_option()

    torch.manual_seed(42)

    # set up communication with sevrer
    comm = set_commu(opt)

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # global_model = copy.deepcopy(model)
    init_model = copy.deepcopy(model)
    w_parameter_init = get_model_array(model.classifier)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # build optimizer for feature extractor and classifier
    optimizer = optim.SGD([ 
                {'params': model.encoder.parameters(), 'lr': 1e-4},   # 0
                {'params': model.classifier.parameters(), 'lr': opt.learning_rate}],
                momentum=opt.momentum,
                weight_decay=opt.weight_decay)


    best_acc = 0
    best_confusion = np.zeros((opt.num_class, opt.num_class))
    record_loss = np.zeros(opt.epochs)
    record_acc = np.zeros(opt.epochs)

    record_enc_1 = np.zeros(opt.epochs)
    record_enc_2 = np.zeros(opt.epochs)
    record_enc_3 = np.zeros(opt.epochs)
    record_cls = np.zeros(opt.epochs)

    compute_time_record = np.zeros(opt.epochs)
    commu_time_record = np.zeros(int(opt.epochs/opt.fl_epoch))
    all_time_record = np.zeros(opt.epochs + 2)

    temp_encoder_dis = np.zeros(3)

    all_time_record[0] = time.time()

    # training routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train_multi(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        record_loss[epoch-1] = loss

        compute_time_record[epoch-1] = time2 - time1
        all_time_record[epoch] = time.time()

        # evaluation
        loss, val_acc, confusion = validate_multi(val_loader, model, criterion, opt)
        record_acc[epoch-1] = val_acc
        if val_acc > best_acc:
             best_acc = val_acc
             best_confusion = confusion

        encoder_1_dis, encoder_2_dis, encoder_3_dis, classifier_dis = calculate_cos_dis(model, init_model, opt)
        record_enc_1[epoch-1] = encoder_1_dis
        record_enc_2[epoch-1] = encoder_2_dis
        record_enc_3[epoch-1] = encoder_3_dis
        record_cls[epoch-1] = classifier_dis
        print("Node {} distance:".format(opt.usr_id), encoder_1_dis, encoder_2_dis, encoder_3_dis, classifier_dis)

        temp_encoder_dis[0] = encoder_1_dis
        temp_encoder_dis[1] = encoder_2_dis
        temp_encoder_dis[2] = encoder_3_dis

        # # communication with the server every fl_epoch 
        if (epoch % opt.fl_epoch) == 0:

            print("Node {} sends weight to the server:".format(opt.usr_id))
            w_parameter = get_model_array(model.classifier) #obtain the model parameters or gradients 
            w_update = w_parameter - w_parameter_init

            comm_time1 = time.time()
            comm.send2server(temp_encoder_dis, 2)
            comm.send2server(w_update,0)

            new_w_update, sig_stop = comm.recvOUF()

            comm_time2 = time.time()
            commu_epoch = int(epoch/opt.fl_epoch - 1)
            commu_time_record[commu_epoch] = comm_time2 - comm_time1

            print("Received weight from the server:", new_w_update.shape)
            print("Received signal from the server:", sig_stop)

            ## only update fusion and classifier
            w_parameter += new_w_update
            reset_model_parameter(w_parameter, model.classifier)

            ## record current model weight
            w_parameter_init = w_parameter


    # evaluation
    loss, val_acc, confusion = validate_multi(val_loader, model, criterion, opt)

    print("Testing accuracy of node {} is : {}".format(opt.usr_id, val_acc))
    np.savetxt(opt.result_path + "record_loss.txt", record_loss)
    np.savetxt(opt.result_path + "record_acc.txt", record_acc)
    np.savetxt(opt.result_path + "record_confusion.txt", confusion)
    np.savetxt(opt.result_path + "record_enc_1_dis.txt", record_enc_1)
    np.savetxt(opt.result_path + "record_enc_2_dis.txt", record_enc_2)
    np.savetxt(opt.result_path + "record_enc_3_dis.txt", record_enc_3)
    np.savetxt(opt.result_path + "record_cls_dis.txt", record_cls)

    np.savetxt(opt.result_path + "compute_time_record.txt", compute_time_record)
    np.savetxt(opt.result_path + "commu_time_record.txt", commu_time_record)
    np.savetxt(opt.result_path + "all_time_record.txt", all_time_record)

    comm.disconnect(1)


if __name__ == '__main__':
    main()
