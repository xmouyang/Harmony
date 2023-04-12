from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model

from model import MySingleModel, My2Model, My3Model
import data_pre as data
from communication import COMM
import random

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

############################
# Fix the seed
############################
seed = 0
os.environ['PYTHONHASHSEED']=str(seed)
np.random.seed(seed)
random.seed(seed)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # FL
    parser.add_argument('--id_gpu', default=1, type=int, 
        help='which gpu to use.')
    parser.add_argument('--usr_id', type=int, default=20,
                        help='user id')
    parser.add_argument('--fl_epoch', type=int, default=5,
                    help='communication to server after the epoch of local training')
    parser.add_argument('--server_address', type=str, default="10.54.20.12",
                        help='server_address')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--local_modality', type=str, default='gps',
        choices = ['gps', 'lidar', 'image', 'g+l', 'g+i', 'l+i', 'all'], help='which data to use as input')

    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')


    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,100,150',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--strategy', type=str ,default='one_hot', 
        help='labeling strategy to use',choices=['baseline','one_hot','reg'])

    # model dataset
    parser.add_argument('--model', type=str, default='MyMMmodel')
    parser.add_argument('--approach', type=str, default='unifl')
    parser.add_argument('--dataset', type=str, default='FLASH',
                        choices=['MHAD', 'FLASH', 'ours'], help='dataset')
    parser.add_argument('--num_class', type=int, default=64,
                        help='num_class')
    parser.add_argument('--num_of_train', type=int, default=50,
                        help='num_of_train')
    parser.add_argument('--num_of_test', type=int, default=150,
                        help='num_of_test')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=int, default='1',
                        help='id for recording multiple runs')

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

    # set the path according to the environment
    opt.result_path = './save_uniFL_results/node_{}/{}_results/'.format(opt.usr_id, opt.dataset)

    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)

    return opt



def set_loader(opt):

    print('******************Load data from the specified usr_id *************************')

    if opt.local_modality == "all":

        x1_train, y_train = data.load_data('gps', opt.usr_id, opt.strategy, 0)
        x2_train, _ = data.load_data('lidar', opt.usr_id, opt.strategy, 0)
        x3_train, _ = data.load_data('image', opt.usr_id, opt.strategy, 0)

        x1_test, y_test = data.load_data('gps', opt.usr_id, opt.strategy, 1)
        x2_test, _ = data.load_data('lidar', opt.usr_id, opt.strategy, 1)
        x3_test, _ = data.load_data('image', opt.usr_id, opt.strategy, 1)

        print(x1_test.shape)
        print(x1_train.shape)

        train_dataset = data.Multimodal_dataset(x1_train, x2_train, x3_train, y_train)
        test_dataset = data.Multimodal_dataset(x1_test, x2_test, x3_test, y_test)

    else:

        x_train, y_train = data.load_data(opt.local_modality, opt.usr_id, opt.strategy, 0)
        x_test, y_test = data.load_data(opt.local_modality, opt.usr_id, opt.strategy, 1)

        print(y_train)
        print(y_test)
        print(x_train.shape)
        print(x_test.shape)

        train_dataset = data.Unimodal_dataset(x_train,y_train)
        test_dataset = data.Unimodal_dataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, 
        pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, 
        pin_memory=True, shuffle=True)

    print('******************Succesfully generated the data*************************')

    return train_loader, test_loader


def set_model(opt):

    if opt.local_modality == "all":
        model = My3Model(num_classes = opt.num_class, modality = opt.local_modality)
    else:
        model = MySingleModel(num_classes = opt.num_class, modality = opt.local_modality)

    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion



def train_single(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (input_data1, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            labels = labels.cuda()
        bsz = input_data1.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(input_data1)
        loss = criterion(output, labels)

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        # update metric
        losses.update(loss.item(), bsz)
        top1.update(acc5[0], bsz)

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


def train_multi(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # label_list = []

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
        output = model(input_data1, input_data2, input_data3)
        loss = criterion(output, labels)

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        # update metric
        losses.update(loss.item(), bsz)
        top1.update(acc5[0], bsz)

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


def validate_single(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    confusion = np.zeros((opt.num_class, opt.num_class))

    with torch.no_grad():
        end = time.time()
        for idx, (input_data1, labels) in enumerate(val_loader):

            if torch.cuda.is_available():
                input_data1 = input_data1.float().cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(input_data1)
            loss = criterion(output, labels)

            # update metric
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), bsz)
            top1.update(acc5[0], bsz)

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
            output = model(input_data1, input_data2, input_data3)
            loss = criterion(output, labels)

            # update metric
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), bsz)
            top1.update(acc5[0], bsz)

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
        # print(param)

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
    server_addr = opt.server_address#

    if opt.local_modality == "gps":
        server_port = 9999
    elif opt.local_modality == "lidar":
        server_port = 9998
    elif opt.local_modality == "image":
        server_port = 9997

    # server_port = 9998
    comm = COMM(server_addr,server_port, opt.usr_id)

    comm.send2server('hello',-1)

    print(comm.recvfserver())

    comm.send2server(opt.local_modality ,1)

    return comm


def main():

    opt = parse_option()

    torch.manual_seed(42)

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    w_parameter_init = get_model_array(model)
        
    best_acc = 0
    best_confusion = np.zeros((opt.num_class, opt.num_class))
    record_loss = np.zeros(opt.epochs)
    record_acc = np.zeros(opt.epochs)

    # set up communication with sevrer
    comm = set_commu(opt)

    compute_time_record = np.zeros(opt.epochs)
    commu_time_record = np.zeros(int(opt.epochs/opt.fl_epoch))
    all_time_record = np.zeros(opt.epochs + 2)

    all_time_record[0] = time.time()

    # training routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        if opt.local_modality == "all":
            loss = train_multi(train_loader, model, criterion, optimizer, epoch, opt)
        else:
            loss = train_single(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        record_loss[epoch-1] = loss

        compute_time_record[epoch-1] = time2 - time1
        all_time_record[epoch] = time.time()

        # evaluation
        if opt.local_modality == "all":
            loss, val_acc, confusion = validate_multi(val_loader, model, criterion, opt)
        else:
            loss, val_acc, confusion = validate_single(val_loader, model, criterion, opt)
        record_acc[epoch-1] = val_acc
        if val_acc > best_acc:
             best_acc = val_acc
             best_confusion = confusion

        # # communication with the server every fl_epoch 
        if (epoch % opt.fl_epoch) == 0:

            ## send model update to the server
            print("Node {} sends weight to the server:".format(opt.usr_id))
            w_parameter = get_model_array(model) #obtain the model parameters or gradients 
            w_update = w_parameter - w_parameter_init

            comm_time1 = time.time()
            comm.send2server(w_update,0)

            ## recieve aggregated model update from the server
            new_w_update, sig_stop = comm.recvOUF()
            print("Received weight from the server:", new_w_update.shape)
            print("Received signal from the server:", sig_stop)

            comm_time2 = time.time()
            commu_epoch = int(epoch/opt.fl_epoch - 1)
            commu_time_record[commu_epoch] = comm_time2 - comm_time1

            ## update the model according to the received weights
            new_w = w_parameter_init + new_w_update
            reset_model_parameter(new_w, model)
            w_parameter_init = new_w

            if (epoch / opt.fl_epoch) == 19:

                if (opt.usr_id % 10) == 6:
                    print("Save FL model!")
                    fl_model_path = "./save_uniFL/{}_models/".format(opt.dataset)

                    if not os.path.isdir(fl_model_path):
                        os.makedirs(fl_model_path)
                    if opt.local_modality == 'gps':
                        save_model(model.encoder, optimizer, opt, opt.epochs, os.path.join(fl_model_path, 'last_gps.pth'))
                    elif opt.local_modality == 'lidar':
                        save_model(model.encoder, optimizer, opt, opt.epochs, os.path.join(fl_model_path, 'last_lidar.pth'))
                    elif opt.local_modality == 'image':
                        save_model(model.encoder, optimizer, opt, opt.epochs, os.path.join(fl_model_path, 'last_image.pth'))


    print("Testing accuracy of node {} is : {}".format(opt.usr_id, val_acc))
    np.savetxt(opt.result_path + "record_loss.txt", record_loss)
    np.savetxt(opt.result_path + "record_acc.txt", record_acc)
    np.savetxt(opt.result_path + "record_confusion.txt", confusion)

    np.savetxt(opt.result_path + "compute_time_record.txt", compute_time_record)
    np.savetxt(opt.result_path + "commu_time_record.txt", commu_time_record)
    np.savetxt(opt.result_path + "all_time_record.txt", all_time_record)
    
    comm.disconnect(1)



if __name__ == '__main__':
    main()
