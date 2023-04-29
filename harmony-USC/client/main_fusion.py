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
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model

from model import MyMMModel, MySingleModel
import data_pre as data
from communication import COMM
import copy
from sklearn.metrics.pairwise import cosine_similarity


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # FL
    parser.add_argument('--usr_id', type=int, default=0,
                        help='user id')
    parser.add_argument('--fl_epoch', type=int, default=10,
                    help='communication to server after the epoch of local training')
    parser.add_argument('--server_address', type=str, default="localhost",
                        help='server_address')
    parser.add_argument('--local_modality', type=str, default='both',
                    help='indicator of local modality')#both, acc, gyr

    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=99,
                        help='number of training epochs')

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
    parser.add_argument('--mu', type=float, default=1e-2,
                        help='mu')

    # model dataset
    parser.add_argument('--model', type=str, default='MyMMmodel')
    parser.add_argument('--approach', type=str, default='mmfl_fusion')
    parser.add_argument('--algorithm', type=str, default='fedavg', choices=['fedprox', 'fedavg'])
    parser.add_argument('--dataset', type=str, default='USC-HAR',
                        choices=['USC-HAR', 'UTD-MHAD', 'ours'], help='dataset')
    parser.add_argument('--num_class', type=int, default=12,
                        help='num_class')
    parser.add_argument('--num_of_train', type=int, default=100,
                        help='num_of_train')
    parser.add_argument('--num_of_test', type=int, default=250,
                        help='num_of_test')
    parser.add_argument('--dim_enc_single', type=int, default=int(39456/2),
                    help='dimension of encoder in singlemodal model')
    parser.add_argument('--dim_enc', type=int, default=int(39456),
                    help='dimension of encoder in singlemodal model')


    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')


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
    opt.model_path = './save_uni_FL/{}_models'.format(opt.dataset)
    opt.result_path = './save_fedfuse_results/node_{}/{}_results/'.format(opt.usr_id, opt.dataset)

    opt.load_folder = opt.model_path
    if not os.path.isdir(opt.load_folder):
        os.makedirs(opt.load_folder)

    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)

    return opt


# construct data loader
def set_loader(opt):
   
    data_path = "../../USC-data/node_{}/".format(opt.usr_id)

    # load data (already normalized)
    x_train_1 = np.load(data_path + "x_train_1.npy")
    x_train_2 = np.load(data_path + "x_train_2.npy")
    y_train = np.load(data_path + "y_train.npy")
    x_test_1 = np.load(data_path + "x_test_1.npy")
    x_test_2 = np.load(data_path + "x_test_2.npy")
    y_test = np.load(data_path + "y_test.npy")

    print(x_train_1.shape)
    print(x_train_2.shape)
    print(x_test_1.shape)
    print(x_test_2.shape)

    np.savetxt(opt.result_path + "y_train.txt", y_train)
    np.savetxt(opt.result_path + "y_test.txt", y_test)
    
    train_dataset = data.Multimodal_dataset(x_train_1, x_train_2, y_train)
    test_dataset = data.Multimodal_dataset(x_test_1, x_test_2, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader, test_loader


def load_single_model(opt, modality):

    ckpt_path = opt.load_folder + "/last_" + modality + ".pth"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    return state_dict


def set_model(opt):

    model = MyMMModel(input_size=1, num_classes=opt.num_class)
    criterion = torch.nn.CrossEntropyLoss()

    state_dict_1 = load_single_model(opt, "acc")
    state_dict_2 = load_single_model(opt, "gyr")

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    model.encoder.acc_cnn_layers.load_state_dict(state_dict_1)
    model.encoder.gyr_cnn_layers.load_state_dict(state_dict_2)

    return model, criterion



def train_multi(train_loader, model, global_model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    prox = AverageMeter()

    end = time.time()
    for idx, (input_data1, input_data2, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            input_data2 = input_data2.cuda()
            labels = labels.cuda()
        bsz = input_data1.shape[0]

        # compute loss
        output = model(input_data1, input_data2)
        loss = criterion(output, labels)

        if opt.algorithm == 'fedprox':

            proximal_term = 0.0

            # iterate through the current and global model parameters
            for w, w_t in zip(model.parameters(), global_model.parameters()) :
                proximal_term += (w-w_t).norm(2)
            loss = loss + (opt.mu / 2) * proximal_term

        acc, _ = accuracy(output, labels, topk=(1, 5))

        # update metric
        losses.update(loss.item(), bsz)
        top1.update(acc[0], bsz)
        if opt.algorithm == 'fedprox':
            prox.update(proximal_term.item(), bsz)

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
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'prox@1 {prox.val:.3f} ({prox.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, prox=prox))
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
        for idx, (input_data1, input_data2, labels) in enumerate(val_loader):

            if torch.cuda.is_available():
                input_data1 = input_data1.float().cuda()
                input_data2 = input_data2.float().cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(input_data1, input_data2)
            loss = criterion(output, labels)

            # update metric
            acc, _ = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), bsz)
            top1.update(acc[0], bsz)

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

            elif len(param.shape) == 4:

                para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3])
                temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3])))
                temp_index += para_len  

            else:

                para_len = param.shape[0]
                temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight))
                temp_index += para_len                   


def set_commu(opt):

    #prepare the communication module
    server_addr = opt.server_address
    server_port = 9998
    comm = COMM(server_addr, server_port, opt.usr_id)

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

        if len_idx < opt.dim_enc_single:
            enc_1_dis += (w-w_t).norm(2).cpu().detach().numpy()
        elif len_idx >= opt.dim_enc_single and len_idx < opt.dim_enc:
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
    cls_dis = 0.0

    model_weight = get_model_array(model)
    init_model_weight = get_model_array(init_model)

    enc_1_dis = cosine_similarity(model_weight[0:opt.dim_enc_single].reshape(1, -1), init_model_weight[0:opt.dim_enc_single].reshape(1, -1))
    enc_2_dis = cosine_similarity(model_weight[opt.dim_enc_single:opt.dim_enc].reshape(1, -1), init_model_weight[opt.dim_enc_single:opt.dim_enc].reshape(1, -1))
    cls_dis = cosine_similarity(model_weight[opt.dim_enc:].reshape(1, -1), init_model_weight[opt.dim_enc:].reshape(1, -1))

    return enc_1_dis, enc_2_dis, cls_dis



def main():

    opt = parse_option()

    torch.manual_seed(42)


    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    global_model = copy.deepcopy(model)
    w_parameter_init = get_model_array(global_model)

    init_model = copy.deepcopy(model)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # build optimizer for feature extractor and classifier
    optimizer = optim.SGD([ 
                {'params': model.encoder.parameters(), 'lr': 1e-4},   # 0
                {'params': model.gru.parameters(), 'lr': opt.learning_rate},
                {'params': model.classifier.parameters(), 'lr': opt.learning_rate}],
                momentum=opt.momentum,
                weight_decay=opt.weight_decay)


    best_acc = 0
    best_confusion = np.zeros((opt.num_class, opt.num_class))
    record_loss = np.zeros(opt.epochs)
    record_acc = np.zeros(opt.epochs)

    record_enc_1 = np.zeros(opt.epochs)
    record_enc_2 = np.zeros(opt.epochs)
    record_cls = np.zeros(opt.epochs)

    # set up communication with sevrer
    comm = set_commu(opt)

    # training routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train_multi(train_loader, model, global_model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        record_loss[epoch-1] = loss

        # evaluation
        loss, val_acc, confusion = validate_multi(val_loader, model, criterion, opt)
        record_acc[epoch-1] = val_acc
        if val_acc > best_acc:
             best_acc = val_acc
             best_confusion = confusion

        encoder_1_dis, encoder_2_dis, classifier_dis = calculate_dis(model, init_model, opt)
        record_enc_1[epoch-1] = encoder_1_dis
        record_enc_2[epoch-1] = encoder_2_dis
        record_cls[epoch-1] = classifier_dis
        print("Node {} distance:".format(opt.usr_id), encoder_1_dis, encoder_2_dis, classifier_dis)

        # # communication with the server every fl_epoch 
        if (epoch % opt.fl_epoch) == 0 and (epoch / opt.fl_epoch) > 1:


            print("Node {} sends weight to the server:".format(opt.usr_id))
            w_parameter = get_model_array(model) #obtain the model parameters or gradients 
            w_update = w_parameter - w_parameter_init
            comm.send2server(w_update,0)

            # print("averaged model weight:", np.mean(w_parameter))

            new_w_update, sig_stop = comm.recvOUF()
            print("Received weight from the server:", new_w_update)
            print("Received signal from the server:", sig_stop)

            ## only update fusion and classifier
            w_parameter[opt.dim_enc:] += new_w_update[opt.dim_enc:]
            reset_model_parameter(w_parameter, model)

            ## record current model weight
            global_model = copy.deepcopy(model)
            w_parameter_init = w_parameter


    # evaluation
    loss, val_acc, confusion = validate_multi(val_loader, model, criterion, opt)

    print("Testing accuracy of node {} is : {}".format(opt.usr_id, val_acc))
    np.savetxt(opt.result_path + "record_loss.txt".format(opt.usr_id), record_loss)
    np.savetxt(opt.result_path + "record_acc.txt".format(opt.usr_id), record_acc)
    np.savetxt(opt.result_path + "record_confusion.txt", confusion)
    np.savetxt(opt.result_path + "record_enc_1_dis.txt", record_enc_1)
    np.savetxt(opt.result_path + "record_enc_2_dis.txt", record_enc_2)
    np.savetxt(opt.result_path + "record_cls_dis.txt", record_cls)

    comm.disconnect(1)


if __name__ == '__main__':
    main()
