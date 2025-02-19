# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/
# --------------------------------------------------------

import math
import sys
#from typing import Iterable, Optionalbeit
import numpy as np
import os
import torch

from timm.data import Mixup
from timm.utils import accuracy
import pickle
import util.misc as misc
import util.lr_sched as lr_sched

import datetime
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn=None, log_writer=None,
                    args=None):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    header = 'Time: [{}] Epoch: [{}]'.format(current_time, epoch)

    print_freq = 50

    accum_iter = 1

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))


    for data_iter_step, (samples,context,context2, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, epoch)):
        optimizer.zero_grad()
        
        samples = samples.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        context2 = context2.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)


        

        with torch.cuda.amp.autocast():
            outputs,outputs2,outputs3 = model(samples,context,context2)
            
            loss = criterion(outputs, targets)+criterion(outputs2, targets)+criterion(outputs3, targets)
        loss_value = loss.item()


        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            optimizer.zero_grad()
            continue

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
                    
        
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




def accuracy_cls(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    num_classes = 30
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    class_top1_correct = torch.zeros(num_classes)
    class_top5_correct = torch.zeros(num_classes)
    class_sample_count = torch.zeros(num_classes)
    # 计算每个类别的 Top-1 和 Top-5 命中次数
    for i in range(num_classes):
        # 对于每个样本，计算该样本是否属于当前类别
        is_class_i = (target == i).to(torch.int)  # 将布尔值转换为整数 1 或 0

        # 计算 Top-1 命中次数
        top1_correct_class_i = (correct[0] * is_class_i).sum()
        class_top1_correct[i] = top1_correct_class_i.item()

        # 计算 Top-5 命中次数
        top5_correct_class_i = (correct.sum(dim=0) * is_class_i).sum()
        class_top5_correct[i] = top5_correct_class_i.item()

        class_sample_count[i] = is_class_i.sum().item()
    # 计算每个类别的 acc1 和 acc5
    total_samples = len(target)
    acc1_per_class = torch.zeros(num_classes)
    acc5_per_class = torch.zeros(num_classes)
    for i in range(num_classes):
        if class_sample_count[i] > 0:
            acc1_per_class[i] = class_top1_correct[i] / class_sample_count[i]
            acc5_per_class[i] = class_top5_correct[i] / class_sample_count[i]
        else:
            acc1_per_class[i] = -1
            acc5_per_class[i] = -1
    acc1 = torch.mean(acc1_per_class).item()
    acc5 = torch.mean(acc5_per_class).item()
    #return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk],acc1_per_class,acc5_per_class
    return (acc1,acc5)
    #return acc1_per_class,acc5_per_class

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    stat_wyn = []

    stat_target = []
    stat_pred = []

    outputs = []
    outputs1,outputs2,outputs3 = [],[],[]
    targets = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        context = batch[1]
        context2 = batch[2]
        target = batch[-1]
        #print(target.numpy())
        images = images.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        context2 = context2.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        #with torch.cuda.amp.autocast():
        #记得改
        with torch.cuda.amp.autocast():
            output1,output2,output3 = model(images,context,context2)
            #output = (output1+output2+output3)/3.0
            output = output3
            #output = model(images)
            loss = criterion(output, target)

        _, pred = output.topk(1, 1, True, True)
        pred_numpy = pred.detach().cpu().numpy()
        stat_pred.append(pred_numpy)
        target_np = target.detach().cpu().numpy()
        stat_target.append(target_np)
        
        
        outputs.append(output)

        output1_numpy = output1.detach().cpu().numpy()
        output2_numpy = output2.detach().cpu().numpy()
        output3_numpy = output3.detach().cpu().numpy()
        
        outputs1.append(output1_numpy)
        outputs2.append(output2_numpy)
        outputs3.append(output3_numpy)



        targets.append(target)


        acc = accuracy(output, target, topk=(1, 5))
        acc1, acc5 = acc

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    path = os.path.join(r"/media/dell/DATA/wyn/code/MAEPretrain_SceneClassification/save", 'tmptmptmp_pred.pkl')
    with open(path, "wb") as f:
        pickle.dump(stat_pred, f)

    path = os.path.join(r"/media/dell/DATA/wyn/code/MAEPretrain_SceneClassification/save", 'tmptmptmp_target.pkl')
    with open(path, "wb") as f:
        pickle.dump(stat_target, f)

    path = os.path.join(r"/media/dell/DATA/wyn/code/MAEPretrain_SceneClassification/save", 'outputs1.pkl')
    with open(path, "wb") as f:
        pickle.dump(outputs1, f)

    path = os.path.join(r"/media/dell/DATA/wyn/code/MAEPretrain_SceneClassification/save", 'outputs2.pkl')
    with open(path, "wb") as f:
        pickle.dump(outputs2, f)

    path = os.path.join(r"/media/dell/DATA/wyn/code/MAEPretrain_SceneClassification/save", 'outputs3.pkl')
    with open(path, "wb") as f:
        pickle.dump(outputs3, f)




    output = torch.cat(outputs,dim=0)
    target = torch.cat(targets,dim=0)
    acc = accuracy_cls(output, target, topk=(1, 5))
    acc1, acc5 = acc
    print('*Class!!! Acc@1 {top1:.3f} Acc@5 {top5:.3f}'.format(top1=acc1, top5=acc5))

    metric_logger.synchronize_between_processes()
    print('*Instance!!!! Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))


    #return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return {'acc1':acc1,'acc5':acc5}

