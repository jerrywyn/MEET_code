
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.utils.data import WeightedRandomSampler


from util.datasets_MEET_1125 import build_dataset




def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default=None, type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.25e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')





    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--dataset', default=None, type=str,help='type of dataset')

    parser.add_argument("--split", default=None, type=int, help='trn-tes ratio')

    parser.add_argument("--tag", default=None, type=int, help='different idx (trn_num for millionaid, idx for others)')

    parser.add_argument("--exp_num", default=0, type=int, help='number of experiment times')

    parser.add_argument("--save_freq", default=10, type=int, help='number of saving frequency')

    parser.add_argument("--eval_freq", default=1, type=int, help='number of evaluation frequency')



    return parser


def main(args):
    print("start")

    device = torch.device(args.device)

    path, _ = os.path.split(args.finetune)

    args.output_dir = os.path.join(path, str(args.model) + '_fintune' + '_' + str(args.dataset) + '_' + str(
        args.split) + '_' + str(args.input_size)) + '_' + str(args.postfix)
    os.makedirs(args.output_dir, exist_ok=True)

    exp_record = np.zeros([3, args.exp_num + 2])

    open(os.path.join(args.output_dir, "log.txt"), mode="w", encoding="utf-8")


    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, is_val=False, args=args)
    dataset_val = build_dataset(is_train=False, is_val=True, args=args)
    dataset_test = build_dataset(is_train=False, is_val=False, args=args)

    tmp_labels = dataset_train.targets.copy()
    tmp = np.unique(tmp_labels, return_counts=True)  # 每个类别出现了几次
    tmp_sum = tmp[1]

    tmp_up = np.clip(tmp_sum, 0, 4000)
    cls_weight = tmp_up / tmp_sum
    cls_weight = cls_weight / np.sum(cls_weight)
    cls_weight = torch.tensor(cls_weight)
    tmp_labels = torch.tensor(tmp_labels, dtype=torch.long)
    weights = cls_weight[tmp_labels]
    sampler_train_head = WeightedRandomSampler(weights=weights, num_samples=150000, replacement=True)
    sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    
  




    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)


    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_train_head = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train_head,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )


    from util.config_peft import _C as cfg
    cfg_model_file = os.path.join("./configs/model", "clip_vit_b16_peft" + ".yaml")
    cfg.defrost()
    cfg.merge_from_file(cfg_model_file)

    model = None


    from util.swin_transformer_large_ori83_adaptformer_final import \
        SwinTransformer_adaptformer_final
    model = SwinTransformer_adaptformer_final(train_init_loader=data_loader_train_head,
                                                            device=device, cfg=cfg)


    from engine_finetune_multi_sup_0508 import train_one_epoch, evaluate

    model_dict = model.state_dict()
    pretrained_weights = torch.load("swin_large_patch4_window7_224_22k.pth", map_location=device)["model"]
    pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    loaded_keys = set(pretrained_dict.keys())
    existing_keys = set(model_dict.keys())

    print("Successfully loaded params:")
    for k in loaded_keys:
        print(k)

    print("\nUnused weights in the pretrained file:")
    for k in pretrained_weights.keys():
        if k not in loaded_keys:
            print(k)

    print("\nMissing weights in the current model:")
    for k in existing_keys:
        if k not in loaded_keys:
            print(k)

    model.to(device)


    import time


    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size / 256

    args.lr = args.lr/64*args.batch_size


    try_lr = args.lr
    time.sleep(10)
    optimizer = torch.optim.AdamW([
        {"params": model.tuner1.parameters(), "lr": try_lr*0.01, "weight_decay": args.weight_decay},
        {"params": model.tuner2.parameters(), "lr": try_lr*0.01, "weight_decay": args.weight_decay},
        {"params": model.tuner3.parameters(), "lr": try_lr*0.01, "weight_decay": args.weight_decay},
        {"params": model.head1.parameters(), "lr": try_lr*0.01, "weight_decay": args.weight_decay},
        {"params": model.head2.parameters(), "lr": try_lr*0.01, "weight_decay": args.weight_decay},
        {"params": model.head3.parameters(), "lr": try_lr*0.01, "weight_decay": args.weight_decay},
        {"params": model.global_context_module_12head.parameters(), "lr": try_lr*0.01,
         "weight_decay": args.weight_decay}
    ], lr=try_lr, weight_decay=args.weight_decay)
    
    
    
    loss_scaler = NativeScaler()


    from la_loss import  LogitAdjustedLoss
    tmp_labels = dataset_train.targets.copy()
    tmp = np.unique(tmp_labels, return_counts=True)  # 每个类别出现了几次
    tmp_sum = tmp[1]
    cls_num_list = tmp_sum
    criterion = LogitAdjustedLoss(cls_num_list=torch.tensor(cls_num_list).to(device))


    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    
    best_acc1 = max_accuracy
    
    print("初始化acc")
    print(best_acc1)
    
    
    for epoch in range(20):
        mixup_fn = None
        model.train()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        if epoch % args.eval_freq == 0:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

        if test_stats["acc1"] > best_acc1:
            save_state = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'max_accuracy': max_accuracy,
                          'epoch': epoch,
                          'args': args}

            save_name = "adaptformer_multi_sup_weight_MEET_adjust_lr.pth"

            save_path = os.path.join(args.output_dir, save_name)
            print(f"{save_path} saving......")
            torch.save(save_state, save_path)
            print(f"{save_path} saved !!!")
            best_acc1 = test_stats["acc1"]



    test_stats = evaluate(data_loader_test, model, device)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    max_accuracy = max(max_accuracy, test_stats["acc1"])
    print(f'Max accuracy: {max_accuracy:.2f}%')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Onetime training time {}'.format(total_time_str))



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
