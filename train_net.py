# -*- coding:utf-8 -*-
"""
Training the OneFace model
@create: Mingchao Jiang
@date  : 2021-01-26
"""
from __future__ import print_function

import os
import cv2
import math
import time
import argparse
import random
import numpy as np
from os import path as ospath

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from models.detector import OneFace
from optim.Optimizer import BuildOptimzer
from loss.loss import SetCriterion, MinCostMatcher
from dataset.collate_function import collate_fn
from dataset.default_dataset import BaseDataSet
from utils.build_targets import Targets
from Config import Config

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='OneFace Training')
parser.add_argument('--training_file',
                    default='/data/remote/MarkCamera/facedataset/training/face_annoation_1210_for_train.txt', help='Training dataset directory')
parser.add_argument('--config_file',
                    default='/data/remote/github_code/OneFace/config/resnet50.yaml')
parser.add_argument('--log_writer', default=1, help="write the training log")
parser.add_argument('--ngpu', type=int, default=1)

# ddp
parser.add_argument('--world-size', type=int, default=-1,
                    help="number of nodes for distributed training")
parser.add_argument('--rank', default=-1, type=int,
                help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                help='distributed backend')
parser.add_argument('--multiprocessing-distributed', default=1, type=int,
                help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')
parser.add_argument('--local_rank', default=1)

# apex
parser.add_argument('--use_apex', type=int, default=1, help='use the apex for mixup traininig!!!')

# random seed
def setup_seed(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def translate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def main_worker(gpu, ngpus_per_node, args):
    # each gpu is like a rank
    cfg = Config(args.config_file)()
    args.gpu = gpu

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
        args.rank = args.rank * ngpus_per_node + gpu
    model_arch = "{}-{}".format("OneFace", cfg.BACKBONE.NAME)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    print('rank: {} / {}'.format(args.rank, args.world_size))
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.gpu)

    if args.rank == 0:
        if not os.path.exists(cfg.CHEKCPOINTS.CKPT_PATH):
            os.mkdir(cfg.CHEKCPOINTS.CKPT_PATH)

    # model
    model = OneFace(backbone_name=cfg.MODEL.BACKBONE.NAME, num_classes=cfg.TRAIN.NUM_CLASSES)
    if args.rank == 0:
        print("================{}=============".format(model_arch))
        print(model)
    if torch.cuda.is_available():
        model.cuda(args.gpu)
    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    device = model.device
    # optimizer
    optimbuilder = BuildOptimzer(cfg)
    optimizer = optimbuilder.optimizer(model)

    # loss
    class_weight = cfg.MODEL.CLASS_WEIGHT
    l1_weight = cfg.MODEL.L1_WEIGHT
    giou_weight = cfg.MODEL.GIOU_WEIGHT
    matcher = MinCostMatcher(
        cost_class=class_weight,
        cost_bbox=l1_weight,
        cost_giou=giou_weight
    )
    weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
    losses = ["labels", "boxes"]
    criterion = SetCriterion(
                            num_classes=cfg.TRAIN.NUM_CLASSES,
                            matcher=matcher,
                            weight_dict=weight_dict,
                            losses=losses
    )

    # dataset
    train_dataset = BaseDataSet(cfg, args.training_file)
    if args.rank == 0:
        print("dataset", len(train_dataset))
    # sampler
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # logs
    log_writer = SummaryWriter(cfg.CHEKCPOINTS.LOGS_PATH)

    # dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_fn
    )

    batch_iter = 0
    train_batch = math.ceil(len(train_dataset) / (cfg.TRAIN.BATCH_SIZE * ngpus_per_node))
    total_batch = train_batch * cfg.TRAIN.MAX_EPOCHS

    # training loop
    for epoch in range(cfg.TRAIN.MAX_EPOCHS):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for epoch
        batch_iter = train(cfg, train_loader, model, criterion, optimizer, epoch, args, batch_iter, total_batch, train_batch, log_writer, ngpus_per_node)

        if (epoch + 1) % 5 == 0:
            if args.rank == 0:
                state_dict = translate_state_dict(model.state_dict())
                state_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state_dict, cfg.CHEKCPOINTS.CKPT_PATH + '/'  + model_arch  + '_epoch_{}'.format(epoch+1) + '.pth')

def record_log(log_writer, bbox_loss, class_loss, giou_loss, losses, lr, batch_idx, batch_time):
    log_writer.add_scalar("train/bbox_loss", bbox_loss.data.item(), batch_idx)
    log_writer.add_scalar("train/class_loss", class_loss.data.item(), batch_idx)
    log_writer.add_scalar("train/giou_loss", giou_loss.data.item(), batch_idx)
    log_writer.add_scalar("train/total_loss", losses.data.item(), batch_idx)
    log_writer.add_scalar("learning_rate", lr, batch_idx)
    log_writer.add_scalar("train/batch_time", batch_time, batch_idx)

def train(cfg, train_loader, model, criterion, optimizer, epoch, args, batch_iter, total_batch, train_batch, log_writer, ngpus_per_node):
    model.train()
    device = model.device
    make_targets = Targets(device)
    loader_length = len(train_loader)
    for batch_idx, targets in enumerate(train_loader):
        targets_dict = make_targets.prepare_targets(targets)
        batch_start = time.time()

        lr = adjust_learning_rate(cfg, epoch, batch_idx+1, optimizer, loader_length, ngpus_per_node)

        # forward
        optimizer.zero_grad()
        images = targets_dict["images"]
        outputs_class, outputs_coord, outputs_points = model(images)
        output = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

        # calculate the loss for output with target
        loss_dict = criterion(output, targets_dict)
        weight_dict = criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        losses = sum(loss_dict.values())
        losses.backward()
        optimizer.step()

        batch_time = time.time() - batch_start

        batch_iter += 1
        batch_idx += 1
        
        if args.rank == 0:
            print("Training Epoch: [{}/{}] batchidx:[{}/{}] batchiter: [{}/{}] Loc: {:.4f} Cla: {:.4f} iou: {:.4f} total: {:.4f} LearningRate: {:.6f} Batchtime: {:.4f}s".format(
                epoch+1, cfg.TRAIN.MAX_EPOCHS, batch_idx, train_batch, batch_iter, total_batch, loss_dict['loss_bbox'].data.item(), loss_dict['loss_ce'].data.item(), loss_dict['loss_giou'].data.item(), losses.data.item(), lr, batch_time ))

        if args.log_writer:
            if args.rank == 0:
                record_log(log_writer, loss_dict['loss_bbox'], loss_dict['loss_ce'], loss_dict['loss_giou'], losses, lr, batch_iter, batch_time)

    return batch_iter

def adjust_learning_rate(cfg, epoch, batch_idx, optimizer, loader_length, ngpus_per_node):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    total_epochs = cfg.TRAIN.MAX_EPOCHS
    warm_epochs = cfg.TRAIN.WARM_EPOCHS
    if epoch < warm_epochs:
        epoch += float(batch_idx + 1) / loader_length
        lr_adj = 1. / ngpus_per_node * (epoch * (ngpus_per_node - 1) / warm_epochs + 1)
    elif epoch < int(0.7 * total_epochs):
        lr_adj = 1.
    elif epoch < int(0.9 * total_epochs):
        lr_adj = 1e-1
    else:
        lr_adj = 1e-2
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.TRAIN.BASE_LR * lr_adj
    return cfg.TRAIN.BASE_LR * lr_adj


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = Config(args.config_file)()
    if cfg.SEED is not None:
        setup_seed(cfg.SEED)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        print("ngpus_per_node", ngpus_per_node)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print("ngpus_per_node", ngpus_per_node)
        main_worker(args.gpu, ngpus_per_node, args)

