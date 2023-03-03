import argparse
import os
from re import S
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import pickle
import copy

from models import *
from data_loader import data_loader
from helper import *

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', "vit_b_16"
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', default='/home/cguo/imagenet-raw-data/', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--prune', dest='prune', action='store_true',
                    help='prune and finetune model')
parser.add_argument('--eval', dest='eval', action='store_true',
                    help='Evaluation')
parser.add_argument('--pruning_type', default='tw1', type=str, metavar='N',
                    help='The pruning_type for network pruning, (default: ew)')
parser.add_argument('--pre_masks_dir', default=None, type=str, metavar='N',
                    help='The pre_masks_dir for network pruning, (default: None)')
parser.add_argument('--finetune_steps', default=10000, type=int, metavar='N',
                    help='finetune_steps (default: 10000)')
parser.add_argument('--mini_finetune_steps', default=5000, type=int, metavar='N',
                    help='mini_finetune_steps (default: 5000)')


best_prec1 = 0.0


def get_accuracy(sparsity, tile_size):
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    model = get_model(args.arch)
    model = pruning_model(model)
    # use cuda
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            with open(args.pre_masks_dir, "rb") as file:
                all_mask_values = pickle.load(file)
                load_mask(model, all_mask_values)
    else:
        pass
        # print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading
    train_loader, val_loader = data_loader(args.data, args.batch_size, args.workers, args.pin_memory)

    # if args.evaluate:
    #     validate(val_loader, model, criterion, args.print_freq)
    #     return

    if args.prune:
        # validate(val_loader, model, criterion, args.print_freq)
        print("train size:", len(train_loader))
        prec1, prec5 = prune(val_loader, train_loader, model, optimizer, criterion, args.print_freq)
        return  prec1, prec5

    # if args.eval:
    #     validate(val_loader, model, criterion, args.print_freq)
    #     return


    # for epoch in range(args.start_epoch, args.epochs):
    #     adjust_learning_rate(optimizer, epoch, args.lr)

    #     # train for one epoch
    #     train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

    #     # evaluate on validation set
    #     prec1, prec5 = validate(val_loader, model, criterion, args.print_freq)

    #     # remember the best prec@1 and save checkpoint
    #     is_best = prec1 > best_prec1
    #     best_prec1 = max(prec1, best_prec1)

    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'arch': args.arch,
    #         'state_dict': model.state_dict(),
    #         'best_prec1': best_prec1,
    #         'optimizer': optimizer.state_dict()
    #     }, is_best, args.arch + '.pth')


def train(train_loader, model, criterion, optimizer, epoch, print_freq, stop_step = 1e10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

        if i == stop_step:
            break


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg

def prune(val_loader, train_loader, model, optimizer, criterion, print_freq):
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    pruning_type = args.pruning_type
    model_dir = root_dir() / "train" / args.arch / pruning_type / now
    finetune_steps = int(args.finetune_steps)
    mini_finetune_steps = int(args.mini_finetune_steps)

    if "tw" in pruning_type:
        pruning_layers = [[0,1,2,3],[4,5,6],[7,8,9],[10,11,12]]
        sparsity_stages = [25, 40, 50, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96]
    if "ew" in pruning_type:
        pruning_layers = [[0,1,2,3,4,5,6,7,8,9,10,11,12]]
        sparsity_stages = [25, 40, 50, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96]

    if "bw" in pruning_type:
        pruning_layers = [[0,1],[2,3],[4,5,6],[7,8,9],[10,11,12]]
        sparsity_stages = [25, 40, 50, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96]
    if "twvw" in pruning_type:
        pruning_layers = [[0,1],[2,3],[4,5,6],[7,8,9],[10,11,12]]
        sparsity_stages = [50, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96]

    if args.arch == 'resnet18':
        base_threshold = 69.768
        pruning_layers = [[0,1,2,3,4,5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19]]
        # pruning_layers = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
    elif args.arch == 'resnet50':
        base_threshold = 76.122
        # pruning_layers = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], [24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42], [43,44,45,46,47,48,49,50,51,52]]
        pruning_layers = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]]
    elif args.arch == 'vgg16':
        base_threshold = 90.380
        # pruning_layers = [[0,1],[2,3],[4,5,6],[7,8,9],[10,11,12]]
        pruning_layers = [[0,1,2,3,4,5,6,7,8,9,10,11,12]]
    elif args.arch == 'vit_b_16':
        base_threshold = 95.320
        pruning_layers = [
            [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
            [25,26,27,28,29,30,31,32,33,34,35,36],
            [37,38,39,40,41,42,43,44,45,46,47,48],
        ]
    sparsity_stages = [25,  50,  60,  65,  70,   75,  80, 85,  90, 92, 94, 96]
    threshold_stages = [0.0, 0.2, 0.4, 0.6,  0.8, 1.0, 1.0, 1.5, 3.0, 100, 100, 100]
    early_stop = [100, 100, 100, 100, 100, 100]

    # print('First Evaluation')
    # prec1, prec5 = validate(val_loader, model, criterion, args.print_freq)
    # print("First Evaluation results: %f/%f" % (prec1, prec5))
    # print('Done First Evaluation')

    ## Multi stages to prune the CNN model
    for stage in range(len(sparsity_stages)):
        stage_dir = model_dir / ("sparsity_stage_" + str(sparsity_stages[stage]))
        sparsity = sparsity_stages[stage]
        threshold = base_threshold - threshold_stages[stage]

        print("Stage : %d, Sparsity : %f, Finetune_steps : %d" %(stage, sparsity, finetune_steps))
        print("Early_stop:")
        print(early_stop)

        epoch = 0
        all_layer_dir = stage_dir / "all_layers"
        masks_dir = all_layer_dir / "masks"
        os.makedirs(masks_dir)

        print('Prune !')
        for layer in range(len(pruning_layers)):
            masks_now = pruning_layers[layer]
            update_mask(model, sparsity, pruning_type, masks_now)
        print('Done Prune !!!\n')
        input()
        pruning_info(model)
        for _ in range(6):
            print('Mini Fine Tune !')
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args.print_freq, stop_step = mini_finetune_steps)
            print('Done Mini Fine Tune !\n')


            print('Evaluation !')
            prec1, prec5 = validate(val_loader, model, criterion, args.print_freq)
            print("Pruning Sparsity %d Layer block %d: %f/%f" % (sparsity, layer, prec1, prec5))
            print('Done Evaluation !\n')

            print(prec1)
            print(threshold)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, False, all_layer_dir/('ckpt_' + str(sparsity) + '.pth'))
            previous_checkpoint = all_layer_dir/('ckpt_' + str(sparsity) + '.pth')
            previous_mask_values = copy.deepcopy(dump_mask(model))

            if prec1 > threshold:
                print('GOOD CHECKPOINT!!!\n')
                with open(masks_dir / ("good_mask_" + str(sparsity) + ".pkl"), "wb") as file:
                    pickle.dump(dump_mask(model), file)
                break
            else:
                print('BAD CHECKPOINT!!!\n')
                with open(masks_dir / ("bad_mask_" + str(sparsity) + ".pkl"), "wb") as file:
                    pickle.dump(dump_mask(model), file)
            pruning_info(model)

        assert(previous_checkpoint)
        print("=> loading checkpoint '{}'".format(previous_checkpoint))
        checkpoint = torch.load(previous_checkpoint)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(previous_checkpoint, checkpoint['epoch']))
        load_mask(model, previous_mask_values)

        print('Fine Tune !')
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq, stop_step = finetune_steps)
        print('Done Fine Tune !!!\n')

        print('Last Evaluation !!!')
        prec1, prec5 = validate(val_loader, model, criterion, args.print_freq)
        print("Pruning Sparsity %d Layer block %d: %f/%f" % (sparsity, layer, prec1, prec5))
        print('Done Evaluation !!!\n')

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, False, all_layer_dir/('ckpt_' + str(sparsity) + '.pth'))
        previous_checkpoint = all_layer_dir/('ckpt_' + str(sparsity) + '.pth')
        previous_mask_values = copy.deepcopy(dump_mask(model))
        with open(masks_dir / ("mask_" + str(sparsity) + ".pkl"), "wb") as file:
            pickle.dump(dump_mask(model), file)
    
    return prec1, prec5

if __name__ == '__main__':
    get_accuracy()
