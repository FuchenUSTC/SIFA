import argparse
import os
import time
import json
import numpy as np
import yaml

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

from util import clip_transforms
from util.clip_augmentations import ClipRandAugment
from util.util import ClipGaussianBlur, AverageMeter, merge_scores, accuracy, reduce_tensor
from util.lr_scheduler import get_scheduler
from util.logger import setup_logger
from layer.LSR import *
from dataset.video_merge_dataset import VideoMergeDataset
from dataset.video_dataset import VideoRGBTrainDataset, VideoRGBTestDataset

import model as model_factory
from layer.pooling_factory import get_pooling_by_name

from torch.cuda.amp import GradScaler


def add_config(args, name, config):
    if isinstance(config, dict):
        for key in config.keys():
            add_config(args, key, config[key])
    else:
        setattr(args, name, config)


def merge_config(conf1, conf2):
    if isinstance(conf1, dict) and isinstance(conf2, dict):
        new_config = {}
        key_list = list(set(conf1.keys()).union(set(conf2.keys())))

        for key in key_list:
            if (key in conf1) and (key in conf2): # union of c1 & c2
                new_config[key] = merge_config(conf1.get(key), conf2.get(key))
            else:
                new_config[key] = conf1.get(key) if key in conf1 else conf2.get(key)
        return new_config
    else:
        return conf1 if conf2 is None else conf2


def parse_option():
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--config_file', type=str, required=True, help='path of config file (yaml)')
    parser.add_argument('--local_rank', type=int, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    # load config file, default + base + exp
    config_default = yaml.load(open('./base_config/default.yml', 'r'))
    config_exp = yaml.load(open(args.config_file, 'r'))
    if 'base' in config_exp:
        config_base = yaml.load(open(config_exp['base'], 'r'))
    else:
        config_base = None
    config = merge_config(merge_config(config_default, config_base), config_exp)
    args.C = config
    add_config(args, 'root', config)
    return args


def get_loader(args):
    if args.rand_augment:
        train_transform = transforms.Compose([
            clip_transforms.ClipRandomResizedCrop(args.crop_size, scale=(0.2, 1.), ratio=(0.75, 1.3333333333333333)),
            ClipRandAugment(n=args.ra_n, m=args.ra_m),  # N = [1, 2, 3], M = [5, 7, 9, 11, 13, 15]
            clip_transforms.ClipRandomHorizontalFlip(p=0.0 if args.no_horizontal_flip else 0.5),
            clip_transforms.ToClipTensor(),
            clip_transforms.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if args.time_dim == "T" else transforms.Lambda(lambda clip: torch.cat(clip, dim=0))
        ])        
    else:
        train_transform = transforms.Compose([
            clip_transforms.ClipRandomResizedCrop(args.crop_size, scale=(0.2, 1.), ratio=(0.75, 1.3333333333333333)),
            transforms.RandomApply([
                clip_transforms.ClipColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            clip_transforms.ClipRandomGrayscale(p=0.2),
            transforms.RandomApply([ClipGaussianBlur([.1, 2.])], p=0.5),
            clip_transforms.ClipRandomHorizontalFlip(p=0.0 if args.no_horizontal_flip else 0.5),
            clip_transforms.ToClipTensor(),
            clip_transforms.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if args.time_dim == "T" else transforms.Lambda(lambda clip: torch.cat(clip, dim=0))
        ])

    if args.dataset_class == 'video_dataset':
        assert (args.list_file != '' and args.root_path != '')
        train_dataset = VideoRGBTrainDataset(list_file=args.list_file, root_path=args.root_path,
                                             transform=train_transform, clip_length=args.clip_length,
                                             num_steps=args.num_steps, num_segments=args.num_segments,
                                             format=args.format)
    else:
        assert (args.lmdb_path != '' and args.video_num != 0 and args.repeat_num != 0)
        train_dataset = VideoMergeDataset(args.lmdb_path + '_' + str(args.local_rank), video_num=args.video_num,
                                          repeat_num=args.repeat_num, transform=train_transform,
                                          clip_length=args.clip_length, num_steps=args.num_steps,
                                          num_segments=args.num_segments)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    return train_loader


def get_val_loader(args):
    # only crop the center clip for evaluation
    crop = clip_transforms.ClipCenterCrop
    test_transform = transforms.Compose([
        clip_transforms.ClipResize(size=args.crop_size),
        crop(size=args.crop_size),
        clip_transforms.ToClipTensor(),
        clip_transforms.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if args.time_dim == "T" else transforms.Lambda(lambda clip: torch.cat(clip, dim=0))
    ])

    test_dataset = VideoRGBTestDataset(args.eva_list_file, num_clips=args.num_clips, transform=test_transform, root_path=args.root_path, \
                                       clip_length=args.clip_length, num_steps=args.num_steps, num_segments=args.num_segments, \
                                       format=args.format)
    #test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)
        #sampler=test_sampler, )
    return test_loader    


def build_model(args):
    model = model_factory.get_model_by_name(net_name=args.net_name, pooling_arch=get_pooling_by_name(args.pooling_name),
                              num_classes=args.num_classes, dropout_ratio=args.dropout_ratio, 
                              clip_length=(args.num_segments*args.clip_length), sifa_kernel=args.sifa_kernel).cuda()
    if args.pretrained_model:
        load_pretrained(args, model)
    return model


def load_pretrained(args, model):
    ckpt = torch.load(args.pretrained_model, map_location='cpu')
    if 'model' in ckpt:
        state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
    else:
        state_dict = ckpt

    # convert initial weights
    if args.transfer_weights:
        state_dict = model_factory.transfer_weights(args.net_name, state_dict)
    if args.remove_fc:
        state_dict = model_factory.remove_fc(args.net_name, state_dict)
    if args.remove_defcor_weight:
        state_dict = model_factory.remove_defcor_weight(args.net_name, state_dict)

    [misskeys, unexpkeys] = model.load_state_dict(state_dict, strict=False)
    logger.info('Missing keys: {}'.format(misskeys))
    logger.info('Unexpect keys: {}'.format(unexpkeys))
    logger.info("==> loaded checkpoint '{}'".format(args.pretrained_model))


def save_checkpoint(args, epoch, model, optimizer, scheduler):
    logger.info('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(args.output_dir, 'current.pth'))
    if epoch % args.save_freq == 0:
        torch.save(state, os.path.join(args.output_dir, 'ckpt_epoch_{}.pth'.format(epoch)))


def main(args):
    train_loader = get_loader(args)
    val_loader = get_val_loader(args)
    n_data = len(train_loader.dataset)
    logger.info("length of training dataset: {}".format(n_data))

    model = build_model(args)
    if args.pretrained_model:
        ckpt = torch.load(args.pretrained_model, map_location='cpu')
    
    # print network architecture
    if dist.get_rank() == 0:
        logger.info(model)

    if args.label_smooth:
        criterion = LSR(e=0.1).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.base_learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.resume:
        if args.reverse: 
            optimizer.load_state_dict(ckpt['scheduler'])
        else: optimizer.load_state_dict(ckpt['optimizer'])

    # scheduler
    scheduler = get_scheduler(optimizer, len(train_loader), args)
    if args.resume:
        if args.reverse:
            scheduler.load_state_dict(ckpt['optimizer'])
        else: scheduler.load_state_dict(ckpt['scheduler'])

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=True,
                                    find_unused_parameters=True)
    #model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    #model = DDP(model)
    
    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        summary_writer = None

    # routine
    start_epoch = 1
    if args.resume: start_epoch = ckpt['epoch']
    for epoch in range(start_epoch, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        tic = time.time()
        loss = train(epoch, train_loader, model, criterion, optimizer, scheduler, args)
        logger.info('epoch {}, total time {:.2f}'.format(epoch, time.time() - tic))
        if summary_writer is not None:
            # tensorboard logger
            summary_writer.add_scalar('ins_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        if dist.get_rank() == 0:
            # save model
            save_checkpoint(args, epoch, model, optimizer, scheduler)
            if epoch % args.eva_inter_freq == 0 or epoch == args.epochs:
                # evaluation on test data
                tic_val = time.time()
                eva_accuracy = eval(epoch, val_loader, model, args)
                top1_accuracy = eva_accuracy[0].cuda()
                top3_accuracy = eva_accuracy[1].cuda()
                top5_accuracy = eva_accuracy[2].cuda()
                t1 = top1_accuracy.data.cpu().item()
                t3 = top3_accuracy.data.cpu().item()
                t5 = top5_accuracy.data.cpu().item()                
                logger.info('val top1 accuracy {:.4f}, top3 accuracy: {:.4f}, top5: {:.4f} val time {:.2f}'.format(t1, t3, t5, time.time() - tic_val))


def frozen_bn(model):
    first_bn = True
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            if first_bn:
                first_bn = False
                print('Skip frozen first bn layer: ' + name)
                continue
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def train(epoch, train_loader, model, criterion, optimizer, scheduler, args):
    model.train()
    if args.frozen_bn:
        frozen_bn(model)

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    end = time.time()

    optimizer.zero_grad()
    scaler = GradScaler()
    bnorm = 0

    for idx, (x, label) in enumerate(train_loader):
        bsz = x.size(0)

        # forward
        x = x.cuda(non_blocking=True)  # clip
        label = label.cuda(non_blocking=True)  # label

        # with torch.cuda.amp.autocast():
        # forward and get the predict score
        score = model(x)
        # get crossentropy loss
        if isinstance(score, list):
            loss = criterion(score[0], label) + criterion(score[1], label)
        else:
            loss = criterion(score, label)

        # backward
        scaler.scale(loss / args.iter_size * args.loss_weight).backward()
        #with amp.scale_loss(loss, optimizer) as scaled_loss:
        #    scaled_loss.backward()

        if (idx + 1) % args.iter_size == 0:
            scaler.unscale_(optimizer)
            bnorm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                                   args.clip_gradient)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()

        # update meters
        loss_meter.update(loss.item(), bsz)
        norm_meter.update(bnorm, bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % args.print_freq == 0:
            lr = scheduler.get_lr()[0]
            logger.info(
                'Train: [{:>3d}]/[{:>4d}/{:>4d}] BT={:>0.3f}/{:>0.3f} Loss={:>0.3f}/{:>0.3f} GradNorm={:>0.3f}/{:>0.3f} Lr={:>0.3f}'.format(
                    epoch, idx, len(train_loader),
                    batch_time.val, batch_time.avg,
                    loss.item(), loss_meter.avg,
                    bnorm, norm_meter.avg,lr
                ))

    return loss_meter.avg


def eval(epoch, val_loader, model, args):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    all_scores = np.zeros([len(val_loader) * args.batch_size, args.num_classes], dtype=np.float)
    all_labels = np.zeros([len(val_loader) * args.batch_size], dtype=np.float)
    top_idx = 0
    with torch.no_grad():
        logger.info('==> Validating... num clips {} val video num {}'.format(args.num_clips,args.val_video_num))
        for idx, (x, label) in enumerate(val_loader):
            if idx % 100 == 0:
                logger.info('{}/{}'.format(idx, len(val_loader)))
            bsz = x.size(0)
            score = model(x)
            #score = softmax(score)
            if isinstance(score, list):
                score_numpy = (softmax(score[0]).data.cpu().numpy() + softmax(score[1]).data.cpu().numpy()) / 2
            else:
                score_numpy = softmax(score).data.cpu().numpy()
            label_numpy = label.data.cpu().numpy()
            all_scores[top_idx: top_idx + bsz, :] = score_numpy
            all_labels[top_idx: top_idx + bsz] = label_numpy
            top_idx += bsz
    all_scores = all_scores[:top_idx, :]
    # pooling the scores for each video
    v_all_scores, v_all_labels = merge_scores(all_scores, all_labels, args)
    # compute the accuracy
    acc = accuracy(v_all_scores, v_all_labels, topk=(1, 3, 5))
    return acc


if __name__ == '__main__':
    opt = parse_option()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt.rank = int(os.environ["RANK"])
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(opt.local_rank)
    else:
        torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="sifa")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "train_val_3d.config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    main(opt)
