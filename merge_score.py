import argparse
import numpy as np
import os
import yaml
from sifa_util.util import accuracy


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
    config_default = yaml.load(open('./base_sh/default.yml', 'r'))
    config_exp = yaml.load(open(args.config_file, 'r'))
    if 'base' in config_exp:
        config_base = yaml.load(open(config_exp['base'], 'r'))
    else:
        config_base = None
    config = merge_config(merge_config(config_default, config_base), config_exp)
    args.C = config
    add_config(args, 'root', config)
    return args


if __name__ == '__main__':
    opt = parse_option()
    num_gpu = opt.num_gpu
    num_crop = opt.eva_num_crop
    num_cls = opt.num_classes
    num_clip = opt.eva_num_clips
    score_dir = opt.output_dir
    list_dir = opt.eva_list_file
    
    for crop_id in range(num_crop):
        all_num = 0
        all_data = []
        for gpu_id in range(num_gpu):
            all_data.append(np.load(os.path.join(score_dir, 'all_scores_' + str(crop_id * num_gpu + gpu_id) +'.npy')))
            all_num += all_data[-1].shape[0]
    
        merge_data = np.empty((all_num, num_cls))
        for gpu_id in range(num_gpu):
            merge_data[gpu_id::num_gpu, :] = all_data[gpu_id]

        # make ave score
        num_video = all_num // num_clip
        merge_data = merge_data[0:num_video * num_clip, :]
        if crop_id == 0:
            reshape_data = np.zeros((num_video, num_clip, num_cls))
        reshape_data += np.reshape(merge_data, (num_video, num_clip, num_cls))  / num_crop
    
    reshape_data = np.sort(reshape_data, axis=1)
    # make gt
    gt = np.zeros((num_video, ))
    lines = open(list_dir, 'r').readlines()
    for idx, line in enumerate(lines):
        ss = line.split(' ')
        label = ss[-1]
        gt[idx] = int(label)
    
    max_acc = 0
    max_top_k = 0
    for i in range(num_clip):
        pred = (reshape_data[:, -i-1:, :].mean(axis=1)).argmax(axis=1)
        acc = (pred == gt).mean()
        if acc > max_acc:
            max_acc = acc
            max_top_k = i
        print('top-k-mean, ' + str(i) + ', acc: ' + str(acc))
        
    print('best top-k-mean: ' + str(max_acc))
    best_score = (reshape_data[:, -max_top_k-1:, :].mean(axis=1))
    best_accuracy = accuracy(best_score, gt, topk=(1, 3, 5))
    print('top1/3/5 accuracy: ' + str(best_accuracy))    
