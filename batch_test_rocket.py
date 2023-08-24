import argparse
import os
import sys
import random
import json
import numpy as np
import torch

from typing import Iterable, Optional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (DataLoader, BatchSampler, RandomSampler,
                              SequentialSampler, DistributedSampler)
import util
from models import build_model 
from datasets import build_dataset
from loss import build_criterion 
from common.error import NoGradientError
from common.logger import Logger, MetricLogger, SmoothedValue
from common.functions import *
from common.nest import NestedTensor
from configs import dynamic_load
import time
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

mean_sar = np.array([0.33247536, 0.33247536, 0.33247536],dtype=np.float32).reshape(3,1,1)
std_sar = np.array([0.16769384, 0.16769384, 0.16769384],dtype=np.float32).reshape(3,1,1)
mean_opt = np.array([0.31578836, 0.31578836, 0.31578836],dtype=np.float32).reshape(3,1,1)
std_opt = np.array([0.1530546, 0.1530546 ,0.1530546],dtype=np.float32).reshape(3,1,1)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def test(
    loader: Iterable, model: torch.nn.Module, print_freq=10000., tb_logger=None
):
    model.eval()
    def _transform_inv(img,mean,std):
        img = img * std + mean
        img  = np.uint8(img * 255.0)
        img = img.transpose(1,2,0)
        return img


    logger = MetricLogger(delimiter=' ')
    header = 'Test'
    scores = 0
    i_err = {thr: 0 for thr in np.arange(1,11)}
    thres = [1,3,5,10]
    nums = 0
    dists_sa = []

    for sample_batch in logger.log_every(loader, print_freq, header):
        scores+=1
        images1 = sample_batch["refer"].cuda().float()
        images0 = sample_batch["query"].cuda().float()
        gt_matrix=0
        x,y=sample_batch['pos']
        x,y=int(x), int(y)

        preds = model(images0, images1, gt_matrix)
        samples0 = _transform_inv(images0.detach().cpu().numpy().squeeze(), mean_sar, std_sar)
        samples1 = _transform_inv(images1.detach().cpu().numpy().squeeze(), mean_opt, std_opt)
        #out2 = draw_match5(preds['mkpts0'][:, 1:], preds['mkpts1'][:, 1:], samples0, samples1, x, y)
        out2 = draw_match_nir(preds['mkpts0'][:, 1:], preds['mkpts1'], samples0, samples1, x, y)
        cv2.imwrite(f"train_channel/results_pic/{time.time()}.jpg", out2)
        #cv2.waitKey()
        i_err, num = eval_src_mma(preds['mkpts0'][:,1: ], preds['mkpts1'], samples0, samples1, i_err)
        dist = eval_src_homography(preds['mkpts0'][:,1: ], preds['mkpts1'], samples0, samples1)
        dists_sa.append(dist)
        nums += 1
    correct_sa = np.mean(
            [[float(dist <= t) for t in thres] for dist in dists_sa], axis=0)
    auc_sa = cal_error_auc(dists_sa, thresholds=thres)
    for thr in i_err:
        i_err[thr] = i_err[thr] / nums
    return i_err, auc_sa


def main(args):
    util.init_distributed_mode(args)

    seed = args.seed + util.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Seed used:', seed)

    model: torch.nn.Module = build_model(args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters:', n_params)
    model = model.to(DEV)
    train_dataset, test_dataset = build_dataset(args)
    test_sampler = SequentialSampler(test_dataset)


    dataloader_kwargs = {
        #'collate_fn': train_dataset.collate_fn,
        'pin_memory': False,
        'num_workers': 0,
    }

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        drop_last=True,
        **dataloader_kwargs
    )
    res = {}
    
    # model_names = os.listdir("artifacts/train_femit/resnet101-dual_softmax_dim256-128_depth256-128/")
    max_auc_3 = 0
    max_auc_5 = 0
    max_auc_10 = 0
    max_model_name = ""
    model_names = os.listdir("train_channel/artifacts/train_channel2/resnet101-dual_softmax_dim256-128_depth256-128/")
    for model_name in [x for x in model_names if '139_model_SAR2RGB_rotate_best_mean_std_mish_lbl_2734.7_49.8.pth'in x]:
        state_dict = torch.load(f"train_channel/artifacts/train_channel2/resnet101-dual_softmax_dim256-128_depth256-128/{model_name}", map_location='cpu')
        model.load_state_dict(state_dict['model'])

        save_name = f'{args.backbone_name}-{args.matching_name}'
        save_name += f'_dim{args.d_coarse_model}-{args.d_fine_model}'
        save_name += f'_depth{args.d_coarse_model}-{args.d_fine_model}'

        save_path = os.path.join(args.save_path, save_name)
        os.makedirs(save_path, exist_ok=True)
        if util.is_main_process():
            tensorboard_logger = Logger(save_path)
        else:
            tensorboard_logger = None

        print(f'Start Testing model {model_name} ...')

        test_stats = test(
            test_loader,
            model,
        )
        print(test_stats)
        res[model_name] = {'err':test_stats[0], 'auc': test_stats[1]}
    # print("max model name: ",model_name)
    # print("max auc_3 is: ", max_auc_3)
    print(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str,
                        default='imcnet_config')
    global_cfgs = parser.parse_args()

    args = dynamic_load(global_cfgs.config_name)
    prm_str = 'Arguments:\n' + '\n'.join(
        ['{} {}'.format(k.upper(), v) for k, v in vars(args).items()]
    )
    print(prm_str + '\n')
    print('=='*40 + '\n')

    main(args)

# 奇了怪了
# femit 训练出来  一样的训练方法 怎么差这么远呢
