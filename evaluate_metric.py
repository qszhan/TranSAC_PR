#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pdb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import models as models
import numpy as np
from scipy.stats import weightedtau
import json
import time
from TranSAC import compute_TranSAC
from metrics import NLEEP, SFDA_Score, LogME_Score,  LEEP 
from scipy.stats import kendalltau
from scipy.stats import weightedtau
import pprint
import json
from scipy.stats import pearsonr, spearmanr
from utils import wpearson, return_nclass_data_semi
from sklearn.decomposition import PCA

def save_score(score_dict, fpath):
    with open(fpath, "w") as f:
        # write dict 
        json.dump(score_dict, f)


def exist_score(model_name, fpath):
    with open(fpath, "r") as f:
        result = json.load(f)
        if model_name in result.keys():
            return True
        else:
            return False


def recall_k(score, dset, k):
    # succed = 0
    sorted_score = sorted(score.items(), key=lambda i: i[1], reverse=True)
    sorted_score = {a[0]: a[1] for a in sorted_score}

    gt = finetune_acc[dset]
    sorted_gt = sorted(gt.items(), key=lambda i: i[1], reverse=True)
    sorted_gt = {a[0]: a[1] for a in sorted_gt}

    top_k_gt = sorted_gt.keys()[:k]
    succed = 1 if sorted_score.keys()[0] in top_k_gt else 0
    return succed


def rel_k(score, dset, k):
    sorted_score = sorted(score.items(), key=lambda i: i[1], reverse=True)

    gt = finetune_acc[dset]
    sorted_gt = sorted(gt.items(), key=lambda i: i[1], reverse=True)
    best_model = sorted_gt[0][0]
    sorted_gt = {a[0]: a[1] for a in sorted_gt}

    max_gt = sorted_gt[best_model]
    topk_score_model = [a[0] for i, a in enumerate(sorted_score) if i < k]
    topk_score_ft = [sorted_gt[a] for a in topk_score_model]
    return max(topk_score_ft) / max_gt


def pearson_coef(score, dset):
    global finetune_acc
    score = score.items()
    metric_score = [a[1] for a in score]
    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    tw_metric, _ = pearsonr(metric_score, gt_)
    return tw_metric

def spearmanr_coef(score, dset):
    global finetune_acc
    score = score.items()
    metric_score = [a[1] for a in score]
    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    tw_metric, _ = spearmanr(metric_score, gt_)
    return tw_metric

def wpearson_coef(score, dset):
    global finetune_acc
    score = score.items()
    metric_score = [a[1] for a in score]
    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    tw_metric = wpearson(metric_score, gt_)
    return tw_metric


def w_kendall_metric(score, dset):
    global finetune_acc
    score = score.items()
    metric_score = [a[1] for a in score]
    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    tw_metric, _ = weightedtau(metric_score, gt_)
    return tw_metric


def kendall_metric(score, dset):
    global finetune_acc_ssl
    score = score.items()
    metric_score = [a[1] for a in score]
    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    t_metric, _ = kendalltau(metric_score, gt_)
    return t_metric


finetune_acc = {
    'aircraft': {'resnet34': 84.06, 'resnet50': 84.64, 'resnet101': 85.53, 'resnet152': 86.29, 'densenet121': 84.66,
                 'densenet169': 84.19, 'densenet201': 85.38, 'mnasnet1_0': 66.48, 'mobilenet_v2': 79.68,
                 'googlenet': 80.32, 'inception_v3': 80.15},
    'caltech101': {'resnet34': 91.15, 'resnet50': 91.98, 'resnet101': 92.38, 'resnet152': 93.1, 'densenet121': 91.5,
                   'densenet169': 92.51, 'densenet201': 93.14, 'mnasnet1_0': 89.34, 'mobilenet_v2': 88.64,
                   'googlenet': 90.85, 'inception_v3': 92.75},
    'cars': {'resnet34': 88.63, 'resnet50': 89.09, 'resnet101': 89.47, 'resnet152': 89.88, 'densenet121': 89.34,
             'densenet169': 89.02, 'densenet201': 89.44, 'mnasnet1_0': 72.58, 'mobilenet_v2': 86.44,
             'googlenet': 87.76, 'inception_v3': 87.74},
    'cifar10': {'resnet34': 96.12, 'resnet50': 96.28, 'resnet101': 97.39, 'resnet152': 97.53, 'densenet121': 96.45,
                'densenet169': 96.77, 'densenet201': 97.02, 'mnasnet1_0': 92.59, 'mobilenet_v2': 94.74,
                'googlenet': 95.54,
                'inception_v3': 96.18},
    'cifar100': {'resnet34': 81.94, 'resnet50': 82.8, 'resnet101': 84.88, 'resnet152': 85.66, 'densenet121': 82.75,
                 'densenet169': 84.26, 'densenet201': 84.88, 'mnasnet1_0': 72.04, 'mobilenet_v2': 78.11,
                 'googlenet': 79.84,
                 'inception_v3': 81.49},
    'dtd': {'resnet34': 72.96, 'resnet50': 74.72, 'resnet101': 74.8, 'resnet152': 76.44, 'densenet121': 74.18,
            'densenet169': 74.72, 'densenet201': 76.04, 'mnasnet1_0': 70.12, 'mobilenet_v2': 71.72,
            'googlenet': 72.53,
            'inception_v3': 72.85},
    'flowers': {'resnet34': 95.2, 'resnet50': 96.26, 'resnet101': 96.53, 'resnet152': 96.86, 'densenet121': 97.02,
                'densenet169': 97.32, 'densenet201': 97.1, 'mnasnet1_0': 95.39, 'mobilenet_v2': 96.2,
                'googlenet': 95.76,
                'inception_v3': 95.73},
    'food': {'resnet34': 81.99, 'resnet50': 84.45, 'resnet101': 85.58, 'resnet152': 86.28, 'densenet121': 84.99,
             'densenet169': 85.84, 'densenet201': 86.71, 'mnasnet1_0': 71.35, 'mobilenet_v2': 81.12,
             'googlenet': 79.3,
             'inception_v3': 81.76},
    'pets': {'resnet34': 93.5, 'resnet50': 93.88, 'resnet101': 93.92, 'resnet152': 94.42, 'densenet121': 93.07,
             'densenet169': 93.62, 'densenet201': 94.03, 'mnasnet1_0': 91.08, 'mobilenet_v2': 91.28,
             'googlenet': 91.38,
             'inception_v3': 92.14},
    'sun397': {'resnet34': 61.02, 'resnet50': 63.54, 'resnet101': 63.76, 'resnet152': 64.82, 'densenet121': 63.26,
               'densenet169': 64.1, 'densenet201': 64.57, 'mnasnet1_0': 56.56, 'mobilenet_v2': 60.29,
               'googlenet': 59.89,
               'inception_v3': 59.98},
    'voc2007': {'resnet34': 84.6, 'resnet50': 85.8, 'resnet101': 85.68, 'resnet152': 86.32, 'densenet121': 85.28,
                'densenet169': 85.77, 'densenet201': 85.67, 'mnasnet1_0': 81.06, 'mobilenet_v2': 82.8,
                'googlenet': 82.58,
                'inception_v3': 83.84}
}


 

# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate transferability score.')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default=None,
                        help='name of the dataset to evaluate on')
    parser.add_argument('-me', '--metric', type=str, default=None,
                        help='name of the method for measuring transferability')
    parser.add_argument('--nleep-ratio', type=float, default=5,
                        help='the ratio of the Gaussian components and target data classes')
    parser.add_argument('--parc-ratio', type=float, default=2,
                        help='PCA reduction dimension')
    parser.add_argument('--output_dir', type=str, default='./results_metrics_MS/group_CNN',
                        help='dir of output score')
    parser.add_argument('--rat', type=float, default=0.1,
                        help='ratio to samples a small part of data for metric calculation')
    parser.add_argument('--eps', type=float, default=10,
                        help='eps used in continuous entropy calculation')
    parser.add_argument('--order', type=int, default=5,
                        help='order used in continuous entropy calculation')
    args = parser.parse_args()
    print(args)
    score_dict = {}
 
 

    fpath = args.output_dir
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    fpath = os.path.join(fpath, f'{args.metric}_anahtcz.txt')
    finetune = []
    score = []
    models_hub = ['inception_v3','mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201',
                  'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet']
 
    datasets_hub = ['aircraft', 'caltech101', 'cars', 'cifar10', 'cifar100',  'dtd', 'flowers', 'food', 'pets', 'sun397', 'voc2007']
 
    elapsed_time_all = []
    pearAll = []
    kenAll = []
    WkenAll = []
    Spear = []
    WpearAll = []
 
    
    for dataset in datasets_hub:
        start_time = time.time()
        args.dataset = dataset
 
 
        finetune = []
        score = []
        score_dict = {}
        ht_all = {}
        hz_all = {}
        h_zct_all = {}
        for model in models_hub:
            args.model = model
            model_npy_feature = os.path.join('./results_f_trainval/CNN', f'{args.model}_{args.dataset}_feature.npy')
            model_npy_label = os.path.join('./results_f_trainval/CNN', f'{args.model}_{args.dataset}_label.npy')
            model_npy_probs = os.path.join('./results_f_trainval/CNN', f'{args.model}_{args.dataset}_probs.npy')
            X_features, y_labels, probs = np.load(model_npy_feature), np.load(model_npy_label), np.load(model_npy_probs)
            # print(y_labels.max())
            n_rat_class = y_labels.max() + 1
            # imbalanced sampling
            y_labels, X_features, probs = return_nclass_data_semi(y_labels, X_features, probs, n_rat_class,
                                                                       args.rat)
 

            if args.metric == 'TranSAC':
                n, k = probs.shape
                Transac_Score = compute_TranSAC(probs, X_features, args)
                score_dict[args.model] = Transac_Score
            elif args.metric == 'logme':
                score_dict[args.model] = LogME_Score(X_features, y_labels)
             
            elif args.metric == 'leep': 
                score_dict[args.model] = LEEP(X_features, y_labels, model_name=args.model)
                 
            elif args.metric == 'nleep':
                ratio = 1 if args.dataset in ('food', 'pets') else args.nleep_ratio
                score_dict[args.model] = NLEEP(X_features, y_labels, component_ratio=ratio)
            elif args.metric == 'sfda':
                scaler = MinMaxScaler()
                X_features = scaler.fit_transform(X_features)
                pca = PCA(n_components=20)
                X_features = pca.fit_transform(X_features)
                print(X_features.shape)
                score_dict[args.model] = SFDA_Score(X_features, y_labels)
         
            else:
                raise NotImplementedError
            finetune.append(finetune_acc[args.dataset][args.model])
            print(f'{args.metric} of {args.model}: {score_dict[args.model]}\n')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time: {:.2f} s".format(elapsed_time))

        
 

 
        print(f'Models ranking on {args.dataset} based on {args.metric}: ')
        # print(results)
        print("hz_all", hz_all)
        print("ht_all", ht_all)
        print("score_dict", score_dict)
 
        tw = w_kendall_metric(score_dict, args.dataset)
        t = kendall_metric(score_dict, args.dataset)
        pear = pearson_coef(score_dict, args.dataset)
        wpear = wpearson_coef(score_dict, args.dataset)
        spear = spearmanr_coef(score_dict, args.dataset)
        rel_3 = rel_k(score_dict, args.dataset, k=3)
        rel_1 = rel_k(score_dict, args.dataset, k=1)
        pearAll.append(pear)
        WpearAll.append(wpear)
        kenAll.append(t)
        WkenAll.append(tw)
        Spear.append(spear)
        elapsed_time_all.append(elapsed_time)
        if args.metric == 'TranSAC':
            pearAllSAC.append(h_zct_pear)
            WpearAllSAC.append(h_zct_wpear)
            kenAllSAC.append(h_zct_t)
            WkenAllSAC.append(h_zct_tw)
            SpearSAC.append(h_zct_spear)

        with open(fpath, 'a', encoding='utf-8') as fw:
            fw.write("\n================================")
            fw.write("\nscore_dict:\t")
            for key, value in score_dict.items():
                fw.write(f'{value}\t')
            fw.write("\nrat: {:.3f}".format(args.rat))
            fw.write("\neps: {:.3f}".format(args.eps))
            fw.write("\norder: {:.3f}".format(args.order))
            fw.write("\nElapsed time: {:.2f} s".format(elapsed_time))
            fw.write("\nRel@1    dataset: {:12s} {:12s}: {:2.3f}".format(args.dataset, args.metric, rel_1))
            fw.write("\nRel@3    dataset: {:12s} {:12s}:{:2.3f}".format(args.dataset, args.metric, rel_3))
            fw.write("\nPearson  dataset: {:12s} {:12s}:{:2.3f}".format(args.dataset, args.metric, pear))
            fw.write("\nWPearson dataset: {:12s} {:12s}:{:2.3f}".format(args.dataset, args.metric, wpear))
            fw.write("\nKendall  dataset: {:12s} {:12s}:{:2.3f}".format(args.dataset, args.metric, t))
            fw.write("\nWKendall dataset: {:12s} {:12s}:{:2.3f}".format(args.dataset, args.metric, tw))
            fw.write("\nSpearmanr  dataset: {:12s} {:12s}:{:2.3f}".format(args.dataset, args.metric, spear))
            fw.close()
 
              
        print("Rel@1    dataset:{:12s} {:12s} :{:2.3f}".format(args.dataset, args.metric, rel_1))
        print("Rel@3    dataset:{:12s} {:12s} :{:2.3f}".format(args.dataset, args.metric, rel_3))
        print("Pearson  dataset:{:12s} {:12s} :{:2.3f}".format(args.dataset, args.metric, pear))
        print("WPearson dataset:{:12s} {:12s} :{:2.3f}".format(args.dataset, args.metric, wpear))
        print("Kendall  dataset:{:12s} {:12s} :{:2.3f}".format(args.dataset, args.metric, t))
        print("WKendall dataset:{:12s} {:12s} :{:2.3f}".format(args.dataset, args.metric, tw))
        print("Spearmanr  dataset:{:12s} {:12s} :{:2.3f}".format(args.dataset, args.metric, spear))

        print('*' * 80)

    print("rat, eps, order", args.rat, args.eps, args.order)
    print("pearAll", ' '.join(['{:.3f}'.format(x) for x in pearAll]))
    print("kenAll", ' '.join(['{:.3f}'.format(x) for x in kenAll]))
    print("WkenAll", ' '.join(['{:.3f}'.format(x) for x in WkenAll]))
    print("Spear", ' '.join(['{:.3f}'.format(x) for x in Spear]))
    print("WpearAll", ' '.join(['{:.3f}'.format(x) for x in WpearAll]))
    print("elapsed_time_all", ' '.join(['{:.3f}'.format(x) for x in elapsed_time_all]))
 


