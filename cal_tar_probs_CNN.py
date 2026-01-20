import os
import pdb
import time
import numpy as np
import torch
from timm.models.layers import trunc_normal_
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from functools import partial, reduce
# from utils import iterative_A
from timm.models.vision_transformer import VisionTransformer, _cfg
from multiprocessing import Pool
from ViT_models.mocov3 import VisionTransformerMoCo
# from pathos.multiprocessing import ProcessingPool as Pool
from joblib import Parallel, delayed
import argparse


def softmax(X, copy=True):
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X




def CNNprobs(X, model_name, dataset):
    # Group1: model_name, fc_name, model_ckpt
    ckpt_models = {
        'densenet121': ['classifier.weight', './models/checkpoints/densenet121-a639ec97.pth'],
        'densenet169': ['classifier.weight', './models/checkpoints/densenet169-b2777c0a.pth'],
        'densenet201': ['classifier.weight', './models/checkpoints/densenet201-c1103571.pth'],
        'resnet34': ['fc.weight', './models/checkpoints/resnet34-333f7ec4.pth'],
        'resnet50': ['fc.weight', './models/checkpoints/resnet50-19c8e357.pth'],
        'resnet101': ['fc.weight', './models/checkpoints/resnet101-5d3b4d8f.pth'],
        'resnet152': ['fc.weight', './models/checkpoints/resnet152-b121ed2d.pth'],
        'mnasnet1_0': ['classifier.1.weight', './models/checkpoints/mnasnet1.0_top1_73.512-f206786ef8.pth'],
        'mobilenet_v2': ['classifier.1.weight', './models/checkpoints/mobilenet_v2-b0353104.pth'],
        'googlenet': ['fc.weight', './models/checkpoints/googlenet-1378be20.pth'],
        'inception_v3': ['fc.weight', './models/checkpoints/inception_v3_google-1a9a5a14.pth'],
    }
    ckpt_loc = ckpt_models[model_name][1]
    ckpt = torch.load(ckpt_loc, map_location='cpu')
    fpath = os.path.join('./results_f_trainval', 'CNN')
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    model_npy_probs = os.path.join(fpath, f'{model_name}_{dataset}_probs.npy')
    if os.path.exists(model_npy_probs):
        print(f"Probs of {model_name} on {dataset} has been saved.")
    fc_weight = ckpt_models[model_name][0]
    fc_bias = fc_weight.replace('weight', 'bias')

    fc_weight = ckpt[fc_weight].detach().numpy()
    fc_bias = ckpt[fc_bias].detach().numpy()

    # p(z|x), z is source label
    print("X, fc_weight.T, fc_bias", X.shape, fc_weight.T.shape, fc_bias.shape)
 
    prob = np.dot(X, fc_weight.T) + fc_bias
    prob = softmax(prob)  # p(z|x), N x C(source)
    np.save(model_npy_probs, prob)
    print(f"Probs of {model_name} on {dataset} has been saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate transferability score.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='dtd',
                        help='name of the dataset to evaluate on')
    parser.add_argument('-me', '--metric', type=str, default='logme',
                        help='name of the method for measuring transferability')
    parser.add_argument('--nleep-ratio', type=float, default=5,
                        help='the ratio of the Gaussian components and target data classess')
    parser.add_argument('--parc-ratio', type=float, default=2,
                        help='PCA reduction dimension')
    parser.add_argument('--output_dir', type=str, default='./results_metrics/group_CNN',
                        help='dir of output score')
    args = parser.parse_args()
    print(args)
    fpath = os.path.join(args.output_dir, args.metric)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    fpath = os.path.join(fpath, f'{args.dataset}_metrics_clip.json')
    models_hub = ['inception_v3', 'mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201',
                  'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet']
    datasets_hub = ['aircraft', 'caltech101', 'cars', 'cifar100', 'dtd', 'flowers', 'food', 'pets', 'sun397','voc2007']
    for dataset in datasets_hub:
        start_time = time.time()
        args.dataset = dataset
        for model in models_hub:
            args.model = model
            model_npy_feature = os.path.join(
                './results_f_trainval/CNN',
                f'{args.model}_{args.dataset}_feature.npy')
            X_features = np.load(model_npy_feature)
            CNNprobs(X_features, model, dataset)


