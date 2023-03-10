import sys
import os
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import torch
from torch import nn


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)

def label_to_onehot(gt, num_classes, ignore_index=-1):
    '''
    Converts segmentation label to one hot format
    gt: ground truth label with size (N, H, W)
    num_classes: number of classes
    ignore_index: index(es) for ignored classes
    '''
    N, H, W = gt.size()
    gt_ = gt
    gt_[gt_ == ignore_index] = num_classes
    onehot = torch.zeros(N, gt_.size(1), gt_.size(2), num_classes + 1).cuda()
    onehot = onehot.scatter_(-1, gt_.unsqueeze(-1), 1)          

    return onehot.permute(0, 3, 1, 2)