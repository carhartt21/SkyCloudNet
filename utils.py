import sys
import os
import logging
import re
import functools
import fnmatch
import numpy as np
import torch
import torch.nn as nn
from json import load
from math import ceil
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import faiss 
import random
import logging
import pickle 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# from sklearn.utils.linear_assignment_ import linear_assignment

import warnings
warnings.filterwarnings('ignore')

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

from PIL import ImageFont
from PIL import ImageDraw 

def str_list(l):
    return '_'.join([str(x) for x in l]) 

def visualize_result_with_attr(info, pred_seg, pred_attr, cfg, gt_seg = None, gt_attr = None, epoch=''):
    colors=[]
    with open(cfg.DATASET.classInfo) as f:
        cls_info = load(f)
    font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 150)
    for c in cls_info:
        colors.append(cls_info[c]['color'])
    img = Image.open(info)
    img = img.resize((pred_seg.shape[1], pred_seg.shape[0]), resample = Image.Resampling.BILINEAR)
    # print predictions in descending order
    pred = np.int32(pred_seg)
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    if gt_seg.size > 0:
        gt_color = colorEncode(gt_seg, colors).astype(np.uint8)
        im_vis = np.concatenate((img, pred_color, gt_color), axis=1)
    else: 
        im_vis = np.concatenate((img, pred_color), axis=1)
    img_name = info.split('/')[-1]
    img_out = Image.fromarray(im_vis)
    draw = ImageDraw.Draw(img_out)
    draw.text((500, 500), 'Predicted: {} | GT: {}'.format(pred_attr, gt_attr), (255,255,255), font=font)
    img_out.save(
        os.path.join(cfg.TEST.result, '{}_{}_{}_{}_{}.png'
            .format(epoch, img_name[:-4], cfg.MODEL.arch_encoder, cfg.MODEL.arch_decoder_attr, cfg.MODEL.arch_decoder_seg)))


def visualize_result(info, pred, cfg, gt_seg=None, epoch=''):
    colors=[]
    with open(cfg.DATASET.classInfo) as f:
        cls_info = load(f)
    for c in cls_info:
        colors.append(cls_info[c]['color'])
    colors = np.array(colors, dtype='uint8')
    img = Image.open(info)
    img = img.resize((pred.shape[1], pred.shape[0]), resample=Image.Resampling.BILINEAR)
    # print predictions in descending order
    pred = np.int32(pred)
    # pixs = pred.size
    # uniques, counts = np.unique(pred, return_counts=True)
    # print("Predictions in [{}]:".format(info))
    # for idx in np.argsort(counts)[::-1]:
    #     name = names[str(uniques[idx] + 1)]
    #     ratio = counts[idx] / pixs * 100
    #     if ratio > 0.1:
    #         print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    if gt_seg.size > 0:
        gt_color = colorEncode(gt_seg, colors).astype(np.uint8)
        im_vis = np.concatenate((img, pred_color, gt_color), axis=1)
    else: 
        im_vis = np.concatenate((img, pred_color), axis=1)
    img_name = info.split('/')[-1]
    Image.fromarray(pred_color).save(os.path.join(cfg.TEST.result, '{}_{}_prediction.png'.format(epoch, img_name[:-4])))
    Image.fromarray(im_vis).save(
        os.path.join(cfg.TEST.result, '{}_{}{}{}{}.png'
            .format(epoch, img_name[:-4], cfg.MODEL.arch_encoder, cfg.MODEL.arch_decoder_attr, cfg.MODEL.arch_decoder_seg)))


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
    N, _, _ = gt.size()
    gt_ = gt
    gt_[gt_ == ignore_index] = num_classes
    onehot = torch.zeros(N, gt_.size(1), gt_.size(2), num_classes + 1).cuda()
    onehot = onehot.scatter_(-1, gt_.unsqueeze(-1), 1) 

    return onehot.permute(0, 3, 1, 2)

def onehot_to_class(onehot):
    batch_size = onehot.squeeze().shape[0]
    cls = torch.zeros(batch_size, dtype=torch.long)
    if len(onehot.shape) > 1:
        for i in range(onehot.shape[0]):
            _cls = torch.nonzero((onehot[i] + 1))
            if _cls.nelement() > 0:
                cls[i] = _cls
            else:
                cls[i] = torch.tensor([-1])
        return cls.cuda()
    else:
        _cls = torch.nonzero(onehot + 1)
        if _cls.nelement() > 0:
            cls = _cls
        else:
            cls = torch.tensor([-1])
        return cls.cuda()

def setup_logger(distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    fmt = "[%(asctime)s %(levelname)s %(filename)s] %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    return logger


def find_recursive(root_dir, ext='.jpg', names_only=False):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            if names_only:
                files.append(filename)
            else:
                files.append(os.path.join(root, filename))
    return files


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('uint8')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        if label == 255:
            label = -1
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1)).astype(np.uint8)

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def accuracy(preds, label, ignore_index=-1):
    valid = (label >= 0)
    valid = (valid != ignore_index)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(im_pred, im_label, num_class):
    im_pred = np.asarray(im_pred).copy()
    im_label = np.asarray(im_label).copy()

    im_pred += 1
    im_label += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    im_pred = im_pred * (im_label > 0)

    # Compute area intersection:
    intersection = im_pred * (im_pred == im_label)
    (area_intersection, _) = np.histogram(
        intersection, bins=num_class, range=(1, num_class))

    # Compute area union:
    (area_pred, _) = np.histogram(im_pred, bins=num_class, range=(1, num_class))
    (area_lab, _) = np.histogram(im_label, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


class NotSupportedCliException(Exception):
    pass


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end + 1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):

    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "{}"'.format(d))
    return ret


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, align_corners=False):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.align_corners = align_corners
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = nn.functional.interpolate(input=score, size=(h, w), mode='bilinear',
                                              align_corners=self.align_corners)
        loss = self.criterion(score, target)
        return loss

    def forward(self, score, target):
        weights = [0.4, 1]
        print('Len score {}'.format(len(score)))
        assert len(weights) == len(score)
        combined_loss = [w * self._forward(x, target) for (w, x) in zip(weights, score)]
        # print(combined_loss)
        return sum(combined_loss)


def run_mini_batch_kmeans(args, logger, dataloader, model, view):
    """
    num_init_batches: (int) The number of batches/iterations to accumulate before the initial k-means clustering.
    num_batches     : (int) The number of batches/iterations to accumulate before the next update. 
    """
    kmeans_loss  = AverageMeter()
    faiss_module = get_faiss_module(args)
    data_count   = np.zeros(args.K_train)
    featslist    = []
    num_batches  = 0
    first_batch  = True
    
    # Choose which view it is now. 
    dataloader.dataset.view = view

    model.train()
    with torch.no_grad():
        for i_batch, (indice, image) in enumerate(dataloader):
            # 1. Compute initial centroids from the first few batches. 
            if view == 1:
                image = eqv_transform_if_needed(args, dataloader, indice, image.cuda(non_blocking=True))
                feats = model(image)
            elif view == 2:
                image = image.cuda(non_blocking=True)
                feats = eqv_transform_if_needed(args, dataloader, indice, model(image))
            else:
                # For evaluation. 
                image = image.cuda(non_blocking=True)
                feats = model(image)

            # Normalize.
            if args.metric_test == 'cosine':
                feats = F.normalize(feats, dim=1, p=2)
            
            if i_batch == 0:
                logger.info('Batch input size : {}'.format(list(image.shape)))
                logger.info('Batch feature : {}'.format(list(feats.shape)))
            
            feats = feature_flatten(feats).detach().cpu()
            if num_batches < args.num_init_batches:
                featslist.append(feats)
                num_batches += 1
                
                if num_batches == args.num_init_batches or num_batches == len(dataloader):
                    if first_batch:
                        # Compute initial centroids. 
                        # By doing so, we avoid empty cluster problem from mini-batch K-Means. 
                        featslist = torch.cat(featslist).cpu().numpy().astype('float32')
                        centroids = get_init_centroids(args, args.K_train, featslist, faiss_module).astype('float32')
                        D, I = faiss_module.search(featslist, 1)

                        kmeans_loss.update(D.mean())
                        logger.info('Initial k-means loss: {:.4f} '.format(kmeans_loss.avg))
                        
                        # Compute counts for each cluster. 
                        for k in np.unique(I):
                            data_count[k] += len(np.where(I == k)[0])
                        first_batch = False
                    else:
                        b_feat = torch.cat(featslist)
                        faiss_module = module_update_centroids(faiss_module, centroids)
                        D, I = faiss_module.search(b_feat.numpy().astype('float32'), 1)

                        kmeans_loss.update(D.mean())

                        # Update centroids. 
                        for k in np.unique(I):
                            idx_k = np.where(I == k)[0]
                            data_count[k] += len(idx_k)
                            centroid_lr    = len(idx_k) / (data_count[k] + 1e-6)
                            centroids[k]   = (1 - centroid_lr) * centroids[k] + centroid_lr * b_feat[idx_k].mean(0).numpy().astype('float32')
                    
                    # Empty. 
                    featslist   = []
                    num_batches = args.num_init_batches - args.num_batches

            if (i_batch % 100) == 0:
                logger.info('[Saving features]: {} / {} | [K-Means Loss]: {:.4f}'.format(i_batch, len(dataloader), kmeans_loss.avg))

    centroids = torch.tensor(centroids, requires_grad=False).cuda()

    return centroids, kmeans_loss.avg




def compute_labels(args, logger, dataloader, model, centroids, view):
    """
    Label all images for each view with the obtained cluster centroids. 
    The distance is efficiently computed by setting centroids as convolution layer. 
    """
    K = centroids.size(0)

    # Choose which view it is now. 
    dataloader.dataset.view = view

    # Define metric function with conv layer. 
    metric_function = get_metric_as_conv(centroids)

    counts = torch.zeros(K, requires_grad=False).cpu()
    model.eval()
    with torch.no_grad():
        for i, (indice, image) in enumerate(dataloader):
            if view == 1:
                image = eqv_transform_if_needed(args, dataloader, indice, image.cuda(non_blocking=True))
                feats = model(image)
            elif view == 2:
                image = image.cuda(non_blocking=True)
                feats = eqv_transform_if_needed(args, dataloader, indice, model(image))

            # Normalize.
            if args.metric_train == 'cosine':
                feats = F.normalize(feats, dim=1, p=2)

            B, C, H, W = feats.shape
            if i == 0:
                logger.info('Centroid size      : {}'.format(list(centroids.shape)))
                logger.info('Batch input size   : {}'.format(list(image.shape)))
                logger.info('Batch feature size : {}\n'.format(list(feats.shape)))

            # Compute distance and assign label. 
            scores  = compute_negative_euclidean(feats, centroids, metric_function) 

            # Save labels and count. 
            for idx, idx_img in enumerate(indice):
                counts += postprocess_label(args, K, idx, idx_img, scores, n_dual=view)

            if (i % 200) == 0:
                logger.info('[Assigning labels] {} / {}'.format(i, len(dataloader)))
    weight = counts / counts.sum()

    return weight

def compute_negative_euclidean(featmap, centroids, metric_function):
    centroids = centroids.unsqueeze(-1).unsqueeze(-1)
    return - (1 - 2*metric_function(featmap)\
                + (centroids*centroids).sum(dim=1).unsqueeze(0)) # negative l2 squared 

def get_metric_as_conv(centroids):
    N, C = centroids.size()

    centroids_weight = centroids.unsqueeze(-1).unsqueeze(-1)
    metric_function  = nn.Conv2d(C, N, 1, padding=0, stride=1, bias=False)
    metric_function.weight.data = centroids_weight
    metric_function = nn.DataParallel(metric_function)
    metric_function = metric_function.cuda()
    
    return metric_function



# def evaluate(args, logger, dataloader, classifier, model):
#     logger.info('====== METRIC TEST : {} ======\n'.format(args.metric_test))
#     histogram = np.zeros((args.K_test, args.K_test))

#     model.eval()
#     classifier.eval()
#     with torch.no_grad():
#         for i, (_, image, label) in enumerate(dataloader):
#             image = image.cuda(non_blocking=True)
#             feats = model(image)

#             if args.metric_test == 'cosine':
#                 feats = F.normalize(feats, dim=1, p=2)
            
#             B, C, H, W = feats.size()
#             if i == 0:
#                 logger.info('Batch input size   : {}'.format(list(image.shape)))
#                 logger.info('Batch label size   : {}'.format(list(label.shape)))
#                 logger.info('Batch feature size : {}\n'.format(list(feats.shape)))

#             probs = classifier(feats)
#             probs = F.interpolate(probs, label.shape[-2:], mode='bilinear', align_corners=False)
#             preds = probs.topk(1, dim=1)[1].view(B, -1).cpu().numpy()
#             label = label.view(B, -1).cpu().numpy()

#             histogram += scores(label, preds, args.K_test)
            
#             if i%20==0:
#                 logger.info('{}/{}'.format(i, len(dataloader)))
    
#     # Hungarian Matching. 
#     m = linear_assignment(histogram.max() - histogram)

#     # Evaluate. 
#     acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum() * 100

#     new_hist = np.zeros((args.K_test, args.K_test))
#     for idx in range(args.K_test):
#         new_hist[m[idx, 1]] = histogram[idx]
    
#     # NOTE: Now [new_hist] is re-ordered to 12 thing + 15 stuff classses. 
#     res1 = get_result_metrics(new_hist)
#     logger.info('ACC  - All: {:.4f}'.format(res1['overall_precision (pixel accuracy)']))
#     logger.info('mIOU - All: {:.4f}'.format(res1['mean_iou']))

#     # For Table 2 - partitioned evaluation.
#     if args.thing and args.stuff:
#         res2 = get_result_metrics(new_hist[:12, :12])
#         logger.info('ACC  - Thing: {:.4f}'.format(res2['overall_precision (pixel accuracy)']))
#         logger.info('mIOU - Thing: {:.4f}'.format(res2['mean_iou']))

#         res3 = get_result_metrics(new_hist[12:, 12:])
#         logger.info('ACC  - Stuff: {:.4f}'.format(res3['overall_precision (pixel accuracy)']))
#         logger.info('mIOU - Stuff: {:.4f}'.format(res3['mean_iou']))
    
#   return acc, res1


def get_faiss_module(args):
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False 
    cfg.device     = 0 #NOTE: Single GPU only. 
    idx = faiss.GpuIndexFlatL2(res, args.in_dim, cfg)

    return idx

def get_init_centroids(args, K, featlist, index):
    clus = faiss.Clustering(args.in_dim, K)
    clus.seed  = np.random.randint(args.seed)
    clus.niter = args.kmeans_n_iter
    clus.max_points_per_centroid = 10000000
    clus.train(featlist, index)

    return faiss.vector_float_to_array(clus.centroids).reshape(K, args.in_dim)

def module_update_centroids(index, centroids):
    index.reset()
    index.add(centroids)

    return index 

def fix_seed_for_reproducability(seed):
    """
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic. 

    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068 
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi 
    """
    # Use random seed.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def worker_init_fn(seed):
    return lambda x: np.random.seed(seed + x)    

def postprocess_label(args, K, idx, idx_img, scores, n_dual):
    out = scores[idx].topk(1, dim=0)[1].flatten().detach().cpu().numpy()

    # Save labels. 
    if not os.path.exists(os.path.join(args.save_model_path, 'label_' + str(n_dual))):
        os.makedirs(os.path.join(args.save_model_path, 'label_' + str(n_dual)))
    torch.save(out, os.path.join(args.save_model_path, 'label_' + str(n_dual), '{}.pkl'.format(idx_img)))
    
    # Count for re-weighting. 
    counts = torch.tensor(np.bincount(out, minlength=K)).float()

    return counts


def eqv_transform_if_needed(args, dataloader, indice, input):
    if args.equiv:
        input = dataloader.dataset.transform_eqv(indice, input)

    return input  


def get_transform_params(args):
    inv_list = []
    eqv_list = []
    if args.augment:
        if args.blur:
            inv_list.append('blur')
        if args.grey:
            inv_list.append('grey')
        if args.jitter:
            inv_list.extend(['brightness', 'contrast', 'saturation', 'hue'])
        if args.equiv:
            if args.h_flip:
                eqv_list.append('h_flip')
            if args.v_flip:
                eqv_list.append('v_flip')
            if args.random_crop:
                eqv_list.append('random_crop')
    
    return inv_list, eqv_list


def collate_train(batch):
    if batch[0][-1] is not None:
        indice = [b[0] for b in batch]
        image1 = torch.stack([b[1] for b in batch])
        image2 = torch.stack([b[2] for b in batch])
        label1 = torch.stack([b[3] for b in batch])
        label2 = torch.stack([b[4] for b in batch])

        return indice, image1, image2, label1, label2
    
    indice = [b[0] for b in batch]
    image1 = torch.stack([b[1] for b in batch])

    return indice, image1

def freeze_all(model):
    for param in model.module.parameters():
        param.requires_grad = False 


def initialize_classifier(args):
    classifier = get_linear(args.in_dim, args.K_train)
    classifier = nn.DataParallel(classifier)
    classifier = classifier.cuda()

    return classifier

def get_linear(indim, outdim):
    classifier = nn.Conv2d(indim, outdim, kernel_size=1, stride=1, padding=0, bias=True)
    classifier.weight.data.normal_(0, 0.01)
    classifier.bias.data.zero_()

    return classifier


def feature_flatten(feats):
    if len(feats.size()) == 2:
        # feature already flattened. 
        return feats
    
    feats = feats.view(feats.size(0), feats.size(1), -1).transpose(2, 1)\
            .contiguous().view(-1, feats.size(1))
    
    return feats 

def get_datetime(time_delta):
    days_delta = time_delta // (24*3600)
    time_delta = time_delta % (24*3600)
    hour_delta = time_delta // 3600 
    time_delta = time_delta % 3600 
    mins_delta = time_delta // 60 
    time_delta = time_delta % 60 
    secs_delta = time_delta 

    return '{}:{}:{}:{}'.format(days_delta, hour_delta, mins_delta, secs_delta)

