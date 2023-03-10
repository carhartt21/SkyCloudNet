import os
import time
import argparse
from packaging import version
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Our libs
from config import cfg
from data import ValDataset
from model import ModelBuilder, MultiLabelModule, UnsupervisedSegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
# from confusion_matrix import ConfusionMatrix

def visualize_result(data, pred, dir_result, cls_info, epoch=''):
    colors = []
    for c in cls_info:
        colors.append(cls_info[c]['color'])
    (img, label, info) = data
    # label
    label_color = colorEncode(label, colors).astype(np.uint8)
    # prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)
    # aggregate images
    im_vis = np.concatenate((img, label_color, pred_color),
                            axis=1).astype(np.uint8)
    img_name = info.split('/')[-1]
    #save prediction individually 
    Image.fromarray(pred_color).save(os.path.join(dir_result, '{}_{}_pred.png'.format(img_name[:-4], epoch[:-4])))
    #save composition 
    Image.fromarray(im_vis).save(os.path.join(dir_result, '{}_{}_comp.png'.format(img_name[:-4], epoch[:-4])))


def evaluate(multilabel_module, cloud_seg_module, loader, cfg, gpu):
    with open(cfg.DATASET.classInfo) as f:
        cls_info = json.load(f)
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    multilabel_module.eval()
    cloud_seg_module.eval()
    pbar = tqdm(total=len(loader), ascii=True, desc='Running evaluation')
    for batch_data in loader:
        batch_data = batch_data[0]
        sky_seg_label = as_numpy(batch_data['seg_label'][0])
        cloud_seg_label = as_numpy(batch_data['cloud_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            seg_size = (sky_seg_label.shape[0], sky_seg_label.shape[1])
            pred_seg = torch.zeros(1, cfg.DATASET.num_seg_class, seg_size[0], seg_size[1])
            pred_seg = async_copy_to(pred_seg, gpu)


            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                encoder_result, [_, pred_seg]  = multilabel_module(feed_dict, seg_size=seg_size)
                pred_cloud = cloud_seg_module(batch_data, encoder_result, pred_seg, seg_size=seg_size)  

        if cfg.VAL.sky_seg:
            _, sky_pred = torch.max(pred_seg, dim=1)
            sky_pred = as_numpy(sky_pred.squeeze(0).cpu())
            acc, pix = accuracy(sky_pred, sky_seg_label)
            intersection, union = intersectionAndUnion(sky_pred, sky_seg_label, cfg.DATASET.num_seg_class)

        if cfg.VAL.cloud_seg:
            _, cloud_pred = torch.max(pred_cloud, dim=1)
            cloud_pred = as_numpy(cloud_pred.squeeze(0).cpu())
            acc, pix = accuracy(cloud_pred, cloud_seg_label)
            intersection, union = intersectionAndUnion(cloud_pred, cloud_seg_label, cfg.DATASET.num_seg_class + cfg.DATASET.num_cloud_class)


        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # visualization
        if cfg.VAL.visualize:
            if cfg.VAL.sky_seg:
                if not os.path.isdir(os.path.join(cfg.DIR, 'result/sky')):
                    os.makedirs(os.path.join(cfg.DIR, 'result/sky'))                
                visualize_result(
                    (batch_data['img_ori'], sky_seg_label, batch_data['info']),
                    sky_pred, os.path.join(cfg.DIR, 'result/sky'), cls_info, cfg.VAL.checkpoint)                   
            if cfg.VAL.cloud_seg:
                if not os.path.isdir(os.path.join(cfg.DIR, 'result/cloud')):
                    os.makedirs(os.path.join(cfg.DIR, 'result/cloud'))                   
                visualize_result(
                    (batch_data['img_ori'], cloud_seg_label, batch_data['info']),
                    cloud_pred, os.path.join(cfg.DIR, 'result/cloud'), cls_info, cfg.VAL.checkpoint)
        pbar.update(1)

    # summary
    logger.info('[Eval Summary]:')
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        logger.info('class {}, IoU: {:.4f}'.format(cls_info[str(i)]['name'], _iou))

    logger.info('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average() * 100, time_meter.average()))
    return


def main(cfg, gpu):
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.enabled = False
    
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder_attr = builder.build_decoder(
        arch='attribute_head',
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_attr_class,
        weights=cfg.MODEL.weights_decoder_attr) 
    net_decoder_skyseg = builder.build_decoder(
        arch='sky_seg_head',
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_seg_class,
        weights=cfg.MODEL.weights_decoder_skyseg)
    net_decoder_cloudseg = builder.build_decoder(
        arch='cloud_seg_head',
        fc_dim=cfg.MODEL.fc_dim,
        num_class= cfg.DATASET.num_seg_class + cfg.DATASET.num_cloud_class,
        weights=cfg.MODEL.weights_decoder_cloudseg
        )        

    crit = nn.NLLLoss(ignore_index=-1)

    multilabel_module = MultiLabelModule(
            net_encoder, net_decoder_skyseg, net_decoder_attr, crit, crit)

    cloud_seg_module = UnsupervisedSegmentationModule(net_decoder_cloudseg)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    multilabel_module.cuda()
    cloud_seg_module.cuda()

    # Main loop
    evaluate(multilabel_module, cloud_seg_module, loader_val, cfg, gpu)

    logger.info('Evaluation Done!')


if __name__ == '__main__':
    assert version.Version(torch.__version__) >= version.Version('1.4.0'), \
        'PyTorch>=1.4.0 is required'

    parser = argparse.ArgumentParser(
        description='PyTorch Semantic Segmentation Validation'
    )
    parser.add_argument(
        '--cfg',
        default='config/config.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        '--gpu',
        default=0,
        help='gpu to use'
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    logger = setup_logger(distributed_rank=0)   
    logger.info('Loaded configuration file {}'.format(args.cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_{}'.format(cfg.VAL.checkpoint))
    cfg.MODEL.weights_decoder_attr = os.path.join(
        cfg.DIR, 'decoder_attr_{}'.format(cfg.VAL.checkpoint))        
    cfg.MODEL.weights_decoder_skyseg = os.path.join(
        cfg.DIR, 'decoder_seg_{}'.format(cfg.VAL.checkpoint))
    cfg.MODEL.weights_decoder_cloudseg = os.path.join(
        cfg.DIR, 'decoder_cloudseg_{}'.format(cfg.VAL.checkpoint))           
    assert os.path.exists(cfg.MODEL.weights_encoder), 'checkpoint does not exist!'
    assert os.path.exists(cfg.MODEL.weights_decoder_skyseg), 'checkpoint does not exist!'
    assert os.path.exists(cfg.MODEL.weights_decoder_attr), 'checkpoint does not exist!'

    if not os.path.isdir(os.path.join(cfg.DIR, 'result')):
        os.makedirs(os.path.join(cfg.DIR, 'result'))

    cfg.freeze()
    main(cfg, args.gpu)
