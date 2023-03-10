from ctypes import Structure
from inspect import CO_VARARGS
from attr import attr
from numpy import size, ones, zeros, nan_to_num
from sqlalchemy import true
import torch
import torch.nn as nn
import torchvision
import os
from models import resnet, mobilenet
from utils import onehot_to_class
from scipy.ndimage.measurements import label        


BatchNorm2d = torch.nn.BatchNorm2d

def conv3x3_bn_relu(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=stride, padding=1, bias=bias),
        BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

def count_regions(pred, is_label=False):
    if not is_label:
        _, pred = torch.max(pred, dim=0)
    array = pred.cpu().detach().numpy()
    struct = ones((3,3,), dtype=int)
    n = zeros(array.shape[0])
    for i in range(array.shape[0]):
        _, n[i] = label(array[i], struct)
    return torch.tensor(n, requires_grad=True) 

class UnsupervisedSegmentationBase(nn.Module):
    def __init__(self):
        super(UnsupervisedSegmentationBase, self).__init__()

class UnsupervisedSegmentationModule(UnsupervisedSegmentationBase):
    def __init__(self, net_dec_cloudseg, scale = 4):
        super(UnsupervisedSegmentationModule, self).__init__()
        self.decoder_cloud_seg = net_dec_cloudseg
        self.scale = scale
        self.crit = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, feed_dict, feat_map, skyseg_result, seg_size=None):
        if type(feed_dict) is list:
            feed_dict = feed_dict[0]
            # also, convert to torch.cuda.FloatTensor
            if torch.cuda.is_available():
                feed_dict['img_data'] = feed_dict['img_data'].cuda()
                feed_dict['seg_label'] = feed_dict['seg_label'].cuda()
                feed_dict['cloud_label'] = feed_dict['cloud_label'].cuda()
            else:
                raise RuntimeError('Cannot convert torch.Floattensor into torch.cuda.FloatTensor')
        _, sky_mask = torch.max(skyseg_result, dim=1)
        sky_mask = sky_mask.unsqueeze(dim=1)
        x_mask = nn.functional.interpolate(
            sky_mask.float(), size=(feat_map[-1].shape[2],feat_map[-1].shape[3]), mode='bilinear', align_corners=False)
        x_mask = x_mask.cuda()
        feat_map[-1] = x_mask*feat_map[-1]
        pred = self.decoder_cloud_seg(feat_map, seg_size=seg_size)
        if self.training:
            cloud_loss = self.crit(pred, feed_dict['cloud_label'])
            return pred, cloud_loss
        else: 
            pred = nn.functional.log_softmax(pred)
            return pred

class MultiLabelModuleBase(nn.Module):
    def __init__(self):
        super(MultiLabelModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        if type(pred) is list:
            pred = pred[-1]
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class MultiLabelModule(MultiLabelModuleBase):
    def __init__(self, net_enc, net_dec_seg, net_dec_attr, crit_sky, crit_attr, deep_sup_scale=None):
        super(MultiLabelModule, self).__init__()
        self.encoder = net_enc
        self.decoder_sky_seg = net_dec_seg
        self.decoder_attributes = net_dec_attr
        self.crit_sky = crit_sky
        self.crit_attr = crit_attr
        self.deep_sup_scale = deep_sup_scale

    def attribute_loss(self, pred, label, crit):
        loss_time = loss_season = loss_weather = 0
        label = label.view(-1, 2)
        loss_time = torch.nan_to_num(crit(pred[0], label[0]))
        loss_season = torch.nan_to_num(crit(pred[1], label[1]))
        loss_weather = torch.nan_to_num(crit(pred[2], label[2]))    
        return [loss_time, loss_season, loss_weather]

    def segmentation_loss(self, pred, label, crit):
        frag_weight = 0.1
        crit_frag = nn.L1Loss()
        if torch.sum(label) > 0:
            seg_loss = crit(pred, label)
            frag_loss = crit_frag(count_regions(pred), count_regions(label, True))
        else:
            seg_loss = 0
            frag_loss = crit_frag(torch.tensor(count_regions(pred), requires_grad=True), torch.tensor((2,2)))
        return seg_loss + frag_weight * frag_loss

    def forward(self, feed_dict, *, seg_size=None):
        if type(feed_dict) is list:
            feed_dict = feed_dict[0]
            if torch.cuda.is_available():
                feed_dict['img_data'] = feed_dict['img_data'].cuda()
                feed_dict['seg_label'] = feed_dict['seg_label'].cuda()
                feed_dict['attr_label'] = feed_dict['attr_label'].cuda()
            else:
                raise RuntimeError('Cannot convert torch.Floattensor into torch.cuda.FloatTensor')
        #training
        if self.training:
            encoder_result = self.encoder(feed_dict['img_data'], return_feature_maps=True)
            pred_attr, pred = self.decoder_attributes(encoder_result)
            pred_seg = self.decoder_sky_seg(encoder_result, pred, seg_size=seg_size)
            seg_result = nn.functional.log_softmax(pred_seg, dim=1)
            try:
                loss_attr = self.attribute_loss(pred_attr, feed_dict['attr_label'], self.crit_attr)
            except RuntimeError as e:
                print('RunTimeError: {}'.format(e))
                loss_attr = [0, 0, 0]
            loss_sky = self.segmentation_loss(pred_seg, feed_dict['seg_label'], self.crit_sky)
            acc_sky = self.pixel_acc(pred_seg, feed_dict['seg_label'])
            return encoder_result, loss_sky, sum(loss_attr) / 3, acc_sky, pred_seg, pred_attr
        # inference
        else:
            encoder_result = self.encoder(feed_dict['img_data'], return_feature_maps=True)
            pred_attr, pred = self.decoder_attributes(encoder_result)
            pred_attr = [nn.functional.softmax(pred_attr_) for pred_attr_ in pred_attr]
            pred_seg = self.decoder_sky_seg(encoder_result, pred, seg_size=seg_size)
            pred = [pred_attr, pred_seg]
            return encoder_result, pred


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, arch='vgg16_dilated', fc_dim=1024, weights=''):
        pretrained = True if len(weights) == 0 else False
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(torch.load(weights))
        return net_encoder

    def build_decoder(self, arch='', fc_dim=1024, num_class=150,
                      segSize=384, weights='', use_softmax=False):

        print('Building: {}'.format(arch))
        if arch == 'attribute_head':
            net_decoder = AttributeHead(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)           
        elif arch == 'sky_seg_head':
            net_decoder = SkySegHead(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'cloud_seg_head':
            net_decoder = CloudSegHead(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)                                                      
        else:
            raise Exception('Architecture undefined!')

        if os.path.exists(weights):
            print('Loading weights for net_decoder {}'.format(arch))
            net_decoder.load_state_dict(torch.load(weights))
        else: 
            print('Initializing weights for net_decoder {}'.format(arch))
            net_decoder.apply(self.weights_init)    
        return net_decoder


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, original_resnet, dilate_scale=8, dropout2d=False):
        super(ResnetDilated, self).__init__()
        self.dropout2d = dropout2d
        from functools import partial

        if dilate_scale == 8:
            original_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            original_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            original_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        self.features = nn.Sequential(*list(original_resnet.children())[:-2])

        if self.dropout2d:
            self.dropout = nn.Dropout2d(0.5)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.features(x)
        if self.dropout2d:
            x = self.dropout(x)
        return x

class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out
        else:
            return [self.features(x)]


class CloudSegHead(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 4),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(CloudSegHead, self).__init__()
        self.use_softmax = use_softmax

        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales) * 512, fpn_dim, 1)

        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: 
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for _ in range(len(fpn_inplanes) - 1): 
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, seg_size=None):
        x = conv_out[-1]
        input_size = x.size()
        ppm_out = [x]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(x),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) 

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) 
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        x = nn.functional.interpolate(
            x, size=seg_size, mode='bilinear', align_corners=False)

        if self.use_softmax: 
            x = nn.functional.softmax(x, dim=1)
            return x

        return x


class AttributeHead(nn.Module):
    def __init__(self, num_class=(4, 4, 4), fc_dim=4096, segSize=384,
                 use_softmax=False):
        super(AttributeHead, self).__init__()
        self.segSize = segSize
        self.use_softmax = use_softmax
        self.num_class = num_class
        self.multi_label_pred = isinstance(self.num_class, list) or isinstance(self.num_class, tuple)

        # multiple branches for each category
        if self.multi_label_pred:
            self.conv = nn.ModuleList(
                [nn.Sequential(
                    # convs, dilation=2
                    nn.Conv2d(fc_dim, fc_dim, 3, 1, 2, 2, bias=False),
                    nn.BatchNorm2d(fc_dim, momentum=0.1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(fc_dim, fc_dim, 3, 1, 2, 2, bias=False),
                    nn.BatchNorm2d(fc_dim, momentum=0.1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(fc_dim, fc_dim, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(fc_dim, momentum=0.1),
                    nn.ReLU(inplace=True),
                    )
                for _ in self.num_class]
            )
            self.pooling = nn.ModuleList(
                [nn.AdaptiveAvgPool2d(output_size= (1, 1)) for _ in self.num_class]
            )
            self.conv_last = nn.ModuleList(
            [nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=fc_dim, out_features=n_c + 1))
                for n_c in self.num_class]
            )
        else:
            self.conv_last = nn.Conv2d(fc_dim, self.num_class, 1, 1, 0, bias=False)

    def forward(self, conv_out):
        x = conv_out[-1]
        if self.multi_label_pred:
            pred = []
            for i in range(len(self.num_class)):
                _x = self.conv[i](x)
                _xp = self.pooling[i](_x)
                _xp = torch.flatten(_xp, 1)
                pred.append(self.conv_last[i](_xp))
        else: 
            pred = self.conv_last(x)
        return pred, _x

class SkySegHead(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 4),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(SkySegHead, self).__init__()
        self.use_softmax = use_softmax

        self.fuse_attr = conv3x3_bn_relu(4096, 2048, 1)

        # Pooling
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales) * 512, fpn_dim, 1)

        # Feature fusion
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for _ in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, attr_result=None, seg_size=None):
        x = conv_out[-1]
        if (attr_result.nelement() > 0):
            x = torch.cat([x, attr_result], 1)
            x = self.fuse_attr(x)


        input_size = x.size()
        ppm_out = [x]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(x),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) 

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) 
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        x = nn.functional.interpolate(
            x, size=seg_size, mode='bilinear', align_corners=False)

        if self.use_softmax:  # is True during inference
            x = nn.functional.softmax(x, dim=1)
            return x

        return x


class AttributeHead(nn.Module):
    def __init__(self, num_class=(4, 4, 4), fc_dim=4096, segSize=384,
                 use_softmax=False):
        super(AttributeHead, self).__init__()
        self.segSize = segSize
        self.use_softmax = use_softmax
        self.num_class = num_class
        self.multi_label_pred = isinstance(self.num_class, list) or isinstance(self.num_class, tuple)

        # one branch for each category
        if self.multi_label_pred:
            self.conv = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(fc_dim, fc_dim, 3, 1, 2, 2, bias=False),
                    nn.BatchNorm2d(fc_dim, momentum=0.1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(fc_dim, fc_dim, 3, 1, 2, 2, bias=False),
                    nn.BatchNorm2d(fc_dim, momentum=0.1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(fc_dim, fc_dim, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(fc_dim, momentum=0.1),
                    nn.ReLU(inplace=True),
                    )
                for _ in self.num_class]
            )
            self.pooling = nn.ModuleList(
                [nn.AdaptiveAvgPool2d(output_size= (1, 1)) for _ in self.num_class]
            )
            self.conv_last = nn.ModuleList(
            [nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=fc_dim, out_features=n_c + 1))
                for n_c in self.num_class]
            )
        else:
            self.conv_last = nn.Conv2d(fc_dim, self.num_class, 1, 1, 0, bias=False)

    def forward(self, conv_out):
        x = conv_out[-1]
        if self.multi_label_pred:
            pred = []
            for i in range(len(self.num_class)):
                _x = self.conv[i](x)
                _xp = self.pooling[i](_x)
                _xp = torch.flatten(_xp, 1)
                pred.append(self.conv_last[i](_xp))
        else: 
            pred = self.conv_last(x)

        # if self.use_softmax:
        #     x = nn.functional.softmax(x)
        # else:
        #     x = nn.functional.log_softmax(x)
        return pred, _x

