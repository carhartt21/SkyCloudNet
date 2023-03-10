from operator import attrgetter
import os
import json
import string
from typing import Iterator
from sqlalchemy import false, null
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL.Image import DecompressionBombError
from PIL import ImageFile, UnidentifiedImageError
from custom_transforms import *
ImageFile.LOAD_TRUNCATED_IMAGES = True



#for testing
from config import cfg

def show_batch(sample_batched):
    '''Show image with segmentation for a batch of samples.'''
    images_batch, segm_batch = sample_batched['img_data'], sample_batched['seg_label']
    # im_size = images_batch.size(2)
    fig = plt.figure()
    grid1 = utils.make_grid(images_batch)
    grid2 = utils.make_grid(segm_batch)
    # fig.add_subplot(1, 2, 1)
    plt.imshow(grid1.numpy().transpose((1, 2, 0)))
    # fig.add_subplot(1, 2, 2)
    plt.imshow(grid2.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')
    plt.axis('off')
    plt.ioff()
    plt.show()


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.Resampling.NEAREST
    elif interp == 'bilinear':
        resample = Image.Resampling.BILINEAR
    elif interp == 'bicubic':
        resample = Image.Resampling.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.parse_input_list(odgt, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            with open(odgt, 'r') as input_list:
                self.list_sample = json.load(input_list)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def transform_image(self, index, image):
        # Base transform
        image = self.transform_base(index, image)

        if self.mode == 'compute':
            if self.view == 1:
                image = self.transform_inv(index, image, 0)
                image = self.transform_tensor(image)
            elif self.view == 2:
                image = self.transform_inv(index, image, 1)
                image = TF.resize(image, self.res1, Image.BILINEAR)
                image = self.transform_tensor(image)
            else:
                raise ValueError('View [{}] is an invalid option.'.format(self.view))
            return (image, )
        elif 'train' in self.mode:
            # Invariance transform. 
            image1 = self.transform_inv(index, image, 0)
            image1 = self.transform_tensor(image1)

            if self.mode == 'baseline_train':
                return (image1, )
            
            image2 = self.transform_inv(index, image, 1)
            image2 = TF.resize(image2, self.res1, Image.BILINEAR)
            image2 = self.transform_tensor(image2)

            return (image1, image2)
        else:
            raise ValueError('Mode [{}] is an invalid option.'.format(self.mode))


    def transform_eqv(self, indice, image):
        if 'random_crop' in self.eqv_list:
            image = self.random_resized_crop(indice, image)
        if 'h_flip' in self.eqv_list:
            image = self.random_horizontal_flip(indice, image)
        if 'v_flip' in self.eqv_list:
            image = self.random_vertical_flip(indice, image)

        return image


    def init_transforms(self):
        N = self.num_sample
        
        # Transforms for equivariance

        self.random_horizontal_flip = RandomHorizontalTensorFlip(N=N)
        self.random_vertical_flip   = RandomVerticalFlip(N=N)
        self.random_resized_crop    = RandomResizedCrop(N=N, res=self.res1, scale=self.scale)

        # Tensor transform. 
        self.transform_tensor = TensorTransform()
    

    def transform_label(self, index):
        # TODO Equiv. transform.
        if self.mode == 'train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)))
            label2 = torch.load(os.path.join(self.labeldir, 'label_2', '{}.pkl'.format(index)))
            label1 = torch.LongTensor(label1)
            label2 = torch.LongTensor(label2)

            X1 = int(np.sqrt(label1.shape[0]))
            X2 = int(np.sqrt(label2.shape[0]))
            
            label1 = label1.view(X1, X1)
            label2 = label2.view(X2, X2)

            return label1, label2

        elif self.mode == 'baseline_train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)))
            label1 = torch.LongTensor(label1)

            X1 = int(np.sqrt(label1.shape[0]))
            
            label1 = label1.view(X1, X1)

            return (label1, )

        return (None, )

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor
        segm = (np.float32(np.array(segm)) / 255.)
        segm = torch.from_numpy(np.array(segm)).long()

        return segm

    def cloud_segm_transform(self, segm):
        # to tensor
        segm = np.float32(np.array(segm))-1
        segm = torch.from_numpy(np.array(segm)).long()

        return segm

    def attr_transform(self, attr, threshold = 0.75):
        # to tensor, -1 to num_attributes - 1
        attr_c = []
        for c in zip(*(iter(attr),) * 4):
            _max = np.argmax(c)
            if c[_max]>=threshold:
                attr_c.append((np.argmax(c) + 1))
            elif c[_max]<0:
                attr_c.append(-1)
            else:
                attr_c.append(0)
        attr = torch.from_numpy(np.array(attr_c)).long()
        return attr

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, batch_per_gpu=1, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.num_attributes = opt.num_attr_class
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate BATCH height and width
        # the batch's h and w shall be larger than each sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), \
                self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Padding both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'
        batch_info = np.empty((self.batch_per_gpu), dtype = 'object')
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()
        batch_cloud_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()            
        batch_attr = torch.zeros(
            self.batch_per_gpu,
            len(list(self.num_attributes))).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = this_record['fpath_img']
            segm_path = this_record['fpath_segm']
            attr_path = this_record['fpath_attr']
            try:
                cloud_segm_path = this_record['fpath_cloud']
            except KeyError:
                cloud_segm_path = '' 

            try: 
                img = Image.open(image_path).convert('RGB')
            except UnidentifiedImageError as e:
                print(e)
                continue
            except OSError as e:
                print(e)
                continue
            
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')

            if not segm_path == '':
                segm = Image.open(segm_path).convert('L')
            else:
                segm = np.zeros(0)

            if not cloud_segm_path == '':
                cloud_segm = Image.open(cloud_segm_path).convert('L')
            else:
                cloud_segm = np.zeros(0)

            if not attr_path == '':     
                attr = np.load(attr_path)
                assert(attr.size != 0)
            else: 
                attr = np.zeros(0)

            # random_flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if segm.size != 0:
                    segm = segm.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # note that each sample within a mini batch has different scale param
            if segm.size != 0:
                segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

                # further downsample seg label to avoid seg label misalignment during loss calcualtion
                segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
                segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
                segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
                segm_rounded.paste(segm, (0, 0))
                segm = imresize(
                    segm_rounded,
                    (segm_rounded.size[0] // self.segm_downsampling_rate, \
                    segm_rounded.size[1] // self.segm_downsampling_rate), \
                    interp='nearest')

            # note that each sample within a mini batch has different scale param
            if cloud_segm.size != 0:
                cloud_segm = imresize(cloud_segm, (batch_widths[i], batch_heights[i]), interp='nearest')

                # further downsample seg label to avoid seg label misalignment during loss calcualtion
                segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
                segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
                segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
                segm_rounded.paste(segm, (0, 0))
                cloud_segm = imresize(
                    cloud_segm,
                    (cloud_segm.size[0] // self.segm_downsampling_rate, \
                    cloud_segm.size[1] // self.segm_downsampling_rate), \
                    interp='nearest')

            # image transform, to torch float tensor 3xHxW
            img = self.img_transform(img)

            # segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)

            # segm cloud_transform, to torch long tensor HxW
            cloud_segm = self.cloud_segm_transform(cloud_segm)
            
            # attr to torch tensor
            attr = self.attr_transform(attr)

            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_info[i] = this_record['fpath_img']
            if segm.nelement() != 0 :
                batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm
            else:
                batch_segms[i][:batch_height, :batch_width] = -1

            if cloud_segm.nelement() != 0 :
                batch_cloud_segms[i][:cloud_segm.shape[0], :cloud_segm.shape[1]] = cloud_segm
            else:
                batch_cloud_segms[i][:batch_height, :batch_width] = -1                

            if attr.nelement() != 0:
                batch_attr[i][:attr.shape[0]] = attr
            else:
                batch_attr[i][:3] = 0                

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        output['cloud_label'] = batch_cloud_segms
        output['attr_label'] = batch_attr
        output['info'] = batch_info
        return output

    def __len__(self):
        return int(1e10) # It's a fake length, every loader maintains its own list
        #return self.num_sampleclass

if __name__ == '__main__':

    cfg.merge_from_file('/home/chge7185/repositories/outdoor_attribute_estimation/config/config.yaml')

    dataset_train = TrainDataset(
        '/home/chge7185/repositories/outdoor_attribute_estimation/data',
        '/home/chge7185/repositories/outdoor_attribute_estimation/data/imageList.odgt',
        cfg.DATASET,
        batch_per_gpu=2)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=False,  # parameter is not used
        # collate_fn=user_scattered_collate,
        num_workers=1,
        drop_last=True,
        pin_memory=True)

    # print(next(iterator_train))


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, spatial=False, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        self.spatial = spatial

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        attr_path = os.path.join(self.root_dataset, this_record['fpath_attr'])
        try:
            cloud_segm_path = os.path.join(self.root_dataset, this_record['fpath_cloud'])
            if this_record['fpath_cloud'] != '':
                cloud_segm = Image.open(cloud_segm_path).convert('L')
            else:            
                cloud_segm = np.zeros(0)
        except KeyError:
            cloud_segm = np.zeros(0)

        try: 
            img = Image.open(image_path).convert('RGB')
        except UnidentifiedImageError as e:
            print(e)

        if this_record['fpath_segm'] != '':
            segm = Image.open(segm_path).convert('L')
        else:
            segm = np.zeros(0)
                
        if this_record['fpath_attr'] != '':     
            attr = np.load(attr_path)
            assert(attr.size != 0)
        else: 
            attr = np.zeros(0)

        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path).convert('L')

        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)
            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)
            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')
                       
            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)


        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        # segm cloud_transform, to torch long tensor HxW
        cloud_segm = self.cloud_segm_transform(cloud_segm)
        batch_cloud_segms = torch.unsqueeze(cloud_segm, 0)

        # attr to torch tensor
        attr = self.attr_transform(attr)
        batch_attr = torch.unsqueeze(attr, 0)                  
        
        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['seg_label'] = batch_segms
        output['attr_label'] = batch_attr
        output['cloud_label'] = batch_cloud_segms
        output['info'] = os.path.join(self.root_dataset, this_record['fpath_img'])

        return output

    def __len__(self):
        return self.num_sample
