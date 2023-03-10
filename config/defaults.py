from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/ade20k-resnet50dilated-ppm_deepsup"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/"
_C.DATASET.list_train = "./data/training.odgt"
_C.DATASET.list_val = "./data/validation.odgt"
_C.DATASET.list_stats = ""
_C.DATASET.num_attr_class = (4, 4, 5)
_C.DATASET.num_seg_class = 2
_C.DATASET.num_cloud_class = 2
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.imgSizes = (300, 375, 450, 525, 600)
# maximum input image size of long edge
_C.DATASET.imgMaxSize = 1200
# maxmimum downsampling rate of the network
_C.DATASET.padding_constant = 8
# downsampling rate of the segmentation label
_C.DATASET.segm_downsampling_rate = 8
# randomly horizontally flip images when train/test
_C.DATASET.random_flip = True
# randomly crop images if the size exceeds the maximum size
_C.DATASET.random_crop = True
# info file
_C.DATASET.classInfo = "./data/ADE20kClasses.json"

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.arch_encoder = "resnet50"
# weights to finetune net_decoder
_C.MODEL.weights_decoder_attr = ""
_C.MODEL.weights_decoder_skyseg = ""
_C.MODEL.weights_decoder_cloudseg = ""
# number of feature channels between encoder and decoders
_C.MODEL.fc_dim = 2048

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size_per_gpu = 2
# epochs to train for
_C.TRAIN.num_epoch = 20
# epoch to start training. useful if continuing from checkpoint
_C.TRAIN.start_epoch = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000

_C.TRAIN.optim = "SGD"
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_decoder_attr = 0.02
_C.TRAIN.lr_decoder_seg = 0.02
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16
# frequency to display
_C.TRAIN.disp_iter = 50
# frequency for saving segmentation results  
_C.TRAIN.disp_results = 0
# manual seed
_C.TRAIN.seed = 304
# use validation during training
_C.TRAIN.eval = False
# step size for validation calculation
_C.TRAIN.eval_step = 5
# best score from previous Training
_C.TRAIN.best_score = 0
# optimizer data from previous training
_C.TRAIN.optim_data = ""
# use ground truth data during training of the ocr module

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 1
# calculate metrics for sky segmentation
_C.VAL.sky_seg = False
# calculate metrics for cloud segmentation
_C.VAL.cloud_seg = False
# output preddiction results
_C.VAL.visualize = False
# the checkpoint to evaluate on
_C.VAL.checkpoint = "20"


# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = "20"
# folder to output visualization results
_C.TEST.result = "./"