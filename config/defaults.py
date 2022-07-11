from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
_C.MODEL.PRETRAIN_HW_RATIO = 1
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.IF_FEAT_CAT = False
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.GEM_POOLING = False
_C.MODEL.STEM_CONV = False
_C.MODEL.REL_POS = False
_C.MODEL.REL_CLS = False
_C.MODEL.REL_ABS = False
_C.MODEL.REL_CTX = False
_C.MODEL.HEAD_NUM = 12
_C.MODEL.ABS_POS = True
# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# ML Parameter
_C.MODEL.ML = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Name of scheduler
_C.SOLVER.SCHEDULER_NAME = "cos"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# Patch ratio for Patch triplet loss
_C.SOLVER.PATCH_RATIO = [0.5,0.9]
# Loss ratio for triplet loss
_C.SOLVER.LOSS_RATIO = 0.3
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs   
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64
# Use L2 normalized feature in Triplet loss
_C.SOLVER.FEAT_NORM = False
_C.SOLVER.COMB = False
_C.SOLVER.COMB_INDEX = 5
_C.SOLVER.JSD = False
_C.SOLVER.HEAD_WISE = False
_C.SOLVER.REPLACEMENT = False
_C.SOLVER.MEAN_POS = False
# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
_C.TEST.VISUALIZE = False
_C.TEST.VISUALIZE_RANK  = 0
_C.TEST.VISUALIZE_TYPE = 0 
_C.TEST.VISUALIZE_INDEX = 0
_C.TEST.VISUALIZE_METRIC = ""
_C.TEST.HEAD_FUSION = 'max'
_C.TEST.DISCARD_RATIO = 0.9
_C.TEST.EXPLAIN = False
_C.TEST.EVAL_METRIC = ""
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""

# Model INDEX
_C.INDEX = None

# Weight and Biase debug
_C.WANDB = False


_C_test = CN()
# ----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C_test.MODEL = CN()
# Using cuda or cpu for training
_C_test.MODEL.DEVICE = "cuda"
# ID number of GPU
_C_test.MODEL.DEVICE_ID = '0'
# Name of backbone
_C_test.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C_test.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C_test.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C_test.MODEL.PRETRAIN_CHOICE = 'imagenet'
_C_test.MODEL.PRETRAIN_HW_RATIO = 1
# If train with BNNeck, options: 'bnneck' or 'no'
_C_test.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C_test.MODEL.IF_WITH_CENTER = 'no'

_C_test.MODEL.ID_LOSS_TYPE = 'softmax'
_C_test.MODEL.ID_LOSS_WEIGHT = 1.0
_C_test.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C_test.MODEL.IF_FEAT_CAT = False
_C_test.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C_test.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C_test.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C_test.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C_test.MODEL.COS_LAYER = False

# Transformer setting
_C_test.MODEL.DROP_PATH = 0.1
_C_test.MODEL.DROP_OUT = 0.0
_C_test.MODEL.ATT_DROP_RATE = 0.0
_C_test.MODEL.TRANSFORMER_TYPE = 'None'
_C_test.MODEL.STRIDE_SIZE = [16, 16]
_C_test.MODEL.GEM_POOLING = False
_C_test.MODEL.STEM_CONV = False
_C_test.MODEL.REL_POS = False
_C_test.MODEL.REL_CLS = False
_C_test.MODEL.REL_ABS = False
_C_test.MODEL.REL_CTX = False
_C_test.MODEL.HEAD_NUM = 12
_C_test.MODEL.ABS_POS = True
# JPM Parameter
_C_test.MODEL.JPM = False
_C_test.MODEL.SHIFT_NUM = 5
_C_test.MODEL.SHUFFLE_GROUP = 2
_C_test.MODEL.DEVIDE_LENGTH = 4
_C_test.MODEL.RE_ARRANGE = True

# SIE Parameter
_C_test.MODEL.SIE_COE = 3.0
_C_test.MODEL.SIE_CAMERA = False
_C_test.MODEL.SIE_VIEW = False

# ML Parameter
_C_test.MODEL.ML = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C_test.INPUT = CN()
# Size of the image during training
_C_test.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C_test.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C_test.INPUT.PROB = 0.5
# Random probability for random erasing
_C_test.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C_test.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C_test.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C_test.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C_test.DATASETS = CN()
# List of the dataset names for training, as present in paths_c_testatalog.py
_C_test.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C_test.DATASETS.ROOT_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C_test.DATALOADER = CN()
# Number of data loading threads
_C_test.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C_test.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C_test.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C_test.SOLVER = CN()
# Name of optimizer
_C_test.SOLVER.OPTIMIZER_NAME = "Adam"
# Name of scheduler
_C_test.SOLVER.SCHEDULER_NAME = "cos"
# Number of max epoches
_C_test.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C_test.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C_test.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C_test.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C_test.SOLVER.SEED = 1234
# Momentum
_C_test.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C_test.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C_test.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C_test.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# Patch ratio for Patch triplet loss
_C_test.SOLVER.PATCH_RATIO = [0.5,0.9]
# Loss ratio for triplet loss
_C_test.SOLVER.LOSS_RATIO = 0.3
# Settings of weight decay
_C_test.SOLVER.WEIGHT_DECAY = 0.0005
_C_test.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C_test.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C_test.SOLVER.STEPS = (40, 70)
# warm up factor
_C_test.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs   
_C_test.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C_test.SOLVER.WARMUP_METHOD = "linear"

_C_test.SOLVER.COSINE_MARGIN = 0.5
_C_test.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C_test.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C_test.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C_test.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C_test.SOLVER.IMS_PER_BATCH = 64
# Use L2 normalized feature in Triplet loss
_C_test.SOLVER.FEAT_NORM = False
_C_test.SOLVER.COMB = False
_C_test.SOLVER.COMB_INDEX = 5
_C_test.SOLVER.JSD = False
_C_test.SOLVER.HEAD_WISE = False
_C_test.SOLVER.REPLACEMENT = False
_C_test.SOLVER.MEAN_POS = False
# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C_test.TEST = CN()
# Number of images per batch during test
_C_test.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C_test.TEST.RE_RANKING = False
# Path to trained model
_C_test.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C_test.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C_test.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C_test.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C_test.TEST.EVAL = False
_C_test.TEST.VISUALIZE = False
_C_test.TEST.VISUALIZE_RANK  = 0
_C_test.TEST.VISUALIZE_TYPE = 0 
_C_test.TEST.VISUALIZE_INDEX = 0
_C_test.TEST.VISUALIZE_METRIC = ""
_C_test.TEST.HEAD_FUSION = 'max'
_C_test.TEST.DISCARD_RATIO = 0.9
_C_test.TEST.EXPLAIN = False
_C_test.TEST.EVAL_METRIC = ""
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C_test.OUTPUT_DIR = ""

# Model INDEX
_C_test.INDEX = None

# Weight and Biase debug
_C_test.WANDB = False