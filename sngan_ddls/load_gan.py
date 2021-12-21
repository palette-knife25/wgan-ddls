#src/load_gan.py

from os.path import dirname, abspath, exists, join
import glob
import json
import os
import random
import warnings

from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist

from data_util import Dataset_
from utils.style_ops import grid_sample_gradfix
from utils.style_ops import conv2d_gradfix
from metrics.inception_net import InceptionV3
from sync_batchnorm.batchnorm import convert_model
from worker import WORKER
import utils.log as log
import utils.losses as losses
import utils.ckpt as ckpt
import utils.misc as misc
import utils.custom_ops as custom_ops
import models.model as model
import metrics.preparation as pp


def load_gan(local_rank, cfgs):
    
    if local_rank == 0:
        logger = None
    else:
        logger = None
        
        
    Gen, Gen_mapping, Gen_synthesis, Dis, Gen_ema, Gen_ema_mapping, Gen_ema_synthesis, ema =\
        model.load_generator_discriminator(DATA=cfgs.DATA,
                                           OPTIMIZATION=cfgs.OPTIMIZATION,
                                           MODEL=cfgs.MODEL,
                                           STYLEGAN2=cfgs.STYLEGAN2,
                                           MODULES=cfgs.MODULES,
                                           RUN=cfgs.RUN,
                                           device=local_rank,
                                           logger=logger)
    
    return Gen, Dis