import csv
import pickle 
import os
import logging
import argparse
import random
from torch.functional import Tensor
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import numpy as np
import torch
import inspect
from collections import OrderedDict

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
# 这就是包内引用吗
import json
import re

from .utils import OpenUEDataset, get_labels_ner, get_labels_seq, Split


def get_dataset(mode, args, tokenizer):
    dataset = OpenUEDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        labels_seq=get_labels_seq(args),
        labels_ner=get_labels_ner(),
        model_type=args.model_type,
        max_seq_length=args.max_seq_length,
        overwrite_cache=args.overwrite_cache,
        mode=Split(mode), # train, dev, test
        task=args.task_name # seq ner interactive
    )
    
    
    return dataset

