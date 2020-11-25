# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
from _utils.pathfinder import get_repo_path
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from data_processors import UnsexProcessor,ColaProcessor,MnliProcessor,MnliMismatchedProcessor,MrpcProcessor,Sst2Processor,StsbProcessor,QqpProcessor,QnliProcessor,RteProcessor,WnliProcessor
from train_utils import compute_metrics, convert_examples_to_features, id2onehot

import shap

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def compute_shap_for_fastbert(data_dir, bert_model, output_dir, task_name='unsex', do_eval=True, max_seq_length=300, eval_batch_size=1,
         do_train=False, do_lower_case=False, learning_rate=5e-5, train_batch_size=32, seed=42, no_cuda=False,
         local_rank=-1, warmup_proportion=0.1):
    # =================model defination===================
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor,
        'unsex': UnsexProcessor
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification",
        'unsex': 'classification'
    }

    if local_rank == -1 or no_cuda:
        if torch.cuda.is_available() and not no_cuda:
           device = torch.device("cuda")
        elif no_cuda or torch.cuda.is_available():
            device = torch.device("cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(local_rank != -1)))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if not do_train and not do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(output_dir) and os.listdir(output_dir) and do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task_name = task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    train_examples = None
    num_train_optimization_steps = None

    train_examples = processor.get_train_examples(data_dir)
        
    # Prepare model
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(local_rank))
    model = BertForSequenceClassification.from_pretrained(bert_model,
              cache_dir=cache_dir,
              num_labels=num_labels)
    
    model.to(device)
    if local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=num_labels)
    model.to(device)

    #==================End of Model Def=================

    #==================Shap value=================
    # train data preparation
    print(len(train_examples))
    train_features = convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer, output_mode, logger)
   
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

   
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
   
    #==================Data structure=================
    #     [[token ids],[mask], [segs], [label]]      |
    #=================================================
    # train_one_hot = id2onehot(all_input_ids, model.config.vocab_size)
    print("training data in total: ", all_input_ids.size()[0])

    # add randomness of selection
    N_train = len(train_features)
    rand_idx_list = np.random.choice(N_train, size=(100, ), replace=False)
    explainer = shap.DeepBertExplainer(model, all_input_mask[rand_idx_list], all_segment_ids[rand_idx_list], all_input_ids[rand_idx_list], device, model.config.vocab_size, id2onehot)

    print("explainer init finished")
    
    eval_examples = processor.get_test_examples(data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer, output_mode, logger)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)


    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    
    print("test_data in total: ", all_input_ids.size()[0])
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    shap_values = []
    print("-"*20, "SHAP eval begin", "-"*20)
    # data_save_folder = data_dir.split("/")[-1]
    total_inst_num = all_input_ids.size()[0]
    
    for i in range(total_inst_num):
        eval_one_hot = id2onehot(all_input_ids[i:i+1], model.config.vocab_size)
        shap_value = explainer.shap_values(X=eval_one_hot, eval_mask=all_input_mask[i:i+1], seg=all_segment_ids[i:i+1], tk_idx=all_input_ids[i:i+1], ranked_outputs=None)
        values = []
        for lb in range(num_labels):
            tks = all_input_ids[i:i+1]
            seq_len = tks.size()[1]
            right_value = shap_value[lb][0,torch.arange(0, seq_len).long(), tks[0, :]]

            values.append(right_value)
        shap_values.append(values)
        # if i % 5 == 0 and i != 0:
        #     with open('data/SHAP_features/'+data_save_folder+'-'+str(i)+'.npz', 'wb') as f:
        #         np.save(f, shap_values)
        #     shap_values = []
    with open(output_dir + '/fast-bert-shap.npy', 'wb') as f:
        np.save(f, shap_values)


if __name__ == '__main__':
    compute_shap_for_fastbert(
        os.path.join(get_repo_path(), '_data', 'as_csv'),
        os.path.join(get_repo_path(), '_trained_models', 'fast-bert'),
        os.path.join(get_repo_path(), '_explanations'),
        no_cuda=True
    )
