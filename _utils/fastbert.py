from fast_bert import BertLearner
from fast_bert import BertDataBunch
import os
from fast_bert import accuracy
import logging
from _utils.pathfinder import get_experiment_path, get_repo_path
import torch


def get_fastbert_model(experiment):
    data_path = os.path.join(get_repo_path(), '_data', 'as_csv', experiment.name)

    databunch = BertDataBunch(data_path, data_path,
                              tokenizer='bert-base-uncased',
                              train_file='train.csv',
                              val_file='val.csv',
                              label_file='labels.csv',
                              text_col='text',
                              label_col='label',
                              batch_size_per_gpu=8,
                              max_seq_length=512,
                              multi_gpu=True,
                              multi_label=False,
                              model_type='bert')

    fastbert = BertLearner.from_pretrained_model(databunch,
                                                 pretrained_path=os.path.join(get_experiment_path(experiment),
                                                                              'models', 'fastbert'),
                                                 metrics=[{'name': 'accuracy', 'function': accuracy}],
                                                 device=torch.device("cuda"),
                                                 logger=logging.getLogger(), output_dir='output')
    return fastbert
