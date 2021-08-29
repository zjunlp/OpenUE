from utils import Split, get_labels_seq
from utils import OpenUEDataset, openue_data_collator_seq
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from typing import Optional, Dict, Union, Any
from distill_model import TinyBertForRelationClassification, BertForRelationClassification
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
import torch


import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from transformers import BertForTokenClassification, BertConfig, AdamW,BertTokenizer
from transformers import get_linear_schedule_with_warmup
import torch

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_utils import PreTrainedModel
import argparse
import tensorboard

import os
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BertModel,
)
from transformers.data.data_collator import default_data_collator

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    task: Optional[str] = field(
        default='seq'
    )


def _prepare_dataloader(dataset, training_args):
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(
            dataset,
            batch_size=training_args.train_batch_size,
            sampler=sampler,
            collate_fn=openue_data_collator_seq,
            drop_last=training_args.dataloader_drop_last,
        )
    return dataloader

def main():
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    labels_seq = get_labels_seq()
    label_map_seq = {i: label for i, label in enumerate(labels_seq)}
    num_labels = len(labels_seq)

    teacher_config = BertConfig.from_pretrained(
            model_args.model_path,
            num_labels=num_labels,
            id2label=label_map_seq,
            label2id={label: i for i, label in enumerate(labels_seq)},
    )
    student_config = AutoConfig.from_pretrained(
        "hfl/rbt3",
        num_labels=num_labels,
        id2label=label_map_seq,
        label2id={label: i for i, label in enumerate(labels_seq)},
    )
    tokenizer = BertTokenizer.from_pretrained(teacher_config._name_or_path, use_fast=False,)
    teacher_model = BertForRelationClassification.from_pretrained(model_args.model_path,config=teacher_config,).to(device)
    student_model = TinyBertForRelationClassification.from_pretrained("hfl/rbt3",config=student_config,).to(device)

    train_dataset = (
        OpenUEDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels_seq=labels_seq,
            labels_ner=None,
            model_type=teacher_config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
    )

    train_dataloader = _prepare_dataloader(train_dataset, training_args)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    scheduler_class = get_linear_schedule_with_warmup
    scheduler_args = {'num_warmup_steps':training_args.warmup_steps, 'num_training_steps': int(len(train_dataloader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs)}



    def simple_adaptor(batch, model_outputs):
        return {"logits":model_outputs.logits, 'hidden': model_outputs.hidden}


    distill_config = DistillationConfig(
    intermediate_matches=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":4, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":8, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":12,"layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1}])

    train_config = TrainingConfig(output_dir=training_args.output_dir, log_dir=training_args.logging_dir)
    distiller = GeneralDistiller(
        train_config=train_config, distill_config=distill_config,
        model_T=teacher_model, model_S=student_model, 
        adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)

    with distiller:
        distiller.train(optimizer, train_dataloader, 20, scheduler_class=scheduler_class, scheduler_args = scheduler_args, callback=None)


if __name__ == '__main__':

    main()