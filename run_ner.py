import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from torch import nn
from typing import Dict, List, Optional, Tuple
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed, BertConfig, BertTokenizer,
)

from trainer_ner import Trainer

from model import BertForNER

from utils import OpenUEDataset, Split, get_labels_ner, get_labels_seq, openue_data_collator_ner, precision_score, \
    recall_score, f1_score

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
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
    task: Optional[str] = field(
        default='seq'
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

def align_predictions(label_map_ner, batch_predictions, batch_label_ids) -> Tuple[List[int], List[int]]:

    predict = []
    label = []
    for i, batch in enumerate(batch_predictions):
        predictions = np.array(batch_predictions[i])
        label_ids = np.array(batch_label_ids[i])
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != -100:  # 特殊标签，还原句子的原有长度
                    out_label_list[i].append(label_map_ner[label_ids[i][j]])
                    preds_list[i].append(label_map_ner[preds[i][j]])

        predict.extend(preds_list)
        label.extend(out_label_list)

    return predict, label

def compute_metrics(predictions, label_ids, label_map_ner) -> Dict:
    preds_list, out_label_list = align_predictions(label_map_ner, predictions, label_ids)
    return {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # 读取ner的label
    # ["O", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ"]
    labels_ner = get_labels_ner()
    label_map_ner: Dict[int, str] = {i: label for i, label in enumerate(labels_ner)}
    num_labels_ner = len(labels_ner)

    # 读取seq的label
    labels_seq = get_labels_seq()

    # 读取待训练的NER模型
    config = BertConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels_ner,
        id2label=label_map_ner,
        label2id={label: i for i, label in enumerate(labels_ner)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model_ner = BertForNER.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        # label_map_seq=label_map_seq,
        # label_map_ner=label_map_ner
    )

    # Get datasets
    train_dataset = (
        OpenUEDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels_seq=labels_seq,
            labels_ner=labels_ner,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
            task=data_args.task
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        OpenUEDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels_seq=labels_seq,
            labels_ner=labels_ner,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
            task=data_args.task
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        OpenUEDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels_seq=labels_seq,
            labels_ner=labels_ner,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
            task=data_args.task
        )
        if training_args.do_predict
        else None
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model_ner,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=openue_data_collator_ner,
        compute_metrics=compute_metrics,
        label_map_ner=label_map_ner
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
