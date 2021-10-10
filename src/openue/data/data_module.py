import os
from .base_data_module import BaseDataModule
from .processor import get_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import logging

logger = logging.getLogger(__name__)
from .utils import get_labels_ner, get_labels_seq, openue_data_collator_seq, openue_data_collator_ner, openue_data_collator_interactive


collator_set = {"ner": openue_data_collator_ner, "seq": openue_data_collator_seq, "interactive": openue_data_collator_interactive}

class REDataset(BaseDataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.num_labels = len(get_labels_ner()) if args.task_name == "ner" else len(get_labels_seq(args))
        self.collate_fn = collator_set[args.task_name]
        
        num_relations = len(get_labels_seq(args))

        # 默认加入特殊token来表示关系
        add_flag = False
        for i in range(num_relations):
            if f"[relation{i}]" not in self.tokenizer.get_added_vocab():
                add_flag = True
                break
        
        if add_flag:
            relation_tokens = [f"[relation{i}]" for i in range(num_relations)]
            num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': relation_tokens})
            logger.info(f"add total special tokens: {num_added_tokens} \n {relation_tokens}")

    def setup(self, stage=None):
        self.data_train = get_dataset("train", self.args, self.tokenizer)
        self.data_val = get_dataset("dev", self.args, self.tokenizer)
        self.data_test = get_dataset("test", self.args, self.tokenizer)

    def prepare_data(self):
        # download the dataset and move it to the dataset fold
        if not os.path.exists(self.args.data_dir):
            os.system("mkdir ske")
            os.system("wget  http://47.92.96.190/dataset/ske.tar.gz")
            os.system("tar zxvf ske.tar.gz -C ske")
            os.system("mv ske ./dataset")
            os.system("rm ske.tar.gz")
            os.system("mv ./dataset/ske/val.json ./dataset/ske/dev.json")

        pass

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--task_name", type=str, default="ner",choices=["ner", "seq", "interactive"], help="[normal, reloss, ptune]",)
        parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased", help="Number of examples to operate on per forward step.")
        parser.add_argument("--max_seq_length", type=int, default=128, help="Number of examples to operate on per forward step.")
        return parser

    def get_config(self):
        return dict(num_tokens=len(self.tokenizer), num_labels=self.num_labels, tokenizer=self.tokenizer)