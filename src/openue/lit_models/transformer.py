from logging import debug
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# Hide lines below until Lab 5
import wandb
import numpy as np

from openue.models.model import Inference
# Hide lines above until Lab 5

from .base import BaseLitModel
from .metric import f1_eval, compute_f1, acc, compute_metrics, seq_metric
from transformers.optimization import (
    get_linear_schedule_with_warmup,
)
from transformers import AutoConfig, AutoTokenizer
from functools import partial
from openue.models import BertForRelationClassification, BertForNER
from openue.data import get_labels_ner

from functools import partial

class RELitModel(BaseLitModel):
    def __init__(self, args, data_config):
        super().__init__(args, data_config)
        self.loss_fn = nn.BCEWithLogitsLoss()
        label_map_ner = get_labels_ner()
        self.eval_fn = partial(compute_metrics,label_map_ner=label_map_ner)
        self.best_f1 = 0
        
        self._init_model()

    def forward(self, x):
        return self.model(x)

    def _init_model(self):
        #TODO put the parameters from the data_config to the config, maybe use the __dict__?
        config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        config.num_labels = self.data_config['num_labels']
        self.model = BertForNER.from_pretrained(self.args.model_name_or_path, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        loss, logits = self.model(**batch)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        loss, logits = self.model(**batch)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": batch['label_ids_ner'].detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        # due to the batch operation, different batch has different sequence length. add -100 to pad.
        logits = np.concatenate([self.padding(o["eval_logits"]) for o in outputs], axis=0)
        labels = np.concatenate([self.padding(o["eval_labels"]) for o in outputs], axis=0)

        result = self.eval_fn(logits, labels)
        f1 = result['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        logits = self.model(input_ids, attention_mask, token_type_ids)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([self.padding(o["test_logits"]) for o in outputs], axis=0)
        labels = np.concatenate([self.padding(o["test_labels"]) for o in outputs], axis=0)

        result = self.eval_fn(logits, labels)
        f1 = result['f1']
        self.log("Test/f1", f1)

    def padding(self,a):
        if len(a.shape)==3:
            # logits
            bsz, seq, num_ner_labels = a.shape
            t = np.full((bsz, self.args.max_seq_length, num_ner_labels), -100)
            t[:,:seq,:] = a[:,:,:]
        else:
            # labels
            bsz, seq = a.shape
            t = np.full((bsz, self.args.max_seq_length), -100)
            t[:,:seq] = a[:,:]
        return t




class SEQLitModel(BaseLitModel):
    def __init__(self, args, data_config):
        super().__init__(args, data_config)
        self.loss_fn = nn.BCEWithLogitsLoss()
        label_map_ner = get_labels_ner()
        self.eval_fn = seq_metric
        self.best_f1 = 0
        
        self._init_model()

    def forward(self, x):
        return self.model(x)

    def _init_model(self):
        #TODO put the parameters from the data_config to the config, maybe use the __dict__?
        config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        config.num_labels = self.data_config['num_labels']
        self.model = BertForRelationClassification.from_pretrained(self.args.model_name_or_path, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        loss, _, _, _ = self.model(**batch)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        loss, _, logits, _ = self.model(**batch)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": batch['label_ids_seq'].detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        # due to the batch operation, different batch has different sequence length. add -100 to pad.
        logits = np.concatenate([o["eval_logits"] for o in outputs], axis=0)
        labels = np.concatenate([o["eval_labels"] for o in outputs], axis=0)

        result = self.eval_fn(logits, labels)
        f1 = result['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        loss, _, logits, _ = self.model(**batch)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": batch['label_ids_seq'].detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs], axis=0)
        labels = np.concatenate([o["test_labels"] for o in outputs], axis=0)

        result = self.eval_fn(logits, labels)
        f1 = result['f1']
        self.log("Test/f1", f1)



class INFERLitModel(BaseLitModel):
    def __init__(self, args, data_config):
        super().__init__(args, data_config)

        self._init_model()

    def forward(self, x):
        return self.model(x)

    def _init_model(self):
        #TODO put the parameters from the data_config to the config, maybe use the __dict__?
        self.model = Inference(self.args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        loss, _, logits, _ = self.model(**batch)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": batch['label_ids_seq'].detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs], axis=0)
        labels = np.concatenate([o["test_labels"] for o in outputs], axis=0)

        result = self.eval_fn(logits, labels)
        f1 = result['f1']
        self.log("Test/f1", f1)
