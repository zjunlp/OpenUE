""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union, Dict

import numpy as np

import jsonlines

from transformers import PreTrainedTokenizer, is_torch_available, BatchEncoding

logger = logging.getLogger(__name__)


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def get_entities(seq, suffix=False):
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def f1_score(y_true, y_pred, average='micro', suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def accuracy_score(y_true, y_pred):
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t==y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def precision_score(y_true, y_pred, average='micro', suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true, y_pred, average='micro', suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


@dataclass
class InputExample:
    text_id: str
    words: str
    triples: List

@dataclass
class OutputExample:
    h: str
    r: str
    t: str

@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids_seq: Optional[List[int]] = None
    label_ids_ner: Optional[List[int]] = None
    words: str = None

@dataclass
class InputFeatures_Interactive:
    input_ids: List[int] = None
    attention_mask: List[int] = None
    token_type_ids: List[int] = None
    triples: List[List[int]] = None



class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset

    class OpenUEDataset(Dataset):

        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

        def __init__(
            self,
            data_dir: str,
            labels_seq: List,
            labels_ner: List,
            tokenizer: PreTrainedTokenizer,
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            task='seq'

        ):
            # Load data features from cache or dataset file
            cached_examples_file = os.path.join(
                data_dir, "cached_{}_{}.examples".format(mode.value, tokenizer.__class__.__name__),
            )

            if task == 'seq':
                cached_features_file = os.path.join(
                    data_dir, "cached_{}_{}_seq".format(mode.value, tokenizer.__class__.__name__),
                )
            elif task == 'ner':
                cached_features_file = os.path.join(
                    data_dir, "cached_{}_{}_ner".format(mode.value, tokenizer.__class__.__name__),
                )
            elif task == 'interactive':
                cached_features_file = os.path.join(
                    data_dir, "cached_{}_{}_interactive".format(mode.value, tokenizer.__class__.__name__),
                )

            # features是否存在
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                # examples是否存在
                if os.path.exists(cached_examples_file) and not overwrite_cache:
                    logger.info(f"Loading example from dataset file at {data_dir}")
                    examples = torch.load(cached_examples_file)
                else:
                    logger.info(f"Creating example from cached file {cached_examples_file}")
                    examples = read_examples_from_file(data_dir, mode)
                    torch.save(examples, cached_examples_file)

                logger.info(f"Creating features from dataset file at {data_dir}")
                if task == 'seq':
                    self.features = convert_examples_to_seq_features(
                        examples,
                        # labels,
                        labels_seq=labels_seq,
                        labels_ner=labels_ner,
                        max_seq_length=max_seq_length,
                        tokenizer=tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                    )
                elif task == 'ner':
                    self.features = convert_examples_to_ner_features(
                        examples,
                        labels_seq=labels_seq,
                        labels_ner=labels_ner,
                        max_seq_length=max_seq_length,
                        tokenizer=tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                    )
                elif task == 'interactive':
                    self.features = convert_examples_to_interactive_features(
                        examples,
                        labels_seq=labels_seq,
                        labels_ner=labels_ner,
                        max_seq_length=max_seq_length,
                        tokenizer=tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                    )

                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

import json
def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.json")

    examples = []

    with open(file_path, "r+", encoding="utf8") as f:

        text_id = 0
        
        for line in f.readlines():
            item = json.loads(line)
            text = item['text']
            triples = []
            for triple in item['spo_list']:
                triples.append([triple['subject'], triple['predicate'], triple['object']])

            examples.append(InputExample(text_id=text_id, words=text, triples=triples))
            text_id = text_id + 1

    return examples


def convert_examples_to_seq_features(
    examples: List[InputExample],
    labels_seq: List[str],
    labels_ner: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    features = []
    label2id = {label: i for i, label in enumerate(labels_seq)}

    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer(
            example.words,
            add_special_tokens=True,
            # return_overflowing_tokens=True,
            truncation="longest_first",
            max_length=max_seq_length,
        )
        label_ids_seq = []
        for triple in example.triples:
            # input_ids: List[int]
            # attention_mask: List[int]
            # token_type_ids: Optional[List[int]] = None
            # label_ids_seq: Optional[List[int]] = None
            # label_ids_ner: Optional[List[int]] = None
            label_ids_seq.append(label2id[triple[1]])

        features.append(InputFeatures(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                          token_type_ids=inputs['token_type_ids'], label_ids_seq=label_ids_seq))

    return features

def convert_examples_to_ner_features(
    examples: List[InputExample],
    labels_seq: List[str],
    labels_ner: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    label_map_seq = {label: i for i, label in enumerate(labels_seq)}
    label_map_ner = {label: i for i, label in enumerate(labels_ner)}

    features = []
    counter = 0

    def find_word_in_texts(word_ids, texts_ids):
        length = len(word_ids)
        for i, W in enumerate(texts_ids):
            if texts_ids[i] == word_ids[0]:
                if texts_ids[i: i + length] == word_ids:
                    return i, i + length
        return None, None

    for (ex_index, example) in enumerate(examples):
        # 用bert分词，转换为token
        # text = example.text
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        text = example.words
        for triple in example.triples:
            subject = triple[0]
            relation = triple[1]
            object_ = triple[2]

            # cls w1 w2 .. sep w3 w4 sep 000000000
            # token_type
            # 000000000000000 1111111111
            # 转换为id，加上cls以及seq等
            # {"input_ids":[], "token_type_ids":[], "attention_mask":[]}
            inputs = tokenizer(
                text,
                add_special_tokens=True,
                max_length=max_seq_length-2,
                truncation="longest_first"
            )

            inputs['token_type_ids'] = tokenizer.create_token_type_ids_from_sequences(inputs['input_ids'][1:-1],
                                                                                       [label_map_seq[relation]])
            # label_map_seq[relation] 加入关系信息
            inputs['input_ids'] = inputs['input_ids'] + [label_map_seq[relation], tokenizer.sep_token_id]
            inputs['attention_mask'] = inputs['attention_mask'] + [1, 1]

            # 添加split_text文本的标签
            # B-SUB I-SUB / B-OBJ I-OBJ
            split_text_ids = inputs['input_ids']
            # ["O", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "Relation"]
            # 默认所有位置都为'O'
            lable_ner = ['O' for i in range(len(split_text_ids))]

            # 标注subject
            subject_ids = tokenizer.encode(subject, add_special_tokens=False)
            [start_idx, end_idx] = find_word_in_texts(subject_ids, split_text_ids)
            if start_idx is None:
                # logger.info('语料有问题(subject)！%d', ex_index)
                counter = counter + 1
                continue
            lable_ner[start_idx: end_idx] = ['I-SUB' for i in range(len(subject_ids))]
            lable_ner[start_idx] = 'B-SUB'

            # 标注object
            object_ids = tokenizer.encode(object_, add_special_tokens=False)
            [start_idx, end_idx] = find_word_in_texts(object_ids, split_text_ids)
            if start_idx is None:
                # logger.info('语料有问题(object)！%d', ex_index)
                counter = counter + 1
                continue
            lable_ner[start_idx: end_idx] = ['I-OBJ' for i in range(len(object_ids))]
            lable_ner[start_idx] = 'B-OBJ'

            # 标注最后三个字符串，SEP、Relation、SEP
            lable_ner[-1] = 'O'
            # lable_ner[-2] = 'Relation'
            lable_ner[-2] = 'O'
            lable_ner[-3] = 'O'

            assert len(lable_ner) == len(inputs['input_ids']) == len(inputs['token_type_ids']) ==\
                        len(inputs['attention_mask'])

            # 关系抽取标签
            label_id_seq = label_map_seq[relation]

            # NER标签转换
            label_id_ner = [label_map_ner[i] for i in lable_ner]

            features.append(
                InputFeatures(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                    label_ids_ner=label_id_ner,
                    label_ids_seq=label_id_seq,
                    words=text
                )
            )

    # print('语料有问题句子比例是', str(counter/len(examples)))
    return features



def convert_examples_to_interactive_features(
    examples: List[InputExample],
    labels_seq: List[str],
    labels_ner: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    rel2id: Dict =None
):
    label_map_seq = {label: i for i, label in enumerate(labels_seq)}
    label_map_ner = {label: i for i, label in enumerate(labels_ner)}

    features = []
    counter = 0

    def find_word_in_texts(input_ids, entity_ids):
        length = len(entity_ids)
        for i, W in enumerate(input_ids):
            if i+length <= len(input_ids) and input_ids[i: i + length] == entity_ids:
                return i, i + length
        return None, None
    

    for (ex_index, example) in enumerate(examples):
        # 用bert分词，转换为token
        # text = example.text
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        text = example.words
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_seq_length-2,
            truncation="longest_first"
        )
        
        
        triples = []
        

        for triple in example.triples:
            h, r, t = triple
            t_ids = tokenizer(" "+h, add_special_tokens=False)
            h_s, h_e = find_word_in_texts(inputs['input_ids'], t_ids)
            t_ids = tokenizer(" "+t, add_special_tokens=False)
            t_s, t_e = find_word_in_texts(inputs['input_ids'], t_ids)
            r = rel2id[r]
            triples.append([h_s,h_e,t_s,t_e,r])
        

        features.append(
            InputFeatures(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                triples=triples
            )
        )
    logger.info(examples[0])
    logger.info(features[0])

    return features



def get_labels_ner() -> List[str]:
    return ["O", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "Relation"]

def get_labels_seq() -> List[str]:
    class_label = ['Empty', '丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地', '出生日期','创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑', '改编自', '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高', '连载网站','邮政编码', '面积', '首都']
    return class_label

def openue_data_collator_seq(features):
    max_length = [len(f.input_ids) for f in features]
    max_length = max(max_length)

    features_new = []
    length_seq_label = len(get_labels_seq())

    for f in features:
        length = len(f.input_ids)
        distance = max_length - length
        add_zero = [0 for i in range(distance)]

        features_ = {}

        features_['input_ids'] = f.input_ids + add_zero  # 补0
        features_['attention_mask'] = f.attention_mask + add_zero  # 补0
        features_['token_type_ids'] = f.token_type_ids + add_zero  # 补0

        label_id_seq = np.zeros(length_seq_label).tolist()
        for relation_id in f.label_ids_seq:
            if isinstance(relation_id, int):
                label_id_seq[relation_id] = 1
            else:
                print('error')
                print(relation_id)
                print(label_id_seq)
        features_['label_ids_seq'] = label_id_seq
        features_new.append(features_)

    # 将结构体格式变成dict格式
    if not isinstance(features_new[0], (dict, BatchEncoding)):
        features_new = [vars(f) for f in features_new]

    first = features_new[0]
    batch = {}

    batch["label_ids_seq"] = torch.tensor([f["label_ids_seq"] for f in features_new], dtype=torch.float)

    for k, v in first.items():
        if k not in ("label_ids_seq") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features_new])
            else:
                batch[k] = torch.tensor([f[k] for f in features_new], dtype=torch.long)

    return batch

def openue_data_collator_ner(features):
    # 读取ner的label
    max_length = [len(f.input_ids) for f in features]
    max_length = max(max_length)

    features_new = []
    for f in features:
        length = len(f.input_ids)
        distance = max_length - length
        add_zero = [0 for i in range(distance)]
        add_special = [0 for i in range(distance)] 

        features_ = {}

        features_['input_ids'] = f.input_ids + add_zero  # 补0
        features_['attention_mask'] = f.attention_mask + add_zero  # 补0
        features_['token_type_ids'] = f.token_type_ids + add_zero  # 补0
        features_['label_ids_ner'] = f.label_ids_ner + add_special  # 补0, 这里仅仅为了补齐的最长长度, loss计算中有mask存在会被忽略

        features_new.append(features_)

    if not isinstance(features_new[0], (dict, BatchEncoding)):
        features_new = [vars(f) for f in features_new]

    first = features_new[0]
    batch = {}

    label = first["label_ids_ner"].item() if isinstance(first["label_ids_ner"], torch.Tensor) else first["label_ids_ner"]
    dtype = torch.long if isinstance(label, int) else torch.long
    batch["label_ids_ner"] = torch.tensor([f["label_ids_ner"] for f in features_new], dtype=dtype)

    for k, v in first.items():
        if k not in ("label_ids_seq", "label_ids_ner") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features_new])
            else:
                batch[k] = torch.tensor([f[k] for f in features_new], dtype=torch.long)

    return batch

def openue_data_collator_interactive(features):
    max_length_seq = [len(f.input_ids_seq) for f in features]
    max_length_seq = max(max_length_seq)

    max_length_ner = [len(f.input_ids_ner) for f in features]
    max_length_ner = max(max_length_ner)

    features_new = []
    for f in features:
        length = len(f.input_ids_seq)
        distance = max_length_seq - length
        add_zero = [0 for i in range(distance)]

        features_ = {}

        features_['input_ids_seq'] = f.input_ids_seq + add_zero  # 补0
        features_['attention_mask_seq'] = f.attention_mask_seq + add_zero  # 补0
        features_['token_type_ids_seq'] = f.token_type_ids_seq + add_zero  # 补0

        length = len(f.input_ids_ner)
        distance = max_length_ner - length
        add_zero = [0 for i in range(distance)]
        add_special = [0 for i in range(distance)]
        features_['input_ids_ner'] = f.input_ids_ner + add_zero  # 补0
        features_['attention_mask_ner'] = f.attention_mask_ner + add_zero  # 补0
        features_['token_type_ids_ner'] = f.token_type_ids_ner + add_zero  # 补0
        features_['label_ids_ner'] = f.label_ids_ner + add_special  # 补0, 为了补齐的最长长度, loss计算中有mask存在会被忽略
        features_['label_ids'] = f.label_ids

        features_new.append(features_)

    # 将结构体格式变成dict格式
    if not isinstance(features_new[0], (dict, BatchEncoding)):
        features_new = [vars(f) for f in features_new]

    first = features_new[0]
    batch = {}
    # batch["label_ids_seq"] = torch.tensor([f["label_ids_seq"] for f in features], dtype=torch.float)
    batch["label_ids_ner"] = torch.tensor([f["label_ids_ner"] for f in features_new], dtype=torch.float)
    batch["label_ids"] = [f["label_ids"] for f in features_new]

    for k, v in first.items():
        if k not in ("label_ids", "label_ids_ner") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features_new])
            else:
                batch[k] = torch.tensor([f[k] for f in features_new], dtype=torch.long)

    return batch