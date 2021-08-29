import logging
import os
import sys
from dataclasses import dataclass, field
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm, trange
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
import re
import jsonlines
import difflib

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import (
    HfArgumentParser,
    set_seed, BertConfig, BertTokenizer, TrainingArguments, PreTrainedTokenizer,
)

from model import BertForNER, BertForRelationClassification

from utils import Split, get_labels_ner, get_labels_seq, OpenUEDataset, openue_data_collator_interactive, OutputExample

logger = logging.getLogger(__name__)



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    seq_model_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    ner_model_path: str = field(
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

def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

def process(text, result):
    index = 0
    start = None
    labels = {}
    labels['subject'] = []
    labels['object'] = []
    indicator = ''
    for w, t in zip(text, result):
        # ["O", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "Relation"
        if start is None:
            if t == 'B-SUB':
                start = index
                indicator = 'subject'
            elif t == 'B-OBJ':
                start = index
                indicator = 'object'
        else:
            # if t == 'I-SUB' or t == 'I-OBJ':
            #     continue
            if t == "O":
                # print(result[start: index])
                labels[indicator].append(text[start: index])
                start = None
        index += 1
    # print(labels)
    return labels

def predict(model_seq, model_ner, inputs, training_args, label_map_ner, label_map_seq, tokenizer, texts=None):

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(training_args.device)

    inputs_seq = {'input_ids': inputs['input_ids_seq'],
                  'token_type_ids': inputs['token_type_ids_seq'],
                  'attention_mask': inputs['attention_mask_seq'],
                  # 'label_ids_seq': inputs['label_ids_seq']
                  }

    with torch.no_grad():
        outputs_seq = model_seq(**inputs_seq)

        batch_size = inputs_seq['input_ids'].shape[0]
        num_relations = len(label_map_seq.keys())
        max_length = inputs_seq['input_ids'].shape[1]

        # [batch_size, 50]
        # relation_output_sigmoid = outputs_seq[1]
        relation_output_sigmoid = outputs_seq[0]
        if True:
            # 多关系预测
            relation_output_sigmoid_ = relation_output_sigmoid > 0.5
            # # 这个0.5是超参数，超参数
            if torch.sum(relation_output_sigmoid_).tolist() == 0:
                idx = torch.max(relation_output_sigmoid, dim=1)[1].tolist()[0]
                relation_output_sigmoid_[0][idx] = 1

            # [batch_size, 50]
            relation_output_sigmoid_ = relation_output_sigmoid_.long()
            # [batch_size * 50, ]
            relation_output_sigmoid_index = relation_output_sigmoid_.view(-1)
        else:
            # 单关系预测
            relation_output_sigmoid_ = torch.max(relation_output_sigmoid, dim=1)[1]
            tmp1 = torch.zeros(batch_size, num_relations, device=training_args.device)
            relation_output_sigmoid_index = tmp1.scatter_(dim=1, index=relation_output_sigmoid_.view(-1, 1).long(), value=1)
            # [batch_size * 50, ]
            relation_output_sigmoid_index = relation_output_sigmoid_index.view(-1).long()

        index_ = torch.arange(0, num_relations).to(training_args.device)
        index_ = index_.expand(batch_size, num_relations)
        # 需要拼接的部分1：REL
        relation_output_sigmoid_number = torch.masked_select(index_, relation_output_sigmoid_.bool())
        # 需要拼接的部分2：SEP
        cat_sep = torch.full((relation_output_sigmoid_number.shape[0], 1), 102).long().to(training_args.device)
        # 需要拼接的部分3：[1]
        cat_one = torch.full((relation_output_sigmoid_number.shape[0], 1), 1).long().to(training_args.device)
        # 需要拼接的部4：[0]
        cat_zero = torch.full((relation_output_sigmoid_number.shape[0], 1), 0).long().to(training_args.device)

        # 拼接input_ids_seq的输入
        input_ids_ner = torch.unsqueeze(inputs['input_ids_seq'], 1)
        # [batch_size, 50, max_length], 复制50份
        input_ids_ner = input_ids_ner.expand(-1, len(label_map_seq.keys()), -1)
        # [batch_size * 50, max_length]
        input_ids_ner_reshape = input_ids_ner.reshape(batch_size * num_relations, max_length)
        # 选择预测正确的所有关系
        tmp1 = relation_output_sigmoid_index.unsqueeze(dim=1)  # [200, 1]
        mask = tmp1.expand(-1, max_length)  # [200, 79]
        tmp2 = torch.masked_select(input_ids_ner_reshape, mask.bool())
        # n(选出来的关系数字) * max_length
        # n >> batch_size, 因为一句话中有多个关系
        tmp3 = tmp2.view(-1, max_length)
        # 拼接 0
        tmp4 = torch.cat((tmp3, cat_zero), 1)
        # 拼接 0
        input_ids_ner = torch.cat((tmp4, cat_zero), 1)

        # 利用attention中1的求和的到rel_pos的位置
        attention_mask_ner = torch.unsqueeze(inputs['attention_mask_seq'], 1)
        # [batch_size, 50, max_length], 复制50份
        attention_mask_ner = attention_mask_ner.expand(-1, len(label_map_seq.keys()), -1)
        # [batch_size * 50, max_length]
        attention_mask_ner_reshape = attention_mask_ner.reshape(batch_size * num_relations, max_length)
        # 选择预测正确的所有关系
        tmp1 = relation_output_sigmoid_index.unsqueeze(dim=1)  # [200, 1]
        mask = tmp1.expand(-1, max_length)  # [200, 79]
        tmp2 = torch.masked_select(attention_mask_ner_reshape, mask.bool())
        # n(选出来的关系数字) * max_length
        # n >> batch_size, 因为一句话中有多个关系
        tmp3 = tmp2.view(-1, max_length)
        # 利用attention中1的求和的到rel_pos的位置
        rel_pos = torch.sum(tmp3, dim=1)
        (rel_number_find, max_length_find) = input_ids_ner.shape
        one_hot = torch.sparse.torch.eye(max_length_find).long().to(training_args.device)
        rel_pos_mask = one_hot.index_select(0, rel_pos)
        rel_pos_mask_plus = one_hot.index_select(0, rel_pos+1)

        # 拼接input_ids的输入
        input_ids_ner[rel_pos_mask.bool()] = relation_output_sigmoid_number
        input_ids_ner[rel_pos_mask_plus.bool()] = cat_sep.squeeze()

        # 拼接token_type_ids的输入
        token_type_ids_ner = torch.zeros(rel_number_find, max_length_find).to(training_args.device)
        token_type_ids_ner[rel_pos_mask.bool()] = 1
        token_type_ids_ner[rel_pos_mask_plus.bool()] = 1
        token_type_ids_ner = token_type_ids_ner.long()

        # 拼接attention_mask的输入
        # 拼接 0
        tmp4 = torch.cat((tmp3, cat_zero), dim=1)
        # 拼接 0
        tmp5 = torch.cat((tmp4, cat_zero), dim=1)
        tmp5[rel_pos_mask.bool()] = 1
        tmp5[rel_pos_mask_plus.bool()] = 1
        attention_mask_ner_tmp = tmp5

        inputs_ner = {'input_ids': input_ids_ner,
                      'token_type_ids': token_type_ids_ner,
                      'attention_mask': attention_mask_ner_tmp,
                      # 'label_ids_ner': inputs['label_ids_ner'].long()
                      }

        try:
            outputs_ner = model_ner(**inputs_ner)[0]
        except BaseException:
            print('23')

        _, results = torch.max(outputs_ner, dim=2)
        results_np = results.cpu().numpy()
        attention_position_np = rel_pos.cpu().numpy()

        results_list = results_np.tolist()
        attention_position_list = attention_position_np.tolist()
        predict_relation_list = relation_output_sigmoid_number.long().tolist()
        input_ids_list = input_ids_ner.tolist()

        processed_results_list = []
        processed_input_ids_list = []
        for idx, result in enumerate(results_list):
            tmp1 = result[0: attention_position_list[idx]-1]
            tmp2 = input_ids_list[idx][0: attention_position_list[idx]-1]
            processed_results_list.append(tmp1)
            processed_input_ids_list.append(tmp2)

        processed_results_list_BIO = []
        for result in processed_results_list:
            processed_results_list_BIO.append([label_map_ner[token] for token in result])

        # 把结果剥离出来
        index = 0
        triple_output = []

        for ids, BIOS in zip(processed_input_ids_list, processed_results_list_BIO):
            labels = process(ids, BIOS)
            # r = label_map_seq[predict_relation_list[index]]
            r = predict_relation_list[index]

            if len(labels['subject']) == 0:
                h = None
            else:
                h = labels['subject'][0]
                # h = ''.join(tokenizer.convert_ids_to_tokens(h))

            if len(labels['object']) == 0:
                t = None
            else:
                t = labels['object'][0]
                # t = ''.join(tokenizer.convert_ids_to_tokens(t))

            triple_output.append(OutputExample(h=h, r=r, t=t))

            index = index + 1

        return triple_output, inputs['label_ids']


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

    # Set seed
    set_seed(training_args.seed)

    # 读取ner的label
    # ["O", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "Relation"]
    labels_ner = get_labels_ner()
    label_map_ner: Dict[int, str] = {i: label for i, label in enumerate(labels_ner)}
    num_labels_ner = len(labels_ner)

    # 读取seq的label
    labels_seq = get_labels_seq()
    label_map_seq: Dict[int, str] = {i: label for i, label in enumerate(labels_seq)}
    num_labels_seq = len(labels_seq)

    model_name_or_path = model_args.seq_model_path
    # 读取待训练的seq模型
    config = BertConfig.from_pretrained(
        model_name_or_path,
        # './vocab.txt',
        num_labels=num_labels_seq,
        # id2label=label_map_seq,
        label2id={label: i for i, label in enumerate(labels_ner)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model_seq = BertForRelationClassification.from_pretrained(
        model_name_or_path,
        # from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        # label_map_seq=label_map_seq,
        # label_map_ner=label_map_ner
    )

    model_name_or_path = model_args.ner_model_path
    # 读取待训练的ner模型
    config = BertConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels_ner,
        id2label=label_map_ner,
        label2id={label: i for i, label in enumerate(labels_ner)},
        cache_dir=model_args.cache_dir,
    )
    # tokenizer = BertTokenizer.from_pretrained(
    #     model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     use_fast=model_args.use_fast,
    # )
    model_ner = BertForNER.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    model_ner.to(training_args.device)
    model_ner.eval()

    model_seq.to(training_args.device)
    model_seq.eval()

    # texts = ['茶树茶网蝽，Stephanitis chinensis Drake，属半翅目网蝽科冠网椿属的一种昆虫',
    #         '如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈',
    #         '《如果能学会不在乎》是由李玲玉演唱的一首歌曲，收录在《大地的母亲》专辑里']

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
    )

    dataloader = DataLoader(
        test_dataset,
        batch_size=training_args.eval_batch_size,
        sampler=RandomSampler(test_dataset),
        collate_fn=openue_data_collator_interactive,
    )

    epoch_iterator = tqdm(dataloader, desc="Iteration")

    # 计算
    TP = 0
    TP_FP = 0
    TP_FN = 0

    for step, inputs in enumerate(epoch_iterator):
        triple_output, triple_label = predict(model_seq, model_ner, inputs=inputs, training_args=training_args,
                            label_map_ner=label_map_ner, label_map_seq=label_map_seq,
                            tokenizer=tokenizer)
        TP_FN = TP_FN + 1
        # 多关系预测，batch_size为1
        triple_label = triple_label[0]
        for triple in triple_output:
            h = triple.h
            # r = triple.r
            t = triple.t
            if h is None or t is None:
                continue
            TP_FP = TP_FP + 1
            if triple in triple_label:
                TP = TP + 1
                break

    p = TP / TP_FP
    r = TP / TP_FN
    f = 2 * p * r / (p + r)
    print('p', str(p))
    print('r', str(r))
    print('f1', str(f))

if __name__ == "__main__":
    main()

