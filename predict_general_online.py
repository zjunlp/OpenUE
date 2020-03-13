#!/usr/bin/env python
# encoding: utf-8
from bert import tokenization
from extract_features import InputExample, InputSeqExample, convert_class_to_features, convert_seq_to_features
import numpy as np
import requests
import os
import time
import json
from config import config

vocab_file = os.environ.get('vocab_file', 'pretrained_model/chinese_L-12_H-768_A-12/vocab.txt')
max_token_len = os.environ.get('max_token_len', 128)
label_list = config.class_label
token_label_list = config.token_label
schemas_dict_relation_2_object_subject_type = config.schema
tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
token_label_id2label = {}
for (i, label) in enumerate(token_label_list):
    token_label_id2label[i] = label

class SPO_Management():
    def __init__(self, TEST_DATA_DIR, MODEL_OUTPUT_DIR, Competition_Mode=False):
        self.relationship_label_list = config.class_label
        self.text_sentence_list = text_sentence_list
        self.token_in_not_NUK_list = token_in_not_NUK_list
        self.token_label_list = token_label_list

    def get_input_list(self,):
        [text_sentence_list, token_in_not_NUK_list, token_label_list] = self.text_sentence_list,self.token_in_not_NUK_list, self.token_label_list
        reference_spo_list = [None] * len(text_sentence_list)
        return text_sentence_list, token_in_not_NUK_list, token_label_list, reference_spo_list


    #合并由WordPiece切分的词和单字
    def _merge_WordPiece_and_single_word(self, entity_sort_list):
        # [..['B-SUB', '新', '地', '球', 'ge', '##nes', '##is'] ..]---> [..('SUB', '新地球genesis')..]
        entity_sort_tuple_list = []
        for a_entity_list in entity_sort_list:
            entity_content = ""
            entity_type = None
            for idx, entity_part in enumerate(a_entity_list):
                if idx == 0:
                    entity_type = entity_part
                    if entity_type[:2] not in ["B-", "I-"]:
                        break
                else:
                    if entity_part.startswith("##"):
                        entity_content += entity_part.replace("##", "")
                    else:
                        entity_content += entity_part
            if entity_content != "":
                entity_sort_tuple_list.append((entity_type[2:], entity_content))
        return entity_sort_tuple_list

    # 把spo_out.txt 的[SPO_SEP] 分割形式转换成标准列表字典形式
    # 妻子 人物 人物 杨淑慧 周佛海[SPO_SEP]丈夫 人物 人物 周佛海 杨淑慧 ---> dict
    def preprocessing_reference_spo_list(self, refer_spo_str):
        refer_spo_list = refer_spo_str.split("[SPO_SEP]")
        refer_spo_list = [spo.split(" ") for spo in refer_spo_list]
        refer_spo_list = [dict([('predicate', spo[0]),
                                ('object_type', spo[2]), ('subject_type', spo[1]),
                                ('object', spo[4]), ('subject', spo[3])]) for spo in refer_spo_list]
        refer_spo_list.sort(key= lambda item:item['predicate'])
        return refer_spo_list

    # 把模型输出实体标签按照原句中相对位置输出
    def model_token_label_2_entity_sort_tuple_list(self, token_in_not_UNK_list, predicate_token_label_list):
        """
        :param token_in_not_UNK:  ['紫', '菊', '花', '草', '是', '菊', '目', '，', '菊', '科', '，', '松', '果', '菊', '属', '的', '植', '物']
        :param predicate_token_label: ['B-SUB', 'I-SUB', 'I-SUB', 'I-SUB', 'O', 'B-OBJ', 'I-OBJ', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        :return: [('SUB', '紫菊花草'), ('OBJ', '菊目')]
        """
        # 除去模型输出的特殊符号
        def preprocessing_model_token_lable(predicate_token_label_list, token_in_list_lenth):
            # ToDo:检查错误，纠错
            if predicate_token_label_list[0] == "[CLS]":
                predicate_token_label_list = predicate_token_label_list[1:]  # y_predict.remove('[CLS]')
            if len(predicate_token_label_list) > token_in_list_lenth:  # 只取输入序列长度即可
                predicate_token_label_list = predicate_token_label_list[:token_in_list_lenth]
            return predicate_token_label_list
        # 预处理标注数据列表
        predicate_token_label_list = preprocessing_model_token_lable(predicate_token_label_list, len(token_in_not_UNK_list))
        entity_sort_list = []
        entity_part_list = []
        #TODO:需要检查以下的逻辑判断，可能写的不够完备充分
        for idx, token_label in enumerate(predicate_token_label_list):
            # 如果标签为 "O"
            if token_label == "O":
                # entity_part_list 不为空，则直接提交
                if len(entity_part_list) > 0:
                    entity_sort_list.append(entity_part_list)
                    entity_part_list = []
            # 如果标签以字符 "B-" 开始
            if token_label.startswith("B-"):
                # 如果 entity_part_list 不为空，则先提交原来 entity_part_list
                if len(entity_part_list) > 0:
                    entity_sort_list.append(entity_part_list)
                    entity_part_list = []
                entity_part_list.append(token_label)
                entity_part_list.append(token_in_not_UNK_list[idx])
                # 如果到了标签序列最后一个标签处
                if idx == len(predicate_token_label_list) - 1:
                    entity_sort_list.append(entity_part_list)
            # 如果标签以字符 "I-"  开始 或者等于 "[##WordPiece]"
            if token_label.startswith("I-") or token_label == "[##WordPiece]":
                # entity_part_list 不为空，则把该标签对应的内容并入 entity_part_list
                if len(entity_part_list) > 0:
                    entity_part_list.append(token_in_not_UNK_list[idx])
                    # 如果到了标签序列最后一个标签处
                    if idx == len(predicate_token_label_list) - 1:
                        entity_sort_list.append(entity_part_list)
            # 如果遇到 [SEP] 分隔符，说明需要处理的标注部分已经结束
            if token_label == "[SEP]":
                break
        entity_sort_tuple_list = self._merge_WordPiece_and_single_word(entity_sort_list)
        return entity_sort_tuple_list

    # 生成排好序的关系列表和实体列表
    def produce_relationship_and_entity_sort_list(self):
        text_sentence_list, token_in_not_NUK_list, token_label_list, reference_spo_list = self.get_input_list()
        for [text_sentence, token_in_not_UNK, token_label, refer_spo_str] in\
                zip(text_sentence_list, token_in_not_NUK_list, token_label_list, reference_spo_list):
            text = text_sentence.split("\t")[0]
            text_predicate = text_sentence.split("\t")[1]
            token_in = token_in_not_UNK.split("\t")[0].split(" ")
            token_in_predicate = token_in_not_UNK.split("\t")[1]
            assert text_predicate == token_in_predicate
            token_label_out = token_label
            entity_sort_tuple_list = self.model_token_label_2_entity_sort_tuple_list(token_in, token_label_out)
            yield text, text_predicate, entity_sort_tuple_list, None

    def gen_triple(self,  keep_empty_spo_list=False):
        output_dict = dict()
        for text, text_predicate, entity_sort_tuple_list, refer_spo_list in self.produce_relationship_and_entity_sort_list():
            object_type, subject_type = schemas_dict_relation_2_object_subject_type[text_predicate][0]
            subject_list = [value for name, value in entity_sort_tuple_list if name == "SUB"]
            subject_list = list(set(subject_list))
            subject_list = [value for value in subject_list if len(value) >= 2]
            object_list = [value for name, value in entity_sort_tuple_list if name == "OBJ"]
            object_list = list(set(object_list))
            object_list = [value for value in object_list if len(value) >= 2]
            if len(subject_list) == 0 or len(object_list) == 0:
                output_dict.setdefault(text, [])
            for subject_value in subject_list:
                for object_value in object_list:
                    output_dict.setdefault(text, []).append({"object_type": object_type, "predicate": text_predicate,
                                                             "object": object_value, "subject_type": subject_type,
                                                             "subject": subject_value})
        for text, spo_list in output_dict.items():
            line_dict = dict()
            line_dict["text"] = text
            line_dict["spo_list"] = spo_list
            line_json = json.dumps(line_dict, ensure_ascii=False)
            return line_json 

def preprocess(text,flag):
    if flag == 'class':
        text_a =" ".join(tokenizer.tokenize(text))
        example = InputExample(unique_id=None, text_a=text_a, text_b=None)
        feature = convert_class_to_features(example, max_token_len, tokenizer)
    else:
        example = InputSeqExample(guid=None, text_token=text, token_label=None)
        feature = convert_seq_to_features(example, max_token_len, tokenizer,label_list)
    input_ids = np.reshape([feature.input_ids], (1, max_token_len))
    return {
        "inputs": {"input_ids": input_ids.tolist()} 
    }


if __name__ == '__main__':
    if  True:
        text = "《盗墓笔记》是2014年欢瑞世纪影视传媒股份有限公司出品的一部网络季播剧，改编自南派三叔所著的同名小说，由郑保瑞和罗永昌联合导演，李易峰、杨洋、唐嫣、刘天佐、张智尧、魏巍等主演。"
        text = input("Input test sentence:\n")
        start = time.time()
        total_start = time.time()
        resp = requests.post('http://127.0.0.1:8501/v1/models/class:predict', json=preprocess(text,'class'))
        end = time.time()
        print(f"predicate prediction time consuming:{int((end - start) * 1000)}ms")
        probabilities = resp.json()['outputs'][0]
        predicate_predict = []
        for idx, class_probability in enumerate(probabilities):
            if class_probability > 0.5:
                predicate_predict.append(label_list[idx])
        print(predicate_predict)
        text_sentence_list = []
        token_in_not_NUK_list = []
        token_label_list = []
        start = time.time()
        for item in predicate_predict:
            text = "姚明（Yao Ming），男，汉族，无党派人士，1980年9月12日出生于上海市徐汇区，祖籍江苏省苏州市吴江区震泽镇，前中国职业篮球运动员，司职中锋，现任亚洲篮球联合会主席、中国篮球协会主席、中职联公司董事长兼总经理， [1-3]  改革先锋奖章获得者 [4]  。"
            item = "国籍"
            text_token =" ".join(tokenizer.tokenize(text))
            text2 = text_token + "\t" + item
            text2_raw = text + "\t" + item
            resp = requests.post('http://127.0.0.1:8501/v1/models/seq:predict', json=preprocess(text2,'seq'))
            spo_res_raw = resp.json()['outputs']
            print(spo_res_raw)
            predicate_probabilities = spo_res_raw['predicate_probabilities'][0]
            token_label_predictions = spo_res_raw['token_label_predictions'][0]
            token_label_output = [token_label_id2label[id] for id in token_label_predictions]
            text_sentence_list.append(text2_raw)
            token_in_not_NUK_list.append(text2)
            token_label_list.append(token_label_output)
        end = time.time()
        print(f"subject and object  prediction time consuming:{int((end - start) * 1000)}ms")
        spo_manager = SPO_Management(text_sentence_list,token_in_not_NUK_list,token_label_list)
        spo_list = spo_manager.gen_triple(keep_empty_spo_list=True)
        print(spo_list)
        total_end = time.time()
        print(f"total  prediction time consuming:{int((total_end - total_start) * 1000)}ms")
