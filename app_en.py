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
from flask import Flask, render_template, request, jsonify
import sys
from gevent  import monkey
from  gevent import pywsgi
import codecs
import numpy as np
from config_en import config

monkey.patch_all()
app = Flask(__name__)
vocab_file = config.bert_vocab_dir
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
    def __init__(self, text_sentence_list,token_in_not_NUK_list,token_label_list):
        self.relationship_label_list = config.class_label
        self.text_sentence_list = text_sentence_list
        self.token_in_not_NUK_list = token_in_not_NUK_list
        self.token_label_list = token_label_list

    def get_input_list(self,):
        [text_sentence_list, token_in_not_NUK_list, token_label_list] = self.text_sentence_list,self.token_in_not_NUK_list, self.token_label_list
        reference_spo_list = [None] * len(text_sentence_list)
        return text_sentence_list, token_in_not_NUK_list, token_label_list, reference_spo_list

    def _merge_WordPiece_and_single_word(self, entity_sort_list):
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
                        entity_content += " "
                        entity_content += entity_part
            if entity_content != "":
                entity_sort_tuple_list.append((entity_type[2:], entity_content))
        return entity_sort_tuple_list

    def preprocessing_reference_spo_list(self, refer_spo_str):
        refer_spo_list = refer_spo_str.split("[SPO_SEP]")
        refer_spo_list = [spo.split(" ") for spo in refer_spo_list]
        refer_spo_list = [dict([('predicate', spo[0]),
                                ('object_type', spo[2]), ('subject_type', spo[1]),
                                ('object', spo[4]), ('subject', spo[3])]) for spo in refer_spo_list]
        refer_spo_list.sort(key= lambda item:item['predicate'])
        return refer_spo_list

    def model_token_label_2_entity_sort_tuple_list(self, token_in_not_UNK_list, predicate_token_label_list):
        def preprocessing_model_token_lable(predicate_token_label_list, token_in_list_lenth):
            if predicate_token_label_list[0] == "[CLS]":
                predicate_token_label_list = predicate_token_label_list[1:]  # y_predict.remove('[CLS]')
            if len(predicate_token_label_list) > token_in_list_lenth:  # 只取输入序列长度即可
                predicate_token_label_list = predicate_token_label_list[:token_in_list_lenth]
            return predicate_token_label_list
        predicate_token_label_list = preprocessing_model_token_lable(predicate_token_label_list, len(token_in_not_UNK_list))
        entity_sort_list = []
        entity_part_list = []
        for idx, token_label in enumerate(predicate_token_label_list):
            if token_label == "O":
                if len(entity_part_list) > 0:
                    entity_sort_list.append(entity_part_list)
                    entity_part_list = []
            if token_label.startswith("B-"):
                if len(entity_part_list) > 0:
                    entity_sort_list.append(entity_part_list)
                    entity_part_list = []
                entity_part_list.append(token_label)
                entity_part_list.append(token_in_not_UNK_list[idx])
                if idx == len(predicate_token_label_list) - 1:
                    entity_sort_list.append(entity_part_list)
            if token_label.startswith("I-") or token_label == "[##WordPiece]":
                if len(entity_part_list) > 0:
                    entity_part_list.append(token_in_not_UNK_list[idx])
                    if idx == len(predicate_token_label_list) - 1:
                        entity_sort_list.append(entity_part_list)
            if token_label == "[SEP]":
                break
        entity_sort_tuple_list = self._merge_WordPiece_and_single_word(entity_sort_list)
        return entity_sort_tuple_list

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
            #object_type, subject_type = schemas_dict_relation_2_object_subject_type[text_predicate][0]
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
                    if True:
                        output_dict.setdefault(text, []).append({"predicate": text_predicate,
                                                             "object": object_value, 
                                                             "subject": subject_value})
        for text, spo_list in output_dict.items():
            line_dict = dict()
            line_dict["text"] = text
            line_dict["spo_list"] = spo_list
            return line_dict

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

def cut_sentence(text):
    return text.split("。")

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('Demo.html')

@app.route('/IEDemo', methods=['GET', 'POST'])
def IEDemo():
    if request.method == 'GET':
        sentence = request.args.get('q')
    else:
        data = request.get_data()
        sentence = json.loads(data)['query']
    text_list  = cut_sentence(sentence)
    all_res = {}
    for i,text in enumerate(text_list):
        text = text  +"."
        if len(text) == 1:
            continue
        print(text)
        start = time.time()
        total_start = time.time()
        resp = requests.post('http://'+config.tf_serving_addr+'/v1/models/class_wiki:predict', json=preprocess(text,'class'))
        end = time.time()
        print(resp.json())
        rel_time = int((end - start) * 1000)
        print(f"predicate prediction time consuming:{int((end - start) * 1000)}ms")
        probabilities = resp.json()['outputs'][0]
        predicate_predict = []
        candidate = sorted(range(len(probabilities)), key=lambda i: probabilities[i])[-10:]
        print(candidate)
        for item in candidate:
            if probabilities[item] <=0.3:
                continue
            predicate_predict.append(label_list[item])
        text_sentence_list = []
        token_in_not_NUK_list = []
        token_label_list = []
        token_prob_list = []
        start = time.time()
        if len(predicate_predict) == 0:
            continue
        for item in predicate_predict:
            text_token =" ".join(tokenizer.tokenize(text))
            text_not_UNK =" ".join(tokenizer.tokenize_not_UNK(text))
            text2 = text_token + "\t" + item
            text2_raw = text + "\t" + item
            text2_not_UNK = text_not_UNK + "\t" + item
            resp = requests.post('http://'+config.tf_serving_addr+'/v1/models/seq_wiki:predict', json=preprocess(text2,'seq'))
            spo_res_raw = resp.json()['outputs']
            predicate_probabilities = spo_res_raw['predicate_probabilities'][0]
            token_label_predictions = spo_res_raw['token_label_predictions'][0]
            token_label_output = [token_label_id2label[id] for id in token_label_predictions]
            text_sentence_list.append(text2_raw)
            token_in_not_NUK_list.append(text2_not_UNK)
            token_label_list.append(token_label_output)
            token_prob_list.append(token_prob_list)
        end = time.time()
        ent_time = int((end - start) * 1000)
        print(f"subject and object  prediction time consuming:{int((end - start) * 1000)}ms")
        spo_manager = SPO_Management(text_sentence_list,token_in_not_NUK_list,token_label_list)
        spo_list = spo_manager.gen_triple(keep_empty_spo_list=True)

        total_end = time.time()
        total_time = int((total_end - total_start) * 1000)
        print(f"total  prediction time consuming:{int((total_end - total_start) * 1000)}ms")
        spo_list['rel_time'] = rel_time
        spo_list['ent_time'] = ent_time
        spo_list['total_time'] = total_time
        spo_list['class_prob'] = probabilities 
        spo_list['token_pred'] =  token_label_list
        all_res[i] = spo_list
    return json.dumps(all_res,ensure_ascii=False)

if __name__ == '__main__':
    server = pywsgi.WSGIServer(('127.0.0.1', 8887), app)
    server.serve_forever()
