import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bert")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import tokenization
from config import config


class Model_data_preparation(object):

    def __init__(self, DATA_INPUT_DIR="raw_data", DATA_OUTPUT_DIR="",
                 vocab_file_path="vocab.txt", do_lower_case=True,General_Mode = False):
        self.bert_tokenizer = tokenization.FullTokenizer(vocab_file=self.get_vocab_file_path(vocab_file_path),
                                                         do_lower_case=do_lower_case)  
        self.DATA_INPUT_DIR = self.get_data_input_dir(DATA_INPUT_DIR)
        self.DATA_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), DATA_OUTPUT_DIR)
        self.General_Mode = General_Mode

    def get_data_input_dir(self, DATA_INPUT_DIR):
        DATAself_INPUT_DIR = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")), DATA_INPUT_DIR)
        return DATA_INPUT_DIR

    def get_vocab_file_path(self, vocab_file_path):
        print(vocab_file_path)
        return vocab_file_path

    def subject_object_labeling(self, spo_list, text):
        def _spo_list_to_spo_predicate_dict(spo_list):
            spo_predicate_dict = dict()
            for spo_item in spo_list:
                predicate = spo_item["predicate"]
                subject = spo_item["subject"]
                object = spo_item["object"]
                spo_predicate_dict.setdefault(predicate, []).append((subject, object))
            return spo_predicate_dict

        def _gen_event_dic(spo_list):
            res = []
            res_d = {}
            predicate = ""
            for spo_item in spo_list:
                predicate = spo_item["event"]
                if 'time' in spo_item: 
                    time = spo_item["time"]
                    res.append(('time',time))
                if 'location' in spo_item:
                    location = spo_item["location"]
                    res.append(('location',location))
                if 'participant' in spo_item:
                    participant = spo_item["participant"]
                    res.append(('participant',participant))
                if 'denoter' in spo_item:
                    denoter = spo_item["denoter"]
                    res.append(('denoter',denoter))
                if 'object' in spo_item:
                    object = spo_item["object"]
                    res.append(('object',object))
            res_d[predicate] = res
            return res_d

        def _index_q_list_in_k_list(q_list, k_list):
            """Known q_list in k_list, find index(first time) of q_list in k_list"""
            q_list_length = len(q_list)
            k_list_length = len(k_list)
            for idx in range(k_list_length - q_list_length + 1):
                t = [q == k for q, k in zip(q_list, k_list[idx: idx + q_list_length])]
                if all(t):
                    idx_start = idx
                    return idx_start

        def _labeling_type(subject_object, so_type):
            tokener_error_flag = False
            so_tokened = self.bert_tokenizer.tokenize(subject_object)
            so_tokened_length = len(so_tokened)
            idx_start = _index_q_list_in_k_list(q_list=so_tokened, k_list=text_tokened)
            if idx_start is None:
                tokener_error_flag = True
                self.bert_tokener_error_log_f.write(subject_object + " @@ " + text + "\n")
                self.bert_tokener_error_log_f.write(str(so_tokened) + " @@ " + str(text_tokened) + "\n")
            else: 
                labeling_list[idx_start] = "B-" + so_type
                if so_tokened_length == 2:
                    labeling_list[idx_start + 1] = "I-" + so_type
                elif so_tokened_length >= 3:
                    labeling_list[idx_start + 1: idx_start + so_tokened_length] = ["I-" + so_type] * (so_tokened_length - 1)
            return tokener_error_flag

        text_tokened = self.bert_tokenizer.tokenize(text)
        text_tokened_not_UNK = self.bert_tokenizer.tokenize_not_UNK(text)
        if not self.General_Mode:
            spo_predicate_dict = _spo_list_to_spo_predicate_dict(spo_list)
        else:
            spo_predicate_dict = _gen_event_dic(spo_list)
        for predicate, spo_list_form in spo_predicate_dict.items():
            tokener_error_flag = False
            labeling_list = ["O"] * len(text_tokened)
            if not self.General_Mode:
                for (spo_subject, spo_object) in spo_list_form:
                    flag_A = _labeling_type(spo_subject, "SUB")
                    flag_B = _labeling_type(spo_object, "OBJ")
                    if flag_A or flag_B:
                        tokener_error_flag = True
            else:
                for item  in spo_list_form:
                    if item[1]== None:
                        continue
                    flag_A = _labeling_type(item[1],item[0])
                    if flag_A:
                        tokener_error_flag = True

            for idx, token in enumerate(text_tokened):
                if token.startswith("##"):
                    labeling_list[idx] = "[##WordPiece]"
            if not tokener_error_flag:
                self.token_label_and_one_prdicate_out_f.write(" ".join(labeling_list)+"\t"+predicate+"\n")
                self.text_f.write(text + "\n")
                self.token_in_f.write(" ".join(text_tokened)+"\t"+predicate+"\n")
                self.token_in_not_UNK_f.write(" ".join(text_tokened_not_UNK) + "\n")

    def separate_raw_data_and_token_labeling(self):
        if not os.path.exists(self.DATA_OUTPUT_DIR):
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "train"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "valid"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "test"))

        for file_set_type in ["train", "valid"]:
            print(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type)))
            self.token_label_and_one_prdicate_out_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_label_and_one_prdicate_out.txt"), "w", encoding='utf-8')
            self.bert_tokener_error_log_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "bert_tokener_error_log.txt"), "w", encoding='utf-8')

            self.text_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "text.txt"), "w", encoding='utf-8')
            self.token_in_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_in.txt"), "w", encoding='utf-8')
            self.token_in_not_UNK_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_in_not_UNK.txt"), "w", encoding='utf-8')

            if file_set_type == "train":
                path_to_raw_data_file = "train.json"
            elif file_set_type == "valid":
                path_to_raw_data_file = "valid.json"
            else:
                pass
            with open(os.path.join(self.DATA_INPUT_DIR, path_to_raw_data_file), 'r', encoding='utf-8') as f:
                count_numbers = 0
                while True:
                    line = f.readline()
                    if line:
                        count_numbers += 1
                        r = json.loads(line)
                        text = r["text"]
                        spo_list = r["spo_list"]
                        self.subject_object_labeling(spo_list=spo_list, text=text)
                    else:
                        break
            print("all numbers", count_numbers)
            self.text_f.close()
            self.token_in_f.close()
            self.token_in_not_UNK_f.close()
            self.token_label_and_one_prdicate_out_f.close()
            self.bert_tokener_error_log_f.close()

if __name__=="__main__":
    DATA_INPUT_DIR = config.data_dir
    data_type = sys.argv[1]
    DATA_OUTPUT_DIR = "sequence_labeling_data/" + data_type
    Vocab_Path = config.bert_vocab_dir
    General_Mode = False
    model_data = Model_data_preparation(General_Mode = General_Mode,DATA_INPUT_DIR=DATA_INPUT_DIR, DATA_OUTPUT_DIR=DATA_OUTPUT_DIR,vocab_file_path=Vocab_Path)
    model_data.separate_raw_data_and_token_labeling()
