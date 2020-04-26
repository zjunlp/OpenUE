import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bert")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import tokenization
from config import config

class Model_data_preparation(object):

    def __init__(self, RAW_DATA_INPUT_DIR="raw_data", DATA_OUTPUT_DIR="classfication_data",
                 vocab_file_path="vocab.txt", do_lower_case=True, General_Mode=False, Valid_Model=False):
        self.bert_tokenizer = tokenization.FullTokenizer(vocab_file=self.get_vocab_file_path(vocab_file_path),
                                                         do_lower_case=do_lower_case)  # 初始化 bert_token 工具
        self.DATA_INPUT_DIR = self.get_data_input_dir(RAW_DATA_INPUT_DIR)
        self.DATA_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), DATA_OUTPUT_DIR)
        self.General_Mode = General_Mode
        self.Valid_Model= Valid_Model

    def get_data_input_dir(self, DATA_INPUT_DIR):
        DATA_INPUT_DIR = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")), DATA_INPUT_DIR)
        return DATA_INPUT_DIR

    def get_vocab_file_path(self, vocab_file_path):
        print(vocab_file_path)
        return vocab_file_path

    def separate_raw_data_and_token_labeling(self):
        if not os.path.exists(self.DATA_OUTPUT_DIR):
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "train"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "valid"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "test"))

        file_set_type_list = ["train", "valid", "test"]
        #file_set_type_list = ["valid"]
        if self.Valid_Model:
            file_set_type_list = ["test"]
        for file_set_type in file_set_type_list:
            print("produce data will store in: ", os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type)))
            if file_set_type in ["train", "valid", "test"]:
                predicate_out_f = open(
                    os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "predicate_out.txt"), "w",
                    encoding='utf-8')
            text_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "text.txt"), "w",
                          encoding='utf-8')
            token_in_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_in.txt"), "w",
                              encoding='utf-8')
            token_in_not_UNK_f = open(
                os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_in_not_UNK.txt"), "w",
                encoding='utf-8')

            def predicate_to_predicate_file(spo_list):
                predicate_list = [spo['predicate'] for spo in spo_list]
                predicate_list_str = " ".join(predicate_list)
                predicate_out_f.write(predicate_list_str + "\n")

            def gen_event_file(spo_list):
                predicate_out_f.write(spo_list[0]['event'] + "\n")

            if file_set_type == "train":
                path_to_raw_data_file = "train.json"
            elif file_set_type == "valid":
                path_to_raw_data_file = "valid.json"
            else:
                path_to_raw_data_file = "test.json"

            with open(os.path.join(self.DATA_INPUT_DIR, path_to_raw_data_file), 'r', encoding='utf-8') as f:
                count_numbers = 0
                i = 0
                while True:
                    line = f.readline()
                    if line:
                        count_numbers += 1
                        r = json.loads(line)
                        if file_set_type in ["train", "valid"]:
                            spo_list = r["spo_list"]
                        text = r["text"]
                        if text == "":
                            continue
                        text_tokened = self.bert_tokenizer.tokenize(text)
                        text_tokened_not_UNK = self.bert_tokenizer.tokenize_not_UNK(text)

                        if file_set_type in ["train", "valid"]:
                            if not self.General_Mode:
                                predicate_to_predicate_file(spo_list)
                            else:
                                gen_event_file(spo_list)
                        text_f.write(text + "\n")
                        token_in_f.write(" ".join(text_tokened) + "\n")
                        token_in_not_UNK_f.write(" ".join(text_tokened_not_UNK) + "\n")
                    else:
                        break
            print("all numbers", count_numbers)
            print("\n")
            text_f.close()
            token_in_f.close()
            token_in_not_UNK_f.close()

if __name__ == "__main__":
    RAW_DATA_DIR = config.data_dir
    data_type = sys.argv[1]
    DATA_OUTPUT_DIR = "classification_data/" + data_type
    Vocab_Path = config.bert_vocab_dir
    General_Mode = False
    model_data = Model_data_preparation(
        RAW_DATA_INPUT_DIR=RAW_DATA_DIR, DATA_OUTPUT_DIR=DATA_OUTPUT_DIR, vocab_file_path=Vocab_Path,General_Mode=General_Mode)
    model_data.separate_raw_data_and_token_labeling()
