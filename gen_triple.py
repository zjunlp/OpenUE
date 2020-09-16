# coding=utf-8
import os
import sys
import json
from config import config

def get_latest_model_predict_data_dir(new_epochs_ckpt_dir=None):
    def new_report(test_report):
        lists = os.listdir(test_report)  
        lists.sort(key=lambda fn: os.path.getmtime(test_report + "/" + fn)) 
        file_new = os.path.join(test_report, lists[-1])  
        return file_new
    if new_epochs_ckpt_dir is None:
        input_new_epochs = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "output")), "sequnce_infer_out")
        new_ckpt_dir = new_report(input_new_epochs)
        input_new_epochs_ckpt = os.path.join(input_new_epochs, new_ckpt_dir)
        new_epochs_ckpt_dir = new_report(input_new_epochs_ckpt)
    if not os.path.exists(new_ckpt_dir):
        raise ValueError("path do not exist！{}".format(new_epochs_ckpt_dir))
    return new_epochs_ckpt_dir

schemas_dict_relation_2_object_subject_type = config.schema

class File_Management(object):
    def __init__(self, TEST_DATA_DIR=None, MODEL_OUTPUT_DIR=None, Competition_Mode=True):
        self.TEST_DATA_DIR = TEST_DATA_DIR
        #self.MODEL_OUTPUT_DIR = get_latest_model_predict_data_dir(MODEL_OUTPUT_DIR)
        self.MODEL_OUTPUT_DIR = MODEL_OUTPUT_DIR
        self.Competition_Mode = Competition_Mode

    def file_path_and_name(self):
        text_sentence_file_path = os.path.join(self.TEST_DATA_DIR, "text_and_one_predicate.txt")
        token_in_file_path = os.path.join(self.TEST_DATA_DIR, "token_in_not_UNK_and_one_predicate.txt")
        predicate_token_label_file_path = os.path.join(self.MODEL_OUTPUT_DIR, "token_label_predictions.txt")

        file_path_list = [text_sentence_file_path, token_in_file_path, predicate_token_label_file_path]
        file_name_list = ["text_sentence_list", "token_in_not_NUK_list ", "token_label_list",]
        if not self.Competition_Mode:
            spo_out_file_path = os.path.join(self.TEST_DATA_DIR, "spo_out.txt")
            if os.path.exists(spo_out_file_path):
                file_path_list.append(spo_out_file_path)
                file_name_list.append("reference_spo_list")
        return file_path_list, file_name_list

    def read_file_return_content_list(self):
        file_path_list, file_name_list = self.file_path_and_name()
        content_list_summary = []
        for file_path in file_path_list:
            with open(file_path, "r", encoding='utf-8') as f:
                content_list = f.readlines()
                content_list = [content.replace("\n", "") for content in content_list]
                content_list_summary.append(content_list)

        if self.Competition_Mode:
            content_list_length_summary = [(file_name, len(content_list)) for content_list, file_name in
                                           zip(content_list_summary, file_name_list)]
            file_line_number = self._check_file_line_numbers(content_list_length_summary)
        else:
            file_line_number = len(content_list_summary[0])
            print("first file line number: ", file_line_number)
            print("do not check file line! if you need check file line, set Competition_Mode=True")
        print("\n")
        return content_list_summary, file_line_number

    def _check_file_line_numbers(self, content_list_length_summary):
        content_list_length_file_one = content_list_length_summary[0][1]
        for file_name, file_line_number in content_list_length_summary:
            assert file_line_number == content_list_length_file_one
        return content_list_length_file_one


class Sorted_relation_and_entity_list_Management(File_Management):
    def __init__(self, TEST_DATA_DIR, MODEL_OUTPUT_DIR, Competition_Mode=False):
        File_Management.__init__(self, TEST_DATA_DIR=TEST_DATA_DIR, MODEL_OUTPUT_DIR=MODEL_OUTPUT_DIR, Competition_Mode=Competition_Mode)
        self.relationship_label_list = config.class_label
        self.Competition_Mode = Competition_Mode

    def get_input_list(self,):
        content_list_summary, self.file_line_number = self.read_file_return_content_list()
        if len(content_list_summary) == 4:
            [text_sentence_list, token_in_not_NUK_list, token_label_list, reference_spo_list] = content_list_summary
        elif len(content_list_summary) == 3:
            [text_sentence_list, token_in_not_NUK_list, token_label_list] = content_list_summary
            reference_spo_list = [None] * len(text_sentence_list)
        else:
            raise ValueError("check code!")
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
            token_label_out = token_label.split(" ")
            entity_sort_tuple_list = self.model_token_label_2_entity_sort_tuple_list(token_in, token_label_out)
            if self.Competition_Mode:
                yield text, text_predicate, entity_sort_tuple_list, None
            else:
                if refer_spo_str is not None:
                    refer_spo_list = self.preprocessing_reference_spo_list(refer_spo_str)
                else:
                    refer_spo_list = []
                yield text, text_predicate, entity_sort_tuple_list, refer_spo_list

    def produce_output_file(self, OUT_RESULTS_DIR=None, keep_empty_spo_list=False):
        filename = "subject_predicate_object_predict_output.json"
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
        if keep_empty_spo_list:
            filename = "keep_empty_spo_list_" + filename
        if OUT_RESULTS_DIR is None:
            out_path = filename
        else:
            out_path = os.path.join(OUT_RESULTS_DIR, filename)
        if not os.path.exists(OUT_RESULTS_DIR):
            os.makedirs(OUT_RESULTS_DIR)
        result_json_write_f = open(out_path, "w", encoding='utf-8')
        count_line_number = 0
        count_empty_line_number = 0
        for text, spo_list in output_dict.items():
            count_line_number += 1
            line_dict = dict()
            line_dict["text"] = text
            line_dict["spo_list"] = spo_list
            line_json = json.dumps(line_dict, ensure_ascii=False)
            if len(spo_list) == 0:
                count_empty_line_number += 1
            if keep_empty_spo_list:
                result_json_write_f.write(line_json + "\n")
            else:
                if len(spo_list) > 0:
                    result_json_write_f.write(line_json + "\n")

if __name__ == '__main__': 
    TEST_DATA_DIR = "openue/sequence_labeling/sequence_labeling_data/" + sys.argv[1] + "/test/"
    MODEL_OUTPUT_DIR = "output/sequnce_infer_out/wwm/epoch9/"
    OUT_RESULTS_DIR = "output/predict_text_spo_list_result"
    Competition_Mode = True
    spo_list_manager = Sorted_relation_and_entity_list_Management(TEST_DATA_DIR, MODEL_OUTPUT_DIR, Competition_Mode=Competition_Mode)
    spo_list_manager.produce_output_file(OUT_RESULTS_DIR=OUT_RESULTS_DIR, keep_empty_spo_list=True)
