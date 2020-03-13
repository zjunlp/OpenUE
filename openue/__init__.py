import sys
import os
import subprocess
from config import config
sys.path.append("../../")

model_ip = ''
cur_dir = '/apsarapangu/disk1/ningyu.zny/OpenUE/'
cmd_classification  = '''python '''+cur_dir+'''run_classification.py \
  --task_name=classification \
  --do_predict=true \
  --data_dir=tmp/classification/classification_data \
  --vocab_file='''+config.bert_vocab_dir+''' \
  --bert_config_file='''+config.bert_config_dir+''' \
  --init_checkpoint='''+config.class_model_dir+''' \
  --max_seq_length='''+config.max_len+''' \
  --output_dir='''+config.middle_out_dir

cmd_middle = "python "+cur_dir+"openue/classification/prepare_data_for_labeling_infer.py"

cmd_sequence = '''python '''+cur_dir+'''run_sequnce_labeling.py \
  --task_name=SKE_2019 \
  --do_predict=true \
  --data_dir=tmp/sequence_labeling/sequence_labeling_data \
  --vocab_file='''+config.bert_vocab_dir+''' \
  --bert_config_file='''+config.bert_config_dir+''' \
  --init_checkpoint='''+config.seq_model_dir+''' \
  --max_seq_length='''+config.max_len+''' \
  --output_dir='''+config.out_dir
train_class = '''

'''

train_seq = '''

'''
pre_class = "python "+cur_dir+"openue/classification/classification_data_manager.py"
pre_seq = "python "+cur_dir+"openue/sequence_labeling/sequence_labeling_data_manager.py"

def get_model(model_name):
    if model_name == 'ske':
        download_model(model_ip,'ske')

def infer():
    os.system(pre_class)
    os.system(pre_seq)
    os.system(cmd_classification)
    os.system(cmd_middle)
    os.system(cmd_sequence)
def train():
    pass

def train_seq():
    pass

def train_class():
    pass
