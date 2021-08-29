# OpenUE Pytorch

用户可通过以下几个简单的步骤实现基于OpenUE的抽取模型训练和部署

1. ***下载ske数据集***
```
mkdir dataset
wget  http://47.92.96.190/dataset/ske.tar.gz 
tar zxvf ske.tar.gz -C ske
包含all_50_schemas, train.json, test.json, dev.json四个文件
```
2. ***数据预处理***
```
下载预训练语言模型 (e.g., [bert-base-chinese](https://github.com/google-research/bert)) 并放置到对应文件夹
```
3. ***训练分类模型***
```
python run_seq.py --model_name_or_path $pretrained_model_path/bert-base-chinese --data_dir ./dataset --output_dir ./output_seq --save_steps 5000 --num_train_epochs 30 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --do_train --task seq
```
4. ***训练序列标注模型***
```
python run_ner.py --model_name_or_path $pretrained_model_path/bert-base-chinese --data_dir ./dataset --output_dir ./output_ner --save_steps 2000 --num_train_epochs 20 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --do_train --task ner
```
5. ***模型集成与测试***
```
python Interactive.py --seq_model_path ./output_seq/ --ner_model_path ./output_ner/ --output_dir ./output/ --data_dir ./dataset -per_device_eval_batch_size 1 --do_predict --task interactive
```
## 引用

如果您使用或扩展我们的工作，请引用以下文章：

```
@inproceedings{zhang-2020-opennue,
    title = "{O}pe{UE}: An Open Toolkit of Universal Extraction from Text",
    author = "Ningyu Zhang, Shumin Deng, Zhen Bi, Haiyang Yu, Jiacheng Yang, Mosha Chen, Fei Huang, Wei Zhang, Huajun Chen",
    year = "2020",
}
