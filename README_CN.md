[**中文说明**](https://github.com/zjunlp/openue/blob/master/README_CN.md) | [**English**](https://github.com/zjunlp/openue/)
<p align="center">
    <a href="https://github.com/zjunlp/openue"> <img src="https://raw.githubusercontent.com/zjunlp/openue/master/docs/images/logo.png" width="400"/></a>
</p>

<p align="center">
<strong> OpenUE 是一个提供了大量通用抽取功能的工具包。
    </strong>
</p>
<p align="center">
    <a href="https://circleci.com/gh/zju/openue">
        <img src="https://img.shields.io/circleci/build/github/zju/openue/master?token=c19c48a56cf6010fed1a63a9bae86acc72e91c24">
    </a>
    <a href="https://badge.fury.io/py/openue">
        <img src="https://badge.fury.io/py/openue.svg">
    </a>
    <a href="https://github.com/zju/openue/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/zjunlp/openue">
    </a>
</p>

OpenUE 是一个简单可用的通用自然语言信息抽取工具，适用于python 初学者或有经验的机器学习开发人员。

**特点**

  - [用户文档](https://openue-docs.readthedocs.io/en/latest/) 

  - 基于SOTA模型的NLP抽取任务 (兼容BERT, XLNet等预训练模型.)
    - 实体关系抽取
    - 意图识别和槽位填充
    - 事件抽取
    - <em> 更多的任务 </em>
  - 训练和测试接口
  - 快速部署NLP模型
  - [容器部署](https://hub.docker.com/r/)
    - 敏捷部署支持GPU的容器
## 环境
  - python3.6
  - [requirements.txt](https://github.com/zju/openue/blob/master/requirements.txt) 

## 快速开始

#### 安装

##### Anaconda 环境

```
conda create -n openue python=3.6
conda activate openue
conda install  --file requirements.txt 
```

#### 使用方式

用户可通过以下几个简单的步骤实现基于OpenUE的抽取模型训练和部署

##### Entity and Relation Extraction Example

1. ***数据预处理***. 下载预训练语言模型 (e.g., [BERT](https://github.com/google-research/bert)) 并放置到 ***pretrained_model*** 文件夹，下载训练数据 and put all raw data 并放到 ***raw_data folder***, 运行以下脚本
```
sh download_ske_dataset.sh
sh download_bert_cn.sh
sh preprocess.sh  ske
```
2. ***训练分类和序列标注模型***. 在config.py文件配置好参数，并运行 
```
sh train_seq.sh ske
sh train_class.sh ske
```

3. ***测试和评估***. 运行

```
python predict.sh ske
```
4. ***导出和服务化***. 运行
```
sh export_seq.sh ske
sh serving_cls.sh ske
sh serving.sh
```
5. ***交互式预测***. 运行
```
python  predict_online.py
```
6. ***在线演示***.运行
```
python app.py  ske
```
## 工具

```python
>>> import openuee
>>> model = openue.get_model('ske_bert_entity_relation')
>>> res = model.infer('《上海滩》是刘德华的音乐作品，黄沾作曲，收录在《【歌单】酷我热门单曲合辑》专辑中')
>>> print(res)
"spo_list": [{"object_type": "人物", "predicate": "作曲", "object": "黄沾", "subject_type": "歌曲", "subject": "上海滩"}, {"object_type": "音乐专辑", "predicate": "所属专辑", "object": "【歌单】酷我热门单曲合辑", "subject_type": "歌曲", "subject": "上海滩"}, {"object_type": "人物", "predicate": "歌手", "object": "刘德华", "subject_type": "歌曲", "subject": "上海滩"}]
```
请注意，第一次下载检查点和数据可能要花费几分钟。 然后使用`infer`进行句子级实体和关系提取。


## 引用

如果您使用或扩展我们的工作，请引用以下文章：

```
@inproceedings{zhang-2020-opennue,
    title = "{O}pe{UE}: An Open Toolkit of Universal Extraction from Text",
    author = "Ningyu Zhang, Shumin Deng, Zhen Bi, Haiyang Yu, Jiacheng Yang, Mosha Chen, Fei Huang, Wei Zhang, Huajun Chen",
    year = "2020",
}
```
