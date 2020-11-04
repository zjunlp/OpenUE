[**中文说明**](https://github.com/zjunlp/openue/blob/master/README_CN.md) | [**English**](https://github.com/zjunlp/openue/)
<p align="center">
    <a href="https://github.com/zjunlp/openue"> <img src="https://raw.githubusercontent.com/zjunlp/openue/master/docs/images/logo.png" width="400"/></a>
</p>

<p align="center">
<strong> OpenUE is a toolkit that provides a off-the-shelf framework to implement lots of NLP extraction tasks. 
    </strong>
</p>
    <p align="center">
    <a href="https://badge.fury.io/py/openue">
        <img src="https://badge.fury.io/py/openue.svg">
    </a>
    <a href="https://github.com/zjunlp/openue/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/zjunlp/openue">
    </a>
        <a href="http://openue.top">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
</p>


OpenUE allows users ranging from beginner python coders to experienced machine learning engineers to leverage
lots of NLP extraction  tasks in one easy-to-use python package.

**Key Features**

  - [Full Guides and API Documentation](https://openue-docs.readthedocs.io/en/latest/) 

  - Unified API for NLP Tasks with SOTA Pretrained Models (Adaptable with BERT, XLNet, etc.)
    - Entity and Realation Extraction
    - Intent and Slot Filling
    - Event Extraction
    - <em> More in development </em>
  - Training and Inference Interface
  - Rapid NLP Model Deployment
  - [Dockerizing OpenUE with GPUs](https://hub.docker.com/r/)
    - Easily build and run OpenUE containers leveraging NVIDIA GPUs with Docker
## Environment
  - python3.6
  - [requirements.txt](https://github.com/zjunlp/openue/blob/master/requirements.txt) 

## Quick Start

#### Requirements and Installation

##### Anaconda Environment

```
conda create -n openue python=3.6
conda activate openue
conda install  --file requirements.txt 
```

#### Examples and General Use

Once you have installed OpenUE, here are a few examples of what you can run with OpenUE modules:

##### Entity and Relation Extraction Example

1. ***Data Preprocessing***. Put the pretrined language model (e.g., [BERT](https://github.com/google-research/bert)) in the ***pretrained_model*** folder and put all raw data (run script download_ske.sh in the benchmark folder) in the ***raw_data folder***, run
```
sh download_ske_dataset.sh
sh download_bert_cn.sh
sh preprocess.sh  ske
```
2. ***Train Sequence Labeling & Classification Model***. Set all parameters in the file config.py and run 
```
sh train_seq.sh ske
sh train_class.sh ske
```

3. ***Test & Evaluation***. Run 

```
python predict.sh ske
```
4. ***Export & Serving***. Run
```
sh export_seq.sh ske
sh export_class.sh ske
sh serving.sh
```
5. ***Interactive Prediction***. Run
```
python  predict_online.py
```
6. ***Demo***.Run
```
python app.py  ske
```
## Tools

```python
>>> import openue
>>> model = openue.get_model('ske_bert_entity_relation')
>>> res = model.infer('《上海滩》是刘德华的音乐作品，黄沾作曲，收录在《【歌单】酷我热门单曲合辑》专辑中')
>>> print(res)
"spo_list": [{"object_type": "人物", "predicate": "作曲", "object": "黄沾", "subject_type": "歌曲", "subject": "上海滩"}, {"object_type": "音乐专辑", "predicate": "所属专辑", "object": "【歌单】酷我热门单曲合辑", "subject_type": "歌曲", "subject": "上海滩"}, {"object_type": "人物", "predicate": "歌手", "object": "刘德华", "subject_type": "歌曲", "subject": "上海滩"}]
```
Note that it may take a few minutes to download checkpoint and data for the first time. Then use `infer` to do sentence-level entity and relation extraction

## Acknowledgement

## How to Cite

If you use or extend our work, please cite the following paper:

```
@inproceedings{zhang-2020-opennue,
    title = "{O}pe{UE}: An Open Toolkit of Universal Extraction from Text",
    author = "Ningyu Zhang, Shumin Deng, Zhen Bi, Haiyang Yu, Jiacheng Yang, Mosha Chen, Fei Huang, Wei Zhang, Huajun Chen",
    year = "2020",
}
```
