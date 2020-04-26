[**中文说明**](https://github.com/zxlzr/OpenUE/blob/master/README_CN.md) | [**English**](https://github.com/zxlzr/OpenUE/)
<p align="center">
    <a href="https://github.com/zxlzr/OpenUE"> <img src="https://raw.githubusercontent.com/zxlzr/OpenUE/master/docs/img/logo.jpg" width="400"/></a>
</p>

<p align="center">
<strong> OpenUE is an open-source  toolkit that provides a off-the-shelf framework to implement lots of NLP extraction tasks. 
    </strong>
</p>
<p align="center">
    <a href="https://circleci.com/gh/zxlzr/OpenUE">
        <img src="https://img.shields.io/circleci/build/github/zxlzr/OpenUE/master?token=c19c48a56cf6010fed1a63a9bae86acc72e91c24">
    </a>
    <a href="https://badge.fury.io/py/openue">
        <img src="https://badge.fury.io/py/openue.svg">
    </a>
    <a href="https://github.com/zxlzr/OpenUE/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/zxlzr/OpenUE">
    </a>
</p>


OpenUE allows users ranging from beginner python coders to experienced machine learning engineers to leverage
lots of NLP extraction  tasks in one easy-to-use python package.

**Key Features**

  - [Full Guides and API Documentation](https://openue-docs.readthedocs.io/en/latest/) 

  - Unified API for NLP Tasks with SOTA Pretrained Models (Adaptable with BERT, XLNet, etc.)
    - Entity and Realation Extraction
    - Intent and Slot Filling
    - Opinion and Apspect Extraction
    - <em> More in development </em>
  - Training and Inference Interface
  - Rapid NLP Model Deployment
  - [Dockerizing OpenUE with GPUs](https://hub.docker.com/r/)
    - Easily build and run OpenUE containers leveraging NVIDIA GPUs with Docker
## Environment
  - python3.6
  - [requirements.txt](https://github.com/zxlzr/OpenUE/blob/master/requirements.txt) 

## Quick Start

#### Requirements and Installation

##### Anaconda Environment

```
conda create -n openue python=3.6
conda activate openue
pip install -r requirements.txt 
```

#### Examples and General Use

Once you have installed OpenUE, here are a few examples of what you can run with OpenUE modules:

##### Entity and Relation Extraction Example

1. ***Data Preprocessing***. Put the pretrined language model (e.g., [BERT](https://github.com/google-research/bert)) in the ***pretrained_model*** folder and put all raw data (run script download_ske.sh in the benchmark folder) in the ***raw_data folder***, run
```
sh download_ske_dataset.sh
sh download_pretrain_cn_bert.sh
sh preprocess.sh 
```
2. ***Train Sequence Labeling & Classification Model***. Set all parameters in the file config.py and run 
```
sh train_sequence_labeling.sh
sh train_classification.sh

```
You can download the checkpoint [here](), extract and put them in the *output* folder.

3. ***Test***. Run
```
python  predict_online.py
```
4. ***Predict***. Run 
```
python gen_triple.py
```
5. ***Serving***. Run
```
sh export_seq.sh
sh serving_cls.sh
```
6. ***Demo***.Run
```
python app.py 
```

## Tools

```python
>>> import openuee
>>> model = openue.get_model('ske_bert_entity_relation')
>>> res = model.infer('《上海滩》是刘德华的音乐作品，黄沾作曲，收录在《【歌单】酷我热门单曲合辑》专辑中')
>>> print(res)
"spo_list": [{"object_type": "人物", "predicate": "作曲", "object": "黄沾", "subject_type": "歌曲", "subject": "上海滩"}, {"object_type": "音乐专辑", "predicate": "所属专辑", "object": "【歌单】酷我热门单曲合辑", "subject_type": "歌曲", "subject": "上海滩"}, {"object_type": "人物", "predicate": "歌手", "object": "刘德华", "subject_type": "歌曲", "subject": "上海滩"}]
```
Note that it may take a few minutes to download checkpoint and data for the first time. Then use `infer` to do sentence-level entity and relation extraction


## How to Cite

If you use or extend our work, please cite the following paper:

```
@inproceedings{zhang-2020-opennue,
    title = "{O}pe{UE}: An Open Toolkit for Universal  Extraction in Text",
    author = "Ningyu Zhang, Shumin Deng, Huajun Chen",
    year = "2020",
}
```
