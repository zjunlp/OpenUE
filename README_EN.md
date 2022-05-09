[**中文说明**](https://github.com/zjunlp/OpenUE/blob/main/README.md) | [**English**](https://github.com/zjunlp/OpenUE/blob/main/README_EN.md)

<p align="center">
    <a href="https://github.com/zjunlp/openue"> <img src="https://github.com/zjunlp/OpenUE/blob/main/imgs/logo.png" width="400"/></a>
</p>

<p align="center">
<strong> OpenUE is a lightweight toolkit for knowledge graph extraction. 
    </strong>
</p>
    <p align="center">
    <a href="https://badge.fury.io/py/openue">
        <img src="https://badge.fury.io/py/openue.svg">
    </a>
    <a href="https://github.com/zjunlp/OpenUE/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/zjunlp/openue.svg?color=green">
    </a>
        <a href="http://openue.zjukg.org">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
</p>

[OpenUE](https://aclanthology.org/2020.emnlp-demos.1/) is a lightweight knowledge graph extraction tool.

**Features**


  - Knowledge extraction task based on pre-training language model (compatible with pre-training models such as BERT and Roberta.)
    - Named Entity Extraction
    - Event Extraction
    - Slot filling and intent detection
    - <em> more tasks </em>
  - Training and testing interface
  - fast deployment of your extraction models

## Environment

  - python3.8
  - requirements.txt


## Architecture

![框架](./imgs/overview1.png)

It mainly includes **three** modules, as `models`,`lit_models` and `data`.

### models module

It stores our three main models, the relationship recognition model for the single sentence, the named entity recognition model for the relationship in the known sentence, and the inference model that integrates the first two. It is mainly derived from the defined pre-trained models in the `transformers` library.

### lit_models module

The code is mainly inherited from `pytorch_lightning.Trainer`. It can automatically build model training under different hardware such as single card, multi-card, GPU, TPU, etc. We define `training_step` and `validation_step` in it to automatically build training logic for training. 

Because its hardware is not sensitive, we can call the OpenUE training module in a variety of different environments.

### data module

The code for different operations on different data sets is stored in `data`. The `tokenizer` in the `transformers` library is used to segment the data and then turn the data into the features we need according to different datasets.

## Quick start

### Install

#### Anaconda 

```
conda create -n openue python=3.8
conda activate openue
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia # depend on your GPU driver version
python setup.py install
```

#### pip

```shell
pip install openue
```

#### pip dev

```shell
python setup.py develop
```

#### How to use

The data format is a `json` file, the specific example is as follows. (in the ske dataset)

```json
{
	"text": "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部",
	"spo_list": [{
		"predicate": "出生地",
		"object_type": "地点",
		"subject_type": "人物",
		"object": "圣地亚哥",
		"subject": "查尔斯·阿兰基斯"
	}, {
		"predicate": "出生日期",
		"object_type": "Date",
		"subject_type": "人物",
		"object": "1989年4月17日",
		"subject": "查尔斯·阿兰基斯"
	}]
}
```

### Train

Store the data in the `./dataset/` directory for training. If the directory is empty, run the following script to automatically download the data set and pre-trained model and start training. Please keep the network open during the process to avoid model and data download failure.

```shell
# training the ner module
./scripts/run_ner.sh
# training the seq module
./scripts/run_seq.sh
```

Here we use a small demo to show the training briefly, in which only one batch is trained to speed up the display.

![框架](./imgs/demo.gif)

### notebook quick start

[ske dataset training notebook](https://github.com/zjunlp/OpenUE/blob/pytorch/ske.ipynb)
Using the Chinese dataset as an example specifically introduces how to use `lit_models`, `models` and `data` in openue. It is convenient for users to construct their own training logic.


[colab quick start](https://colab.research.google.com/drive/1VNhFYcqDbXl1b3HzU8sc-NgbhV2ZyYzW?usp=sharing) Use colab for fast training your OpenUE models.

### support auto parameter tuning（wandb）
```python
# just need to replace the default logger by the wandb logger
logger = pl.loggers.WandbLogger(project="openue")
```

## Fast depolyment

### Install torchserve-docker

[docker download](https://github.com/pytorch/serve/blob/master/docker/README.md)

### Create the `handler` class corresponding to the model

We have placed the corresponding deployment classes `handler_seq.py` and `handler_ner.py` under the `deploy` folder.

```shell
# use `torch-model-archiver` to pack the files
# extra-files need the files below
# 	- `config.json`, `setup_config.json` config。 
# 	- `vocab.txt` : vocab for the tokenizer
# 	- `model.py` : the code for the model

torch-model-archiver --model-name BERTForNER_en  \
	--version 1.0 --serialized-file ./ner_en/pytorch_model.bin \
	--handler ./deploy/handler.py \
	--extra-files "./ner_en/config.json,./ner_en/setup_config.json,./ner_en/vocab.txt,./deploy/model.py" -f

# put the `.mar` file to the model-store，use curl command to deploy the model
sudo cp ./BERTForSEQ_en.mar /home/model-server/model-store/
curl -v -X POST "http://localhost:3001/models?initial_workers=1&synchronous=false&url=BERTForSEQ_en.mar&batch_size=1&max_batch_delay=200"
```
## Members

Zhejiang University：[Ningyu Zhang](https://person.zju.edu.cn/en/ningyu)、Xin Xie、Zhen Bi、Xiang Chen、Haiyang Yu、Shumin Deng、Hongbin Ye、Guozhou Zheng、Huajun Chen

Alibaba DAMO Academy：Mosha Chen、Chuanqi Tan、Fei Huang

<br>

## Citation

If you use or extend our work, please cite the following articles:

```
@inproceedings{DBLP:conf/emnlp/ZhangDBYYCHZC20,
  author    = {Ningyu Zhang and
               Shumin Deng and
               Zhen Bi and
               Haiyang Yu and
               Jiacheng Yang and
               Mosha Chen and
               Fei Huang and
               Wei Zhang and
               Huajun Chen},
  editor    = {Qun Liu and
               David Schlangen},
  title     = {OpenUE: An Open Toolkit of Universal Extraction from Text},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural
               Language Processing: System Demonstrations, {EMNLP} 2020 - Demos,
               Online, November 16-20, 2020},
  pages     = {1--8},
  publisher = {Association for Computational Linguistics},
  year      = {2020},
  url       = {https://doi.org/10.18653/v1/2020.emnlp-demos.1},
  doi       = {10.18653/v1/2020.emnlp-demos.1},
  timestamp = {Wed, 08 Sep 2021 16:17:48 +0200},
  biburl    = {https://dblp.org/rec/conf/emnlp/ZhangDBYYCHZC20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
