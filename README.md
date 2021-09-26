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
  - requirements.txt


## 框架图

![框架](./imgs/overview.png)


## 快速开始

### 安装

#### Anaconda 环境

```
conda create -n openue python=3.8
conda activate openue
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia # 视自己Nvidia驱动环境选择对应的cudatoolkit版本
python setup.py install
```

#### 使用方式

数据格式为`json`文件，具体例子如下。

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

将数据存放在`./dataset/`目录下之后进行训练。运行以下脚本，将自动下载数据集和预训练模型并开始训练，过程中请保持网络畅通以免模型和数据下载失败。

```shell
./scripts/run_ner.sh
./scripts/run_seq.sh
```

### notebook快速开始

[ske数据集训练notebook](https://github.com/zjunlp/OpenUE/blob/pytorch/ske.ipynb)
在这个数据集中具体介绍了如何使用openue中的`lit_models`,`models`和`data`。方便用户构建自己的训练逻辑。


## 引用

如果您使用或扩展我们的工作，请引用以下文章：

```
@inproceedings{zhang-2020-opennue,
    title = "{O}pe{UE}: An Open Toolkit of Universal Extraction from Text",
    author = "Ningyu Zhang, Shumin Deng, Zhen Bi, Haiyang Yu, Jiacheng Yang, Mosha Chen, Fei Huang, Wei Zhang, Huajun Chen",
    year = "2020",
}
```
