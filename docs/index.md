<h1 align="center" >
    <strong style="color: rgba(0,0,0,.87);">Kashgari</strong>
</h1>

<p align="center">
    <a href="https://github.com/BrikerMan/kashgari/blob/v2-main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/BrikerMan/kashgari.svg?color=blue&style=popout">
    </a>
    <a href="https://travis-ci.com/BrikerMan/Kashgari">
        <img src="https://travis-ci.com/BrikerMan/Kashgari.svg?branch=master"/>
    </a>
    <a href='https://coveralls.io/github/BrikerMan/Kashgari?branch=master'>
        <img src='https://coveralls.io/repos/github/BrikerMan/Kashgari/badge.svg?branch=master' alt='Coverage Status'/>
    </a>
     <a href="https://pepy.tech/project/kashgari-tf">
        <img src="https://pepy.tech/badge/kashgari-tf"/>
    </a>
    <a href="https://pypi.org/project/kashgari/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/kashgari.svg">
    </a>
</p>

!!! danger
    该文档只包括 Kashgari 1.x，Kashgari 2.x 请阅读 https://kashgari.readthedocs.io/en/v2.0.0/

Kashgari 是一个极简且强大的 NLP 框架，可用于文本分类和标注的学习，研究及部署上线。

- **方便易用** Kashgari 提供了简洁统一的 API 和完善的文档，使其非常方便易用。
- **内置迁移学习模块** Kashgari 通过提供 `BertEmbedding`, `GPT2Embedding`，`WordEmbedding` 等特征提取类，方便利用预训练语言模型实现迁移学习。
- **易扩展** Kashgari 提供简便的接口和继承关系，自行扩展新的模型结构非常方便。
- **可用于生产** 通过把 Kashgari 模型导出为 `SavedModel` 格式，可以使用 TensorFlow Serving 模块提供服务，直接在线上环境使用。

## 我们的使命

- 为 **学术研究者** 提供易于实验的环境，可快速验证理论。
- 为 **NLP初学者** 提供易于学习模仿的生产级别工程。
- 为 **NLP工作者** 提供快速搭建文本分类、文本标注的框架，简化日常工作流程。

## 教程

这是一些详细的教程:

- [教程 1: 文本分类](tutorial/text-classification.md)
- [教程 2: 文本标注](tutorial/text-labeling.md)

还有一些博客文章介绍如何使用 Kashgari:

- [15 分钟搭建中文文本分类模型](https://eliyar.biz/nlp_chinese_text_classification_in_15mins/)
- [基于 BERT 的中文命名实体识别（NER)](https://eliyar.biz/nlp_chinese_bert_ner/)
- [BERT/ERNIE 文本分类和部署](https://eliyar.biz/nlp_train_and_deploy_bert_text_classification/)
- [五分钟搭建一个基于BERT的NER模型](https://www.jianshu.com/p/1d6689851622)
- [Multi-Class Text Classification with Kashgari in 15 minutes](https://medium.com/@BrikerMan/multi-class-text-classification-with-kashgari-in-15mins-c3e744ce971d)

## 快速开始

### 安装

!!!important
    tf.keras 版本 pypi 包重命名为 `kashgari-tf`

该项目基于 Tensorflow 1.14.0 和 Python 3.6+.

```bash
pip install kashgari-tf
# CPU
pip install tensorflow==1.14.0
# GPU
pip install tensorflow-gpu==1.14.0
```

### 基础用法

下面我们用 Bi_LSTM 模型实现一个命名实体识别任务：

```python
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.tasks.labeling import BiLSTM_Model

# 加载内置数据集，此处可以替换成自己的数据集，保证格式一致即可
train_x, train_y = ChineseDailyNerCorpus.load_data('train')
test_x, test_y = ChineseDailyNerCorpus.load_data('test')
valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

model = BiLSTM_Model()
model.fit(train_x, train_y, valid_x, valid_y, epochs=50)

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 97)                0
_________________________________________________________________
layer_embedding (Embedding)  (None, 97, 100)           320600
_________________________________________________________________
layer_blstm (Bidirectional)  (None, 97, 256)           235520
_________________________________________________________________
layer_dropout (Dropout)      (None, 97, 256)           0
_________________________________________________________________
layer_time_distributed (Time (None, 97, 8)             2056
_________________________________________________________________
activation_7 (Activation)    (None, 97, 8)             0
=================================================================
Total params: 558,176
Trainable params: 558,176
Non-trainable params: 0
_________________________________________________________________
Train on 20864 samples, validate on 2318 samples
Epoch 1/50
20864/20864 [==============================] - 9s 417us/sample - loss: 0.2508 - acc: 0.9333 - val_loss: 0.1240 - val_acc: 0.9607

"""
```

### 使用 Bert 语言模型

```python
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiGRU_Model
from kashgari.corpus import ChineseDailyNerCorpus

# 此处需要自行下载 BERT 权重
bert_embedding = BERTEmbedding('<bert-model-folder>', sequence_length=30)
model = BiGRU_Model(bert_embedding)

train_x, train_y = ChineseDailyNerCorpus.load_data()
model.fit(train_x, train_y)
```

## 性能指标

| 任务         | 语言 | 数据集         | 得分           | 详情                                                                               |
| ------------ | ---- | -------------- | -------------- | ---------------------------------------------------------------------------------- |
| 命名实体识别 | 中文 | 人民日报数据集 | **94.46** (F1) | [Text Labeling Performance Report](./tutorial/text-labeling.md#performance-report) |

## 贡献

如果对 Kashgari 感兴趣，可以通过多种方式加入到该项目。可以通过查阅 [贡献指南](about/contributing.md) 来了解更多。
