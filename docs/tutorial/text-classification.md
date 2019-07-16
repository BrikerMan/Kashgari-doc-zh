# 文本分类

Kashgari 提供了一系列的文本分类模型。所有的文本分类模型都继承自 `BaseClassificationModel` 类，提供了同样的 API。所以切换模型做实验非常的方便。

接口文档请看： [分类模型 API 文档](../api/tasks.classification.md)

## 内置模型列表

| 模型名称              | 模型描述 |
| --------------------- | -------- |
| BiLSTM\_Model         |          |
| BiGRU\_Model          |          |
| CNN\_Model            |          |
| CNN\_LSTM\_Model      |          |
| CNN\_GRU\_Model       |          |
| AVCNN\_Model          |          |
| KMax\_CNN\_Model      |          |
| R\_CNN\_Model         |          |
| AVRNN\_Model          |          |
| Dropout\_BiGRU\_Model |          |
| Dropout\_AVRNN\_Model |          |
| DPCNN\_Model          |          |

## 训练分类模型

Kashgari 内置了一个意图分类数据集用于测试。您也可以使用自己的数据，只需要把数据集格式化为同样的格式即可。

首先加载内置数据集：

```python
from kashgari.corpus import SMP2018ECDTCorpus

train_x, train_y = SMP2018ECDTCorpus.load_data('train')
valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
test_x, test_y = SMP2018ECDTCorpus.load_data('test')
```

使用数据集训练模型。所有的模型都提供同样的接口，所以你可以 `BiLSTM_Model` 模型替换为任何一个内置的分类模型。

```python
import kashgari
from kashgari.tasks.classification import BiLSTM_Model

import logging
logging.basicConfig(level='DEBUG')

model = BiLSTM_Model()
model.fit(train_x, train_y, valid_x, valid_y)

# 验证模型，此方法将打印出详细的验证报告
model.evaluate(test_x, test_y)

# 保存模型到 `saved_ner_model` 目录下
model.save('saved_classification_model')

# 加载保存模型
loaded_model = kashgari.utils.load_model('saved_classification_model')

# 使用模型进行预测
loaded_model.predict(test_x[:10])
```

## 使用预训练语言模型进行迁移学习

Kashgari 内置了几种预训练语言模型处理模块，简化迁移学习流程。下面是一个使用 BERT 的例子。

```python
import kashgari
from kashgari.tasks.classification import BiGRU_Model
from kashgari.embeddings import BERTEmbedding

import logging
logging.basicConfig(level='DEBUG')

bert_embed = BERTEmbedding('<PRE_TRAINED_BERT_MODEL_FOLDER>',
                           task=kashgari.LABELING,
                           sequence_length=100)
model = BiGRU_Model(bert_embed)
model.fit(train_x, train_y, valid_x, valid_y)
```

你还可以把 BERT 替换成 WordEmbedding 或者 GPT2Embedding 等，更多请查阅 [Embedding 文档](../embeddings/index.md)

## 调整模型超参数

通过模型的 `get_default_hyper_parameters()` 方法可以获取默认超参，将会返回一个字典。通过修改字典来修改超参列表。再使用新的超参字典初始化模型。

假设我们想把 `layer_bi_lstm` 层的神经元数量调整为 32：

```python
from kashgari.tasks.classification import BiLSTM_Model

hyper = BiLSTM_Model.get_default_hyper_parameters()
print(hyper)
# {'layer_bi_lstm': {'units': 128, 'return_sequences': False}, 'layer_dense': {'activation': 'softmax'}}

hyper['layer_bi_lstm']['units'] = 32

model = BiLSTM_Model(hyper_parameters=hyper)
```

## 使用训练回调

Kashgari 是基于 tf.keras, 所以你可以直接使用全部的 [tf.keras 回调类](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)，例如我们使用 TensorBoard 可视化训练过程。

```python
from tensorflow.python import keras
from kashgari.tasks.classification import BiGRU_Model
from kashgari.callbacks import EvalCallBack

import logging
logging.basicConfig(level='DEBUG')

model = BiGRU_Model()

tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)

# 这是 Kashgari 内置回调函数，会在训练过程计算精确度，召回率和 F1
eval_callback = EvalCallBack(kash_model=model,
                             valid_x=valid_x,
                             valid_y=valid_y,
                             step=5)

model.fit(train_x,
          train_y,
          valid_x,
          valid_y,
          batch_size=100,
          callbacks=[eval_callback, tf_board_callback])
```

## 多标签分类

Kashgari 支持多分类多标签分类。

假设我们的数据集是这样的：

```python
x = [
   ['This','news',are',very','well','organized'],
   ['What','extremely','usefull','tv','show'],
   ['The','tv','presenter','were','very','well','dress'],
   ['Multi-class', 'classification', 'means', 'a', 'classification', 'task', 'with', 'more', 'than', 'two', 'classes']
]

y = [
   ['A', 'B'],
   ['A',],
   ['B', 'C'],
   []
]
```

现在我们需要初始化一个 `Processor` 和 `Embedding` 对象，然后再初始化我们的模型。

```python
from kashgari.tasks.classification import BiLSTM_Model
from kashgari.processors import ClassificationProcessor
from kashgari.embeddings import BareEmbedding

import logging
logging.basicConfig(level='DEBUG')

# 需要指定我们使用分类数据处理器，且支持多分类
processor = ClassificationProcessor(multi_label=True)
embed = BareEmbedding(processor=processor)

model = BiLSTM_Model(embed)
model.fit(x, y)
```

## 自定义模型结构

除了内置模型以外，还可以很方便的自定义自己的模型结构。只需要继承 `BaseClassificationModel` 对象，然后实现`get_default_hyper_parameters()` 方法
和 `build_model_arc()` 方法。

```python
from typing import Dict, Any
from tensorflow import keras

from kashgari.tasks.classification.base_model import BaseClassificationModel
from kashgari.layers import L

import logging
logging.basicConfig(level='DEBUG')


class DoubleBLSTMModel(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_blstm1': {
                'units': 128,
                'return_sequences': True
            },
            'layer_blstm2': {
                'units': 128,
                'return_sequences': False
            },
            'layer_dropout': {
                'rate': 0.4
            },
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        # 此处作用是从上层拿到输出张量形状和 Embedding 层的输出
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # 定义你自己的层
        layer_blstm1 = L.Bidirectional(L.LSTM(**config['layer_blstm1']),
                                       name='layer_blstm1')
        layer_blstm2 = L.Bidirectional(L.LSTM(**config['layer_blstm2']),
                                       name='layer_blstm2')

        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')

        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim,
                                                           **config['layer_time_distributed']),
                                                   name='layer_time_distributed')
        layer_activation = L.Activation(**config['layer_activation'])

        # 定义数据流
        tensor = layer_blstm1(embed_model.output)
        tensor = layer_blstm2(tensor)
        tensor = layer_dropout(tensor)
        tensor = layer_time_distributed(tensor)
        output_tensor = layer_activation(tensor)

        # 初始化模型
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

# 此模型可以和任何一个 Embedding 组合使用
model = DoubleBLSTMModel()
model.fit(train_x, train_y, valid_x, valid_y)
```
