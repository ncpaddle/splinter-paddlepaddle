- 论文名称：[Splinter: Few-Shot Question Answering by Pretraining Span Selection](https://paperswithcode.com/paper/few-shot-question-answering-by-pretraining) <br>数据集：SQuAD1.1 <br>验收标准：SQuAD 1.1验证集，16 examples F1=54.6, 128 examples F1=72.7，1024 Examples F1=82.8（见论文Table1）



## 当前模型效果

| 16 examples F1 | 128 examples F1 | 1024 examples F1 |
| -------------- | --------------- | ---------------- |
| 56.2           | None            | 82.4             |



## 对齐工作进度

- **模型结构对齐**

  - [x] 网络结构代码转换
  - [x] 权重转换
  - [x] 预训练模型精度对齐
  - [ ] 整体模型精度对齐

  ——见`模型组网验证`文件夹

  | 预训练模型精度        | 整体模型精度           |
  | --------------------- | ---------------------- |
  | 3.267558383868163e-07 | 1.0885132942348719e-03 |

- **数据读取对齐**

  | 数据对齐精度 |
  | ------------ |
  | 0.0          |

  ——见`前向数据对齐`文件夹

- **评估指标对齐**

- **损失函数对齐**

  .....

- **反向对齐**

- **网络参数对齐**



## Question

1. `paddlenlp.data`中的`Vocab`类的`__init__`中，`collections.defaultdict()`中用到了local的匿名函数，而在多线程`with Pool()`中，用`p.imap()`处理数据时，会报错

```
Pool AttributeError: Can't pickle local object 'Vocab.__init__.<locals>.<lambda>......
```

2. BertTokenizer在encode的时候法把`[MASK]`识别成了`[,  MA,  ##S,  ##K,  ]`五个字符。



