- 论文名称：[Splinter: Few-Shot Question Answering by Pretraining Span Selection](https://paperswithcode.com/paper/few-shot-question-answering-by-pretraining) <br>数据集：SQuAD1.1 <br>验收标准：SQuAD 1.1验证集，16 examples F1=54.6, 128 examples F1=72.7，1024 Examples F1=82.8（见论文Table1）



## 当前模型效果

| 16 examples F1 | 128 examples F1 | 1024 examples F1 |
| -------------- | --------------- | ---------------- |
|                |                 |                  |



# 对齐工作进度

## 模型结构对齐

——见`模型组网验证`文件夹

### 权重转换

作者pytorch模型所有参数都保存在`pytorch_model.bin`文件中，读取该文件，将模型结构与paddle的模型结构名称一一对应，转换权重。

| 模型参数精度 |
| ------------ |
| 0.0          |

### 模型组网正确性验证

将`fake_data`送入pytorch模型和paddle模型，整体模型的中间输出（即预训练模型输出）精度为e-07级，但最终输出精度为e-03级，经二分查找，定位在`nn.Linear`，经该线性层处理后精度下降。该线性层使用多次，导致最终精度为e-03级。

```
sequence_output = outputs[0]  # [batch_size, max_length, dim]
# sequence_output精度e-07
cls = self.get_cls()
start_logits, end_logits = cls(sequence_output, masked_positions)
return start_logits, end_logits
		这两个精度e-03
		

		
```



***精度骤降***：

环境：个人电脑paddlepaddle-2.1.3-cpu

`QuestionAwareSpanSelectionHead`类中，forward时`start_logits = paddle.matmul(temp, start_reps)`，`paddle.matmul`**操作导致精度骤降，输入temp和start_reps精度都是e-07，输出matmul之后精度变为e-03**

将`fake_data`更换，整体模型精度仍为e-03

| 预训练模型精度        | 整体模型精度           |
| --------------------- | ---------------------- |
| 3.267558383868163e-07 | 1.0885132942348719e-03 |

## 验证/测试集数据读取对齐

```
[2021/10/22 22:36:47] root INFO: input_ids_list: 
[2021/10/22 22:36:47] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/22 22:36:47] root INFO: attention_mask_list: 
[2021/10/22 22:36:47] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/22 22:36:47] root INFO: token_type_ids_list: 
[2021/10/22 22:36:47] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/22 22:36:47] root INFO: start_pos_list: 
[2021/10/22 22:36:47] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/22 22:36:47] root INFO: end_pos_list: 
[2021/10/22 22:36:47] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/22 22:36:47] root INFO: input_ids_list2: 
[2021/10/22 22:36:47] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/22 22:36:47] root INFO: attention_mask_list2: 
[2021/10/22 22:36:47] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/22 22:36:47] root INFO: token_type_ids_list2: 
[2021/10/22 22:36:47] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/22 22:36:47] root INFO: start_pos_list2: 
[2021/10/22 22:36:47] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/22 22:36:47] root INFO: end_pos_list2: 
[2021/10/22 22:36:47] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/22 22:36:47] root INFO: diff check passed
```

## 评估指标对齐

```
[2021/10/23 20:55:33] root INFO: exact: 
[2021/10/23 20:55:33] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/23 20:55:33] root INFO: f1: 
[2021/10/23 20:55:33] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/23 20:55:33] root INFO: total: 
[2021/10/23 20:55:33] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/23 20:55:33] root INFO: HasAns_exact: 
[2021/10/23 20:55:33] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/23 20:55:33] root INFO: HasAns_f1: 
[2021/10/23 20:55:33] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/23 20:55:33] root INFO: HasAns_total: 
[2021/10/23 20:55:33] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/23 20:55:33] root INFO: best_exact: 
[2021/10/23 20:55:33] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/23 20:55:33] root INFO: best_exact_thresh: 
[2021/10/23 20:55:33] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/23 20:55:33] root INFO: best_f1: 
[2021/10/23 20:55:33] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/23 20:55:33] root INFO: best_f1_thresh: 
[2021/10/23 20:55:33] root INFO:   mean diff: check passed: True, value: 0.0
[2021/10/23 20:55:33] root INFO: diff check passed
```

## 损失函数对齐

### Fake_data

构造`fake_data`分别送入两个损失函数，计算精度。

### 真实数据

将模型的输出送入两个损失函数，计算精度。由于在模型对齐部分精度只有e-03，即送入两个损失函数的数据精度是e-03，所以导致损失函数对齐精度较`fake_data`低。

| Fake_data            | 真实数据              |
| -------------------- | --------------------- |
| 4.76837158203125e-07 | 4.100799560546875e-05 |

## 学习率对齐

| 学习率对齐精度 |
| -------------- |
| 0.0            |



## 优化器/正则化策略/反向对齐

...

## 训练集数据读取对齐

| 训练集数据对齐精度 |
| ------------------ |
| 0.0                |





## 网络初始化对齐

。。。



## 模型训练对齐





## Question

1. `paddlenlp.data`中的`Vocab`类的`__init__`中，`collections.defaultdict()`中用到了local的匿名函数，而在多线程`with Pool()`中，用`p.imap()`处理数据时，会报错

```
Pool AttributeError: Can't pickle local object 'Vocab.__init__.<locals>.<lambda>......
```

2. BertTokenizer在encode的时候法把`[MASK]`识别成了`[,  MA,  ##S,  ##K,  ]`五个字符。



