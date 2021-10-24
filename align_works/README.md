# 对齐工作说明

## 1. 模型结构对齐

| 预训练模型精度        | 整体模型精度           |
| --------------------- | ---------------------- |
| 3.267558383868163e-07 | 1.0885132942348719e-03 |

> 说明：经二分排查，整体模型精度不够的原因在**paddle.matmul()**，输入的两个矩阵精度都在e-07次方，输出后精度骤降为e-03次方。
>
> ```
> temp = paddle.matmul(query_start_reps, self.start_classifier)  # diff: 3.69778e-07
> start_reps = paddle.transpose(start_reps, perm=[0, 2, 1]) # diff: 2.612227660847566e-07
> start_logits = paddle.matmul(temp, start_reps) # diff: 0.0012255377369001508
> 
> temp = paddle.matmul(query_end_reps, self.end_classifier) # diff: 3.106023598320462e-07
> end_reps = paddle.transpose(end_reps, perm=[0, 2, 1]) # diff: 2.669026457624568e-07
> end_logits = paddle.matmul(temp, end_reps) # diff: 0.0008031659526750445
> ```
>

## 2. 验证/测试集数据读取对齐

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

## 3. 评估指标对齐

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

## 4. 损失函数对齐

我们分别使用fake_data和从模型输出得到的输出数据验证损失函数精度。

| Fake_data            | 模型输出数据          |
| -------------------- | --------------------- |
| 4.76837158203125e-07 | 4.100799560546875e-05 |

> 由于在前向模型结构对齐中paddle.matmul()算子的影响，导致精度不高，进而影响到了该处（模型输出数据送入损失函数）的精度。

## 5. 优化器对齐

见8. 反向对齐

## 6. 学习率对齐

| 学习率对齐精度 |
| -------------- |
| 0.0            |

## 7. 正则化策略对齐

见8. 反向对齐

## 8. 反向对齐

| 反向对齐精度（20轮loss） |
| ------------------------ |
| 2.003300189971924e-03    |

> 和损失对齐一样，受前向模型结构对齐的影响，精度为e-03次方。

## 9. 训练集数据读取对齐

| 训练集数据对齐精度 |
| ------------------ |
| 0.0                |

## 10. 网络初始化对齐

> *对于不同的深度学习框架，网络初始化在大多情况下，即使值的分布完全一致，也无法保证值完全一致，这里也是论文复现中不确定性比较大的地方。如果十分怀疑初始化导致的问题，建议将参考的初始化权重转成paddle模型，加载该初始化模型训练，看下收敛精度。*
>
> 因此我们按照此标准，对模型进行了训练，最终能达到论文的效果，所以是收敛的。

## 11. 模型训练对齐

见主页实验结果
