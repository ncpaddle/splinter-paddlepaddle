- `finetuning`是代码文件夹；`mrqa-few-shot`是数据集文件夹；`splinter`是模型参数及配置；`log_diff`是对齐结果；
- `log_diff`里有两个对齐文件：
  - `pretrain_model_diff.txt`是预训练模型的对齐结果，已经对齐；
  - `model_diff.txt`是论文整个模型的对齐结果，精度不够，只有e-03次方。具体为：在`FullyConnectedLayer`层出了问题。因为`FullyConnectedLayer`使用了四次，前两次给定输入，输出结果是对齐的，后两次给定了另一个输入，输出结果精度降为e-04次方，最终导致精度不断降低，整个模型的输出精度降为了e-03次方。
  - `FullyConnectedLayer`层由一个Linear层、一个GELU操作、一个LayerNorm层组成，目前还未找出精度下降的原因。

- 非常抱歉，由于时间关系，没有完成论文复现工作，只对原论文模型进行了代码的转换以及前向对齐工作，其他工作还在调试运行中。

