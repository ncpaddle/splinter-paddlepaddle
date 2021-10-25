import json

from finetuning.modeling import ModelWithQASSHead, ClassificationHead
from finetuning_pytorch.modeling import ModelWithQASSHead as PyTorchModelWithQASSHead
import numpy as np
import torch
import pickle
import paddle
import random
from reprod_log import ReprodLogger, ReprodDiffHelper
import sys


random.seed(42)
np.random.seed(42)
paddle.seed(42)



# 加载paddle和torch的模型，得到evaluate的输出
model_paddle = ModelWithQASSHead.from_pretrained('../../splinter')
model_paddle.eval()

model_torch = PyTorchModelWithQASSHead.from_pretrained("../../splinter")
model_torch.eval()

'''
cls.span_predictions.start_classifier
[768, 768]
cls.span_predictions.end_classifier
[768, 768]
cls.span_predictions.query_start_transform.dense.weight
[768, 768]
cls.span_predictions.query_start_transform.dense.bias
[768]
cls.span_predictions.query_start_transform.LayerNorm.weight
[768]
cls.span_predictions.query_start_transform.LayerNorm.bias
[768]
cls.span_predictions.query_end_transform.dense.weight
[768, 768]
cls.span_predictions.query_end_transform.dense.bias
[768]
cls.span_predictions.query_end_transform.LayerNorm.weight
[768]
cls.span_predictions.query_end_transform.LayerNorm.bias
[768]
cls.span_predictions.start_transform.dense.weight
[768, 768]
cls.span_predictions.start_transform.dense.bias
[768]
cls.span_predictions.start_transform.LayerNorm.weight
[768]
cls.span_predictions.start_transform.LayerNorm.bias
[768]
cls.span_predictions.end_transform.dense.weight
[768, 768]
cls.span_predictions.end_transform.dense.bias
[768]
cls.span_predictions.end_transform.LayerNorm.weight
[768]
cls.span_predictions.end_transform.LayerNorm.bias
[768]
'''
paddle_params = {}
torch_params = {}

for k,v in model_paddle.named_parameters():
    if 'bert' not in k:
        if 'dense.weight' in k:
            paddle_params[k] = v.numpy()
        else:
            paddle_params[k] = v.numpy()

for k,v in model_torch.named_parameters():
    if 'bert' not in k:
        if 'dense.weight' in k:
            torch_params[k] = v.t().detach().numpy()
        else:
            torch_params[k] = v.detach().numpy()


for k, v in paddle_params.items():
    print(v.tolist()[0])
    print(v.tolist()[1])
    print('------------')
    print(torch_params[k].tolist()[0])
    print(torch_params[k].tolist()[1])
    print(k)
    assert v.tolist() == torch_params[k].tolist()
print('success!')





