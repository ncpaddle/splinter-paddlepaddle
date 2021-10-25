import json

from finetuning.modeling import ModelWithQASSHead, ClassificationHead
from finetuning.pytorch_modeling import ModelWithQASSHead as PyTorchModelWithQASSHead
import numpy as np
import torch
import pickle
import paddle
import random
from reprod_log import ReprodLogger, ReprodDiffHelper


random.seed(42)
np.random.seed(42)
paddle.seed(42)


# 输入fake_data
start_paddle, end_paddle = paddle.load('../finetuning/loss_input.bin')
start_torch, end_torch = torch.tensor(start_paddle.numpy()), torch.tensor(end_paddle.numpy())



loss_p = paddle.nn.CrossEntropyLoss(ignore_index=384)
loss_t = torch.nn.CrossEntropyLoss(ignore_index=384)


out_p = loss_p(start_paddle, end_paddle)
out_t = loss_t(start_torch, end_torch)

# 加载ReprodLogger
rl_torch = ReprodLogger()
rl_paddle = ReprodLogger()
rl_torch.add('loss', out_t.detach().numpy())
rl_paddle.add('loss', out_p.numpy())

rl_torch.save('../log_reprod/loss_torch.npy')
rl_paddle.save('../log_reprod/loss_paddle.npy')




diff = ReprodDiffHelper()
info_torch = diff.load_info('../log_reprod/loss_torch.npy')
info_paddle = diff.load_info('../log_reprod/loss_paddle.npy')
diff.compare_info(info1=info_torch, info2=info_paddle)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/loss_diff.txt')











