import sys
sys.path.append('/mnt/sda/zhouchangzhi/splinter-paddle')
import json

from finetuning.modeling import ModelWithQASSHead, ClassificationHead
from finetuning_pytorch.modeling import ModelWithQASSHead as PyTorchModelWithQASSHead
import numpy as np
import torch
import pickle
import paddle
import random
from reprod_log import ReprodLogger, ReprodDiffHelper





lt = torch.nn.Linear(20, 3)
lp = paddle.nn.Linear(3 ,20, weight_attr=paddle.nn.initializer.KaimingUniform())


for p in lt.parameters():
    print(p)


print(lp.parameters())
assert 1 == 2



random.seed(42)
np.random.seed(42)
paddle.seed(42)


# 加载paddle和torch的模型，得到evaluate的输出
model_paddle = ModelWithQASSHead.from_pretrained('../../splinter', initialize_new_qass=True)
model_paddle.train()
out_paddle = model_paddle.named_parameters()

model_torch = PyTorchModelWithQASSHead.from_pretrained("../../splinter",  initialize_new_qass=True)
model_torch.train()
out_torch = model_torch.named_parameters()


for k, v in out_paddle:
    print(k, v.shape)

assert 1 == 2

paddle_np = np.array([out_paddle[0].numpy(), out_paddle[1].numpy()])
torch_np = np.array([out_torch[0].detach().numpy(), out_torch[1].detach().numpy()])
# paddle_np = np.array([out_paddle.numpy()])
# torch_np = np.array([out_torch.detach().numpy()])

# 加载ReprodLogger
rl_torch = ReprodLogger()
rl_paddle = ReprodLogger()
rl_torch.add('sequence_output', torch_np)
rl_paddle.add('sequence_output', paddle_np)
rl_torch.save('../log_reprod/model_output_torch.npy')
rl_paddle.save('../log_reprod/model_output_paddle.npy')




diff = ReprodDiffHelper()
info_torch = diff.load_info('../log_reprod/model_output_torch.npy')
info_paddle = diff.load_info('../log_reprod/model_output_paddle.npy')
diff.compare_info(info1=info_torch, info2=info_paddle)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/model_diff.txt')











