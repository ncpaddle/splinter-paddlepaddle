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


random.seed(42)
np.random.seed(42)
paddle.seed(42)


# 输入fake_data，得到预测时的数据    在pytorch源代码中生成
inputs = pickle.load(open('model_input.bin', 'rb'))
del inputs['start_positions']
del inputs['end_positions']

# 生成paddle和torch的输入
paddle_inputs = {k: paddle.to_tensor(v) for (k, v) in inputs.items()}
torch_inputs = {k: torch.tensor(v) for (k, v) in inputs.items()}


# 加载paddle和torch的模型，得到evaluate的输出
model_paddle = ModelWithQASSHead.from_pretrained('../../splinter')
model_paddle.eval()
out_paddle = model_paddle(**paddle_inputs)

model_torch = PyTorchModelWithQASSHead.from_pretrained("../../splinter")
model_torch.eval()
out_torch = model_torch(**torch_inputs)

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











