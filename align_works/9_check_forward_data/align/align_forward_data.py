import json

import numpy as np
import torch
import pickle
import paddle
import random
from reprod_log import ReprodLogger, ReprodDiffHelper

# 在不加RandomSampler的情况下保证数据读取一致

# 运行paddle run.py和pytorch run.py得到一个batch_size的数据，保存
# 加载已保存的数据
input_data_paddle = pickle.load(open('input_data_paddle.bin', 'rb'))
input_data_torch = pickle.load(open('input_data_torch.bin', 'rb'))


paddle_input_ids = np.array(input_data_paddle['input_ids'])
torch_input_ids = np.array(input_data_torch['input_ids'])

# 加载ReprodLogger
rl_torch = ReprodLogger()
rl_paddle = ReprodLogger()
rl_torch.add('sequence_output', np.array(torch_input_ids))
rl_paddle.add('sequence_output', np.array(paddle_input_ids))
rl_torch.save('../log_reprod/forward_data_torch.npy')
rl_paddle.save('../log_reprod/forward_data_paddle.npy')



diff = ReprodDiffHelper()
info_torch = diff.load_info('../log_reprod/forward_data_torch.npy')
info_paddle = diff.load_info('../log_reprod/forward_data_paddle.npy')
diff.compare_info(info1=info_torch, info2=info_paddle)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/forward_data_diff.txt')

