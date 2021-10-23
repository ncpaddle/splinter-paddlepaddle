import numpy as np
import torch
import paddle
from reprod_log import ReprodLogger, ReprodDiffHelper

lrt = torch.load('lr_torch.bin')
lrp = paddle.load('lr_paddle.bin')


# 加载ReprodLogger
rl_torch = ReprodLogger()
rl_paddle = ReprodLogger()
rl_torch.add('loss', np.array(lrt))
rl_paddle.add('loss', np.array(lrp))
rl_torch.save('../log_reprod/lr_torch.npy')
rl_paddle.save('../log_reprod/lr_paddle.npy')



diff = ReprodDiffHelper()
info_torch = diff.load_info('../log_reprod/lr_torch.npy')
info_paddle = diff.load_info('../log_reprod/lr_paddle.npy')
diff.compare_info(info1=info_torch, info2=info_paddle)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/loss_diff.txt')











