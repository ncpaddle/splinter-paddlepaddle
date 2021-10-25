import numpy as np
import torch
import paddle
from reprod_log import ReprodLogger, ReprodDiffHelper

#TODO 反向对齐前要修改splinter中的配置文件，把所有的dropout设为0，避免随机影响
#TODO 分别运行finetuning's run.py和finetuning's run_mr.py文件（可以用sh脚本运行），得到两个bin文件后复制到该文件夹

lrt = torch.load('loss_dic_torch.bin')
lrp = paddle.load('loss_dic_paddle.bin')

paddle_list = []
torch_list = []
i = 0
for k, v in lrt.items():
    torch_list.append(v)
    i+=1
for k, v in lrp.items():
    paddle_list.append(v)
print(i)

# 加载ReprodLogger
rl_torch = ReprodLogger()
rl_paddle = ReprodLogger()
rl_torch.add('backward_loss', np.array(torch_list))
rl_paddle.add('backward_loss', np.array(paddle_list))
rl_torch.save('log_reprod/b_loss_torch.npy')
rl_paddle.save('log_reprod/b_loss_paddle.npy')



diff = ReprodDiffHelper()
info_torch = diff.load_info('log_reprod/b_loss_torch.npy')
info_paddle = diff.load_info('log_reprod/b_loss_paddle.npy')
diff.compare_info(info1=info_torch, info2=info_paddle)
diff.report(diff_method='mean', diff_threshold=1e-5, path='log_diff/loss_diff.txt')











