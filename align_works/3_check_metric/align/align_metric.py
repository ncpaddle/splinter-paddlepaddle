import numpy as np
import paddle
import torch
from reprod_log import ReprodLogger, ReprodDiffHelper


# 将50条数据作为输入，执行evaluate函数得到下列两个文件
eval_p = paddle.load('eval_res_paddle_50.bin')
eval_t = torch.load('eval_res_torch_50.bin')

# exact f1 total HasAns_exact HasAns_f1 HasAns_total best_exact best_exact_thresh best_f1 best_f1_thresh

# 加载ReprodLogger
rl_torch = ReprodLogger()
rl_torch.add('exact', np.array(eval_t['exact']))
rl_torch.add('f1', np.array(eval_t['f1']))
rl_torch.add('total', np.array(eval_t['total']))
rl_torch.add('HasAns_exact', np.array(eval_t['HasAns_exact']))
rl_torch.add('HasAns_f1', np.array(eval_t['HasAns_f1']))
rl_torch.add('HasAns_total', np.array(eval_t['HasAns_total']))
rl_torch.add('best_exact', np.array(eval_t['best_exact']))
rl_torch.add('best_exact_thresh', np.array(eval_t['best_exact_thresh']))
rl_torch.add('best_f1', np.array(eval_t['best_f1']))
rl_torch.add('best_f1_thresh', np.array(eval_t['best_f1_thresh']))

rl_paddle = ReprodLogger()
rl_paddle.add('exact', np.array(eval_p['exact']))
rl_paddle.add('f1', np.array(eval_p['f1']))
rl_paddle.add('total', np.array(eval_p['total']))
rl_paddle.add('HasAns_exact', np.array(eval_p['HasAns_exact']))
rl_paddle.add('HasAns_f1', np.array(eval_p['HasAns_f1']))
rl_paddle.add('HasAns_total', np.array(eval_p['HasAns_total']))
rl_paddle.add('best_exact', np.array(eval_p['best_exact']))
rl_paddle.add('best_exact_thresh', np.array(eval_p['best_exact_thresh']))
rl_paddle.add('best_f1', np.array(eval_p['best_f1']))
rl_paddle.add('best_f1_thresh', np.array(eval_p['best_f1_thresh']))

# save
rl_torch.save('../log_reprod/metric_torch.npy')
rl_paddle.save('../log_reprod/metric_paddle.npy')


diff = ReprodDiffHelper()
info_torch = diff.load_info('../log_reprod/metric_torch.npy')
info_paddle = diff.load_info('../log_reprod/metric_paddle.npy')
diff.compare_info(info1=info_torch, info2=info_paddle)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/metric_diff.txt')
