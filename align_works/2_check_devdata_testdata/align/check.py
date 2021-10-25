
import os
import paddle
import torch
from reprod_log import ReprodLogger, ReprodDiffHelper

# 1. finetuning->run.py->get_test_data() ---> obtain: test_data_paddle.npy
# 2. finetuning_pytorch->run_mr.py->get_test_data() ---> obtain:  test_data_torch.npy

# 3. compare
diff = ReprodDiffHelper()
info_torch = diff.load_info('../log_reprod/test_data_torch.npy')
info_paddle = diff.load_info('../log_reprod/test_data_torch.npy')
diff.compare_info(info1=info_torch, info2=info_paddle)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/model_diff.txt')
