from modeling import ModelWithQASSHead, ClassificationHead
from pytorch_modeling import ModelWithQASSHead as PyTorchModelWithQASSHead
import numpy as np
import pickle
import torch
import paddle
import random
from transformers import BertTokenizer, BertModel
from reprod_log import ReprodLogger, ReprodDiffHelper


# 输入fake_data
f_read = open('input_dict.pkl', 'rb')
inputs = pickle.load(f_read)
del inputs['start_positions']
del inputs['end_positions']
f_read.close()
torch_inputs = inputs
paddle_inputs = {k: paddle.to_tensor(v.detach().numpy().tolist()) for (k, v) in inputs.items()}


# torch_inputs = {
#     'input_ids': torch.tensor([[101, 12050, 1106, 104, 102, 24845, 1105, 24845, 1179, 1233, 1643, 106, 102],
#                                [101, 1106,  1179, 1233,24845, 102, 24845, 1105, 1233, 1233, 1643, 102, 0]]),
#     'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 ],
#                                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0 ]]),
#     'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
# }
# paddle_inputs = {
#     'input_ids': paddle.to_tensor([[101, 12050, 1106, 104, 102, 24845, 1105, 24845, 1179, 1233, 1643, 106, 102],
#                                [101, 1106,  1179, 1233,24845, 102, 24845, 1105, 1233, 1233, 1643, 102, 0]]),
#     'token_type_ids': paddle.to_tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 ],
#                                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0 ]]),
#     'attention_mask': paddle.to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
# }



model_paddle2 = ModelWithQASSHead.from_pretrained('../splinter')
model_paddle2.eval()
out_paddle2 = model_paddle2(**paddle_inputs)
paddle2_array = out_paddle2.numpy()


model_torch2 = PyTorchModelWithQASSHead.from_pretrained("../splinter")
model_torch2.eval()
out_torch2 = model_torch2(**torch_inputs)
torch2_array = out_torch2.cpu().detach().numpy()


diff = abs(paddle2_array - torch2_array)
print('max', np.amax(diff))
print('min', np.amin(diff))
print('mean', np.mean(diff))