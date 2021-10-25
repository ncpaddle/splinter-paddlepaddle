from paddlenlp.transformers import BertTokenizer
import paddle
import torch


slp = paddle.load('start_logits_paddle.bin')
slt = torch.load('start_logits_torch.bin', map_location=torch.device('cpu'))


print(slp)
print(slt)