from paddlenlp.transformers import BertTokenizer
import paddle


a=paddle.load('eval_res_paddle.bin')
print(a)