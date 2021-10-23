from paddlenlp.transformers import BertTokenizer
import paddle


a=paddle.load('a.bin')
b=paddle.load('b.bin')
print(a)
print('-------')
print(b)