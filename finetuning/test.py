from paddlenlp.transformers import BertTokenizer


tok = BertTokenizer.from_pretrained('bert-base-cased')

a = tok.encode([['I', 'have', 'a', 'appe'], ['I', 'Chinese']])
print(a)