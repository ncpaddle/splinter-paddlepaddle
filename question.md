1. 前向对齐时所提到的`paddle.matmul()`导致**精度骤降**问题，看了以前一些论文复现同学提的issues，这个问题不止一次出现过，是paddle的matmul算子实现与pytorch不一样导致的。还有就是`paddle.matmul()`的输入值很大可能也会加重精度降低问题，在我们复现的splinter模型中，值就很大。
2. batch_size=1或送入网络中的数据是一个时，训练会出问题，这个问题在cpu上运行不会出现，只会在gpu上出现，我们已经提交issue，issue地址为：[[飞桨复现论文\]batch=1时出现nan · Issue #36665 · PaddlePaddle/Paddle (github.com)](https://github.com/PaddlePaddle/Paddle/issues/36665)
3. **model.eval()后不能回传loss**，也就是不能loss.backward()，导致大家在反向精度对齐时很难处理，因为model.train()会引入很多随机的东西，尤其是对于我们所复现的预训练语言模型+微调范式的论文来说，transformers里面的12层dropout是灾难。不过我们把预训练模型加载的config.json中的dropout率都改为了0，解决了随机的问题。
4. 论文复现指南上所说的**fake_data**，在nlp中很难构造，甚至在我们所复现的论文中，根本无法构造。因此我们的fake_data是抽出了几个batch的数据。
5. **paddlenlp.BertTokenizer**好像没有办法处理`[MASK]`, 下面是`BertTokenizer.__init__`中的三行代码，`self.basic_tokenizer`会把`[MASK]`切成`[`,`MASK`,`]`三部分，`self.wordpiece_tokenizer`会进一步切成`[`,`MA`, `##S`, `##K`,`]`五部分，即使`self.vocab`中有`[MASK]`.

```
self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
self.wordpiece_tokenizer = WordpieceTokenizer(
vocab=self.vocab, unk_token=unk_token)
```

6. paddle的KaiMing初始化不如pytorch的KaiMing初始化适用性强，paddle的默认支持ReLU，而torch的则支持Leaky ReLU在内的许多非线性操作。

7. paddlenlp.transformers的文档中，Bert Model中，个人感觉对于forward中的**参数attention_mask**的注释不够清晰。已提交issue:[paddlenlp.transformers.BertModel中的attention_mask · Issue #1224 · PaddlePaddle/PaddleNLP (github.com)](https://github.com/PaddlePaddle/PaddleNLP/issues/1224)

   

   *It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`. Defaults to `None`, which means nothing needed to be prevented attention to.*看完这句话以为输入维度要 `[batch_size, num_attention_heads, sequence_length, sequence_length]`才行。

   

   然而，看了源码发现如果不给出`attention_mask`，那么模型初始化的`attention_mask`维度是`[batch_size, 1, 1, sequence_length]`，如下面代码：

   ```
   if attention_mask is None:
       attention_mask = paddle.unsqueeze(
           (input_ids == self.pad_token_id
            ).astype(self.pooler.dense.weight.dtype) * -1e9,
           axis=[1, 2])
   ```

   经过自己测试发现，输入的attention_mask维度为`[batch_size, 1, 1, sequence_length]`，没有问题。