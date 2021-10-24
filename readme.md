- 论文名称：[Splinter: Few-Shot Question Answering by Pretraining Span Selection](https://paperswithcode.com/paper/few-shot-question-answering-by-pretraining) <br>数据集：SQuAD1.1 <br>验收标准：SQuAD 1.1验证集，16 examples F1=54.6, 128 examples F1=72.7，1024 Examples F1=82.8（见论文Table1）

# Few-Shot Question Answering by Pretraining Span Selection - Paddle Version

This repoitory contains a **Paddle** implementation of the Splinter.

- Original Paper: https://arxiv.org/pdf/2101.00438v2.pdf
- Original Pytorch Implementation: [oriram/splinter (github.com)](https://github.com/oriram/splinter)

## Introduction

`align_works`: Our all align works about paddlepaddle implementation.  [论文复现指南](https://github.com/PaddlePaddle/models/blob/develop/docs/ThesisReproduction_CV.md#4)

`finetuning`: paddlepaddle version of Splinter

`finetuning_pytorch`: pytorch version of Splinter

`init_splinter/splinter`: model parameters and configs et al .

`mrqa-few-shot`: dataset

## Results

|                | 16 examples F1    | 128 examples F1  | 1024 examples F1  |
| -------------- | ----------------- | ---------------- | ----------------- |
| Ours           | 55.45750403014726 | 68.6470400544094 | 82.44176964790394 |
| Original Paper | 54.6              | 72.7             | 82.8              |



```
128examples, batch_size=13: 70.7981630286274
```

## Align works

[README.md about align works](https://github.com/ncpaddle/splinter-paddlepaddle/tree/main/align_works)





