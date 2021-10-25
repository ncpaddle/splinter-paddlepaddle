# Few-Shot Question Answering by Pretraining Span Selection - Paddle Version

> This repoitory contains a **Paddle** implementation of the Splinter.
>
> - Original Paper: https://arxiv.org/pdf/2101.00438v2.pdf
> - Original Pytorch Implementation: [oriram/splinter (github.com)](https://github.com/oriram/splinter)
> - Run our codes (BaiDu ai studio): https://aistudio.baidu.com/aistudio/projectdetail/2503997?shared=1

## Introduction

`align_works`: Our all align works about paddlepaddle implementation.

`finetuning`: paddlepaddle version of Splinter.

`finetuning_pytorch`: pytorch version of Splinter.

`init_splinter/splinter`: model parameters and configs files.

`mrqa-few-shot`: SQuAD dataset

## Experiment Results

|                | 16 examples F1 | 128 examples F1 | 1024 examples F1 |
| -------------- | -------------- | --------------- | ---------------- |
| Ours           | **56.5**       | **72.7**        | **83.1**         |
| Original Paper | 54.6           | 72.7            | 82.8             |



| train file seed selection | 16 examples F1 | 128 examples F1 | 1024 examples F1 |
| ------------------------- | -------------- | --------------- | ---------------- |
| 42                        | 51.26509539    | 71.00971741     | 82.26968572      |
| 43                        | 56.73208173    | 72.63511586     | 83.65651393      |
| 44                        | 61.64881717    | 73.67941026     | 83.10696752      |
| 45                        | 52.87432194    | 73.40425954     | 83.32543886      |
| 46                        | 60.14548496    | 73.16006015     | 83.27331091      |
| Avg                       | 56.53316024    | 72.77771264     | 83.12638339      |



## How to run

You can find our programs in BaiDu Ai Studio. 

AI Studios url: https://aistudio.baidu.com/aistudio/projectdetail/2503997?shared=1

edition: splinter-paddle-1024

## Align Works

- `forward_diff`: [model_diff.txt](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/align_works/1模型组网验证/log_diff/model_diff.txt)

- `metric_diff`:[metric_diff.txt](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/align_works/3评估指标对齐/log_diff/metric_diff.txt)
- `loss_diff`:[loss_diff.txt](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/align_works/4损失验证2/log_diff/loss_diff.txt)
- `backward_diff`:[backward_diff.txt](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/align_works/5_7_8优化器正则化反向对齐/log_diff/loss_diff.txt)
- `train_align`: experiment results

More details about **align works** in [readme.md](https://github.com/ncpaddle/splinter-paddlepaddle/tree/main/align_works#%E5%AF%B9%E9%BD%90%E5%B7%A5%E4%BD%9C%E8%AF%B4%E6%98%8E) .

## Some Problems and Suggestion

1. paddle.matmul()；
2. batch_size=1  nan问题；（已提交issue）
3. modal.eval()后不能loss.backward()；
5. [MASK]问题；
6. KaiMing初始化；
7. paddlenlp.transformers.BertModel中attention_mask;（已提交issue）

More details about **some problems and suggestion** in [readme.md](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/question.md) .

