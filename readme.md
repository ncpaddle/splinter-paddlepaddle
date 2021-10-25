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

|                | 16 examples F1    | 128 examples F1   | 1024 examples F1  |
| -------------- | ----------------- | ----------------- | ----------------- |
| Ours           | 55.45750403014726 | 71.75794050534279 | 82.44176964790394 |
| Original Paper | 54.6              | 72.7              | 82.8              |

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

## Some Problems and Suggestions

1. paddle.matmul()问题；
2. batch_size=1  nan问题；
3. modal.eval()后不能loss.backward()；
4. fake_data在nlp中很难构造；
5. [MASK]问题；
6. KaiMing初始化；
7. paddlenlp.transformers.BertModel中attention_mask;

More details about **some problems and suggestions** in [readme.md](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/question.md) .

