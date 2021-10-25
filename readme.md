# Splinter-paddle

## 1. Introduction

- This repoitory contains a **paddle** implementation of the Splinter --- \<Few-Shot Question Answering by Pretraining Span Selection\>

- Original Paper: \[2021ACL\][Few-Shot Question Answering by Pretraining Span Selection](https://arxiv.org/pdf/2101.00438v2.pdf)

- Original Pytorch Implementation: [oriram/splinter (github.com)](https://github.com/oriram/splinter)

- Dataset: **SQuAD**



## 2. Experiment Results

|                | 16 examples F1 | 128 examples F1 | 1024 examples F1 |
| -------------- | -------------- | --------------- | ---------------- |
| Ours           | **56.5**       | **72.7**        | **83.1**         |
| Original Paper | 54.6           | 72.7            | 82.8             |



| seed selection of train files | 16 examples F1 | 128 examples F1 | 1024 examples F1 |
| ----------------------------- | -------------- | --------------- | ---------------- |
| 42                            | 51.26509539    | 71.00971741     | 82.26968572      |
| 43                            | 56.73208173    | 72.63511586     | 83.65651393      |
| 44                            | 61.64881717    | 73.67941026     | 83.10696752      |
| 45                            | 52.87432194    | 73.40425954     | 83.32543886      |
| 46                            | 60.14548496    | 73.16006015     | 83.27331091      |
| Avg                           | 56.53316024    | 72.77771264     | 83.12638339      |


## 3. Folders

- `align_works`: Our all align works about paddleimplementation;

- `finetuning`: finetuning codes of Splinter using paddle framework;

- `mrqa-few-shot/squad`: SQuAD dataset;
- `paddlenlp`: changed edition by us based on [paddlepaddle/paddlenlp]([PaddlePaddle/PaddleNLP: An NLP library with Awesome pre-trained Transformer models and easy-to-use interface, supporting wide-range of NLP tasks from research to industrial applications. (github.com)](https://github.com/PaddlePaddle/PaddleNLP));

- [`reprod_log`]([reprod_log/README.md at master · WenmuZhou/reprod_log (github.com)](https://github.com/WenmuZhou/reprod_log/blob/master/README.md)): a third-party library using checking precision between torch's codes and paddle's codes;
- `splinter_init`: model's params and configs.



## 4. Finetuning

> You can git clone this repo to run our codes or use BaiDu AI Studio. 
>
> We suggest you choose BaiDu AI Studio, our program link is:  https://aistudio.baidu.com/aistudio/projectdetail/2503997?shared=1. You can run our codes directly. 
>
> Here's what you need to pay attention to if you choose git clone this repo.

### 4.1 Download Model

You can get model parameters from this [Google Cloud share](https://drive.google.com/drive/folders/1RT9NvOMpmsfIV-q3jXksImV4aqz-gPQN?usp=sharing). 

- splinter ---> align_works/splinter
- splinter_init ---> splinter_init



### 4.2 Run

```shell
$ cd finetuning
$ sh run.sh
```

or

```shell
$ cd finetuning
$ python run.py \
--model_type=bert \
--model_name_or_path="../splinter_init" \
--qass_head=True \
--tokenizer_name="../splinter_init" \
--output_dir="output" \
--train_file="../mrqa-few-shot/squad/squad-train-seed-42-num-examples-16_qass.jsonl" \
--predict_file="../mrqa-few-shot/squad/dev_qass.jsonl" \
--do_train \
--do_eval \
--max_seq_length=384 \
--doc_stride=128 \
--threads=1 \
--save_steps=50000 \
--per_gpu_train_batch_size=12 \
--per_gpu_eval_batch_size=16 \
--learning_rate=3e-5 \
--max_answer_length=10 \
--warmup_ratio=0.1 \
--min_steps=200 \
--num_train_epochs=10 \
--seed=42 \
--use_cache=False \
--evaluate_every_epoch=False \
--initialize_new_qass=False
```



## 5. Align Works

- `forward_diff`: [model_diff.txt](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/align_works/1模型组网验证/log_diff/model_diff.txt)

- `metric_diff`:[metric_diff.txt](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/align_works/3评估指标对齐/log_diff/metric_diff.txt)
- `loss_diff`:[loss_diff.txt](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/align_works/4损失验证2/log_diff/loss_diff.txt)
- `backward_diff`:[backward_diff.txt](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/align_works/5_7_8优化器正则化反向对齐/log_diff/loss_diff.txt)
- `train_align`: experiment results

More details about **align works** in [readme.md](https://github.com/ncpaddle/splinter-paddlepaddle/tree/main/align_works#%E5%AF%B9%E9%BD%90%E5%B7%A5%E4%BD%9C%E8%AF%B4%E6%98%8E) .

## 6. Issues

Issues' details  in [readme.md](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/question.md) .

