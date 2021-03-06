# Splinter-paddle

## 1. Introduction

- This repoitory contains a **paddle** implementation of the Splinter, which is a new question answering pretraining scheme based on few-shot learning. It comes from the 2021 ACL paper *Few-Shot Question Answering by Pretraining Span Selection* and uses **recurring span selection** to enhance the performance. 
- **Paper**:  "[Few-Shot Question Answering by Pretraining Span Selection](https://arxiv.org/pdf/2101.00438v2.pdf)", to appear at ACL 2021.
- **Original Pytorch Implementation**: [oriram/splinter (github.com)](https://github.com/oriram/splinter)
- Dataset: **SQuAD**



## 2. Experiment Results

We experimented with three few-shot scenarios (16, 128, 1024 examples) using classical dataset **SQuAD**, which is a representative part of the experiment in the paper. 

|                          | 16 examples F1 | 128 examples F1 | 1024 examples F1 |
| ------------------------ | -------------- | --------------- | ---------------- |
| Original Paper's Results | 54.6           | **72.7**        | 82.8             |
| Ours Results             | **55.62**      | 72.63           | **82.89**        |

- Ours More Detailed Results:

| seed selection of train files | 16 examples F1 | 128 examples F1 | 1024 examples F1 |
| ----------------------------- | -------------- | --------------- | ---------------- |
| 42                            | 54.38          | 71.45           | 82.46            |
| 43                            | 51.89          | 72.42           | 82.82            |
| 44                            | 61.68          | 73.81           | 83.22            |
| 45                            | 45.14          | 72.43           | 83.39            |
| 46                            | 58.92          | 73.06           | 82.57            |
| **Average**                   | **55.62**      | **72.63**       | **82.89**        |

We can find the instability of few-shot learning. The more samples in the training set, the more stable the results will be, which is consistent with our intuition. 



## 3. Code Structure

- `align_works`: Our all align works about paddleimplementation;
  - `1_check_forward`: the align works about the forward training;
  - `2_check_devdata_testdata`: the align works about dev dataset and test dataset;
  - `3_check_metirc`: the align works about metric;
  - ......
  - `10_check_network_params_init`: the align works about model's parameters init;
- `finetuning`: finetuning codes of Splinter using paddle framework;
- `mrqa-few-shot/squad`: SQuAD dataset;
- `output_avg`: the experiment results;
- `paddlenlp`: changed edition by us based on [paddlepaddle/paddlenlp](https://github.com/PaddlePaddle/PaddleNLP);
- [`reprod_log`](https://github.com/WenmuZhou/reprod_log/blob/master/README.md): a third-party library using checking precision between torch's codes and paddle's codes;
- `splinter_init`: model's params and configs.



## 4. Environment

- Hardward: GPU, CPU
- Framework:
  - PaddlePaddle >= 2.0.0



## 5. Quick start

### 5.1 Run Style

1. Git clone this repo and download model parameters from  [Google Cloud share](https://drive.google.com/drive/folders/1RT9NvOMpmsfIV-q3jXksImV4aqz-gPQN?usp=sharing)
   1. splinter ---> align_works/splinter
   2. splinter_init ---> splinter_init
2. Runing our codes in **BaiDu AI Studio**. Choosing **`Splinter-paddle`** edition from this [link](https://aistudio.baidu.com/aistudio/projectdetail/2503997?shared=1) and runing the program. 

> *We suggest you choose the AI Studio.*



### 5.2 Run Scripts

- We can obtain the average experiment results by the script that can run all of the sampled datasets. 

```shell
python splinter-paddle/finetuning/run_all.py \
    --model_type=bert \
    --model_name_or_path="splinter-paddle/splinter_init" \
    --qass_head=True \
    --tokenizer_name="splinter-paddle/splinter_init" \
    --output_dir="output" \
    --output_dir_avg="output_avg" \
    --train_file="" \
    --predict_file="splinter-paddle/mrqa-few-shot/squad/dev_qass.jsonl" \
    --do_train \
    --do_eval \
    --max_seq_length=384 \
    --doc_stride=128 \
    --threads=4 \
    --save_steps=50000 \
    --per_gpu_train_batch_size=12 \
    --per_gpu_eval_batch_size=16 \
    --learning_rate=3e-5 \
    --max_answer_length=10 \
    --warmup_ratio=0.1 \
    --min_steps=200 \
    --num_train_epochs=10 \
    --seed=128 \
    --use_cache=False \
    --evaluate_every_epoch=False \
    --initialize_new_qass=False
```

- We can obtain a experiment result of a single sampled dataset by this script. 

```shell
python splinter-paddle/finetuning/run.py \
    --model_type=bert \
    --model_name_or_path="splinter-paddle/splinter_init" \
    --qass_head=True \
    --tokenizer_name="splinter-paddle/splinter_init" \
    --output_dir="output_single" \
    --train_file="splinter-paddle/mrqa-few-shot/squad/squad-train-seed-42-num-examples-16_qass.jsonl" \
    --predict_file="splinter-paddle/mrqa-few-shot/squad/dev_qass.jsonl" \
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
    --seed=128 \
    --use_cache=False \
    --evaluate_every_epoch=False \
    --initialize_new_qass=False
```



## 6. Align Works

- `forward_diff`: [model_diff.txt](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/align_works/1_check_forward/log_diff/model_diff.txt)

- `metric_diff`:[metric_diff.txt](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/align_works/3_check_metric/log_diff/metric_diff.txt)
- `loss_diff`:[loss_diff.txt](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/align_works/4_check_loss/log_diff/loss_diff.txt)
- `backward_diff`:[backward_diff.txt](https://github.com/ncpaddle/splinter-paddlepaddle/blob/main/align_works/5-7-8_check_optim-norm-backward/log_diff/loss_diff.txt)
- `train_align`: experiment results

More details about **align works** in [here](https://github.com/ncpaddle/splinter-paddlepaddle/tree/main/align_works#%E5%AF%B9%E9%BD%90%E5%B7%A5%E4%BD%9C%E8%AF%B4%E6%98%8E) .

