# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""
import argparse
import glob
import json
import logging
import os
import pickle
import random
import timeit
import shutil
import numpy as np
from tqdm import tqdm, trange
from paddlenlp.transformers import BertTokenizer
import paddle
from paddle.io import DataLoader
from paddle.optimizer import AdamW
from squad import squad_convert_examples_to_features
from paddlenlp.transformers import LinearDecayWithWarmup

from squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
    normalize_answer, compute_exact, compute_f1, make_eval_dict, merge_eval)

from squad import SquadResult, SquadV1Processor, SquadV2Processor
from modeling import ModelWithQASSHead
from mrqa_processor import MRQAProcessor


logger = logging.getLogger(__name__)
MODEL_TYPES = ('distilbert', 'albert', 'roberta', 'bert', 'xlnet', 'flaubert', 'xlm')
WEIGHTS_NAME = "model_state.pdparams"

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    paddle.seed(args.seed)


def to_list(tensor):
    return tensor.cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_sampler = paddle.io.RandomSampler(train_dataset) #if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_batch_sampler = paddle.io.BatchSampler(sampler=train_sampler, batch_size=args.train_batch_size)
    # train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)


    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.min_steps > 0 and args.min_steps > t_total:
        t_total = args.min_steps
        args.num_train_epochs = args.min_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        p.name for n, p in model.named_parameters() if not any (nd in n for nd in ['bias', 'LayerNorm.weight'])
    ]

    # clip = paddle.nn.ClipGradByNorm(clip_norm=args.max_grad_norm)
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.max_grad_norm)

    scheduler = LinearDecayWithWarmup(learning_rate=args.learning_rate, warmup=int(args.warmup_ratio * t_total),
                                      total_steps=t_total)

    # adamw parameters不支持字典
    optimizer = AdamW(
        parameters=model.parameters(),
        apply_decay_param_fun=lambda x: x in optimizer_grouped_parameters,
        learning_rate=scheduler,
        epsilon=args.adam_epsilon,
        grad_clip=clip,
        weight_decay=args.weight_decay)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0


    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    # Added here for reproductibility
    set_seed(args)

    best_results = {"exact": 0, "f1": 0, "global_step": 0}

    for _ in train_iterator:
        for step, batch in enumerate(train_dataloader):
            if len(batch[0]) == 1:
                continue
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            batch[3] = paddle.squeeze(batch[3], -1)
            batch[4] = paddle.squeeze(batch[4], -1)
            batch = tuple(t for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(**inputs)

            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.clear_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    # tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info(f"loss step {global_step}: {(tr_loss - logging_loss) / args.logging_steps}")
                    logging_loss = tr_loss

                # Only evaluate when single GPU otherwise metrics may not average well
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    results = evaluate(args, model, tokenizer)
                    # for key, value in results.items():
                    #     tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    logger.info("Results: {}".format(results))
                    if args.output_dir:
                        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                        with open(output_eval_file, "a") as writer:
                            writer.write(f'\n{"step"} {global_step}:\n')
                            for key, values in results.items():
                                if isinstance(values, float):
                                    writer.write(f"{key} = {values:.3f}\n")
                                else:
                                    writer.write(f"{key} = {values}\n")

                # Save model checkpoint 这部分并不会使用
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    paddle.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    paddle.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    paddle.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if args.evaluate_every_epoch:
            results = evaluate(args, model, tokenizer)
            if results["f1"] > best_results["f1"]:
                best_results = results
                best_results["global_step"] = global_step
                logger.info("Results: {}".format(best_results))
    best_results_path = os.path.join(args.output_dir, "best_training_eval_results.json")
    json.dump(best_results, open(best_results_path, 'w'))

    return global_step, tr_loss / global_step


def get_raw_scores_nbest(examples, preds, n):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        predictions = preds[qas_id]
        predictions = [pred["text"] for pred in predictions[:n]]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers for prediction in predictions)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers for prediction in predictions)

    return exact_scores, f1_scores

def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True,
                                                          use_cache=args.use_cache)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    all_nbest = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t for t in batch)
        with paddle.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (paddle.ones(batch[0].shape,dtype=paddle.int64) * args.lang_id).to(args.device)}
                    )

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            None if args.dont_output_nbest else output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    if args.nbest_calculation:
        with open(output_nbest_file, "r") as f:
            nbest_predictions = json.load(f)

        for n in [1, 3, 5, 10]:
            exact_scores, f1_scores = get_raw_scores_nbest(examples, nbest_predictions, n)
            nbest_eval = make_eval_dict(exact_scores, f1_scores)
            merge_eval(results, nbest_eval, f"{n}_best")

    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, use_cache=False):
    # Load data features from cache or dataset file
    cache_name = args.tokenizer_name if args.tokenizer_name else list(filter(None, args.model_name_or_path.split("/"))).pop()
    # print(cache_name)  # ../splinter
    if args.dataset_format == "mrqa":
        cache_name = f"{cache_name}_{args.dataset}"
    else:
        cache_name = cache_name + ("_v2" if args.version_2_with_negative else "_v1")
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            cache_name,
            str(args.max_seq_length),
        ),
    )
    # print(cached_features_file)  # .\cached_train_../splinter_squad_384

    # Init features and dataset from cache if it exists
    if use_cache and os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = paddle.load(cached_features_file)  # paddle加载可能会有问题，但是并不使用
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if args.dataset_format == "mrqa":
            processor = MRQAProcessor()
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
        else:
            # print(args.data_dir, args.train_file)
            # None ../mrqa-few-shot/squad/squad-train-seed-42-num-examples-16_qass.jsonl
            examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        # doc_stride 128    max_query_length 64   max_seq_length 384
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="paddle",
            threads=args.threads,
        )

        if use_cache :
            logger.info("Saving features into cached file %s", cached_features_file)
            paddle.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    # if  not evaluate:
    #     # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    #     torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def getParser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--output_dir_avg", default="output", type=str, required=True)
    parser.add_argument("--qass_head", default=False, type=str2bool, help="Whether to use QASS")
    parser.add_argument("--initialize_new_qass", default=True, type=str2bool, help="Whether to re-init QASS params")

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--evaluate_every_epoch", default=True, type=str2bool, help=""
    )

    # parser.add_argument("--disable_segments_embeddings", default=False, type=str2bool)

    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--min_steps",
        default=-1,
        type=int,
        help="If > max(0, total number of training steps) : set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_ratio", default=0, type=float, help="ratio of warmup during training")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=5000, help="Eval every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", type=str2bool, default=True, help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--use_cache", type=str2bool, default=True, help="If False, don't use cache at all (neither loading nor saving)"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    parser.add_argument("--dataset_format", type=str, choices=["mrqa", "squad"], default="mrqa", help="mrqa or squad")
    parser.add_argument("--dataset", type=str, default="squad", choices=["newsqa", "triviaqa", "searchqa", "hotpotqa",
                                                                         "naturalqa", "squad", "bioasq", "textbookqa"])
    parser.add_argument("--dont_output_nbest", action="store_true")
    parser.add_argument("--nbest_calculation", action="store_true")
    parser.add_argument("--examples_num", default=16, type=int, help="16, 128 or 1024 examples")

    return parser

def main(args):
    assert args.dataset_format in ["mrqa", "squad"]

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if args.overwrite_output_dir and os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    args.n_gpu = 1
    # args.device = torch.device("cuda:1")
    args.device = "gpu:0"
    # args.device = 'cpu'
    paddle.set_device(args.device)

    with open(os.path.join(args.output_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO ,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(False),
        # args.fp16,
    )

    # Set seed
    set_seed(args)

    args.model_type = args.model_type.lower()
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    if args.qass_head:
        model = ModelWithQASSHead.from_pretrained(args.model_name_or_path,
                                                  replace_mask_with_question_token=True,
                                                  mask_id=103, question_token_id=104, initialize_new_qass=args.initialize_new_qass,)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False,
                                                use_cache=args.use_cache)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train :
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        paddle.save(args,path=os.path.join(args.output_dir, "training_args.bin"))
        # Load a trained model and vocabulary that you have fine-tuned
        if args.qass_head:
            model = ModelWithQASSHead.from_pretrained(args.output_dir,
                                                      replace_mask_with_question_token=True,
                                                      mask_id=103, question_token_id=104,
                                                      initialize_new_qass=args.initialize_new_qass,
                                                      cache_dir=args.cache_dir if args.cache_dir else None)
        # else:
        #     model = AutoModelForQuestionAnswering.from_pretrained(args.output_dir, cache_dir=args.cache_dir if args.cache_dir else None)  # , force_download=True)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        # model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""

            if args.qass_head:
                model = ModelWithQASSHead.from_pretrained(checkpoint,
                                                          replace_mask_with_question_token=True,
                                                          mask_id=103, question_token_id=104,
                                                          initialize_new_qass=args.initialize_new_qass,
                                                          cache_dir=args.cache_dir if args.cache_dir else None)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

            if args.output_dir:
                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "a") as writer:
                    writer.write(f'\nFinal Eval:\n')
                    for key, values in results.items():
                        if isinstance(values, float):
                            writer.write(f"{key} = {values:.3f}\n")
                        else:
                            writer.write(f"{key} = {values}\n")
    logger.info("Results: {}".format(results))

    return results



def AvgTraing():
    args = getParser().parse_args()
    examples_nums = [16, 128, 1024]
    seed_ids = [42, 43, 44, 45, 46]
    for e_num in examples_nums:
        result_t = {}
        exact, f1, total = [], [], []
        HasAns_exact, HasAns_f1, HasAns_total = [], [], []
        best_exact, best_exact_thresh, best_f1, best_f1_thresh = [], [], [], []
        for seed_id in seed_ids:
            temp_dir = args.output_dir
            args.output_dir = temp_dir + "/output_seed{}_examples{}".format(seed_id, e_num)
            args.train_file = \
                "splinter-paddle/mrqa-few-shot/squad/squad-train-seed-{}-num-examples-{}_qass.jsonl".format(
                    seed_id, e_num
            )

            results = main(args)

            exact.append(results['exact'])
            f1.append(results['f1'])
            total.append(results['total'])
            HasAns_exact.append(results['HasAns_exact'])
            HasAns_f1.append(results['HasAns_f1'])
            HasAns_total.append(results['HasAns_total'])
            best_exact.append(results['best_exact'])
            best_exact_thresh.append(results['best_exact_thresh'])
            exact.append(results['best_f1'])
            exact.append(results['best_f1_thresh'])
            args.output_dir = temp_dir
        result_t['exact'] = np.mean(np.array(exact))
        result_t['f1'] = np.mean(np.array(f1))
        result_t['total'] = np.mean(np.array(total))
        result_t['HasAns_exact'] = np.mean(np.array(HasAns_exact))
        result_t['HasAns_f1'] = np.mean(np.array(HasAns_f1))
        result_t['HasAns_total'] = np.mean(np.array(HasAns_total))
        result_t['best_exact'] = np.mean(np.array(best_exact))
        result_t['best_exact_thresh'] = np.mean(np.array(best_exact_thresh))
        result_t['best_f1'] = np.mean(np.array(best_f1))
        result_t['best_f1_thresh'] = np.mean(np.array(best_f1_thresh))
        result_t_filepath = args.output_dir_avg + "/examples{}_average.json".format(e_num)
        pickle.dump(result_t, open(result_t_filepath, 'wb'))


if __name__ == "__main__":
    AvgTraing()


