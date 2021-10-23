'''
Training/evaluation parameters
Namespace(adam_epsilon=1e-08,
          cache_dir='',
          config_name='',
          data_dir=None,
          dataset='squad',
          dataset_format='mrqa',
          device=device(type='cpu'),
          disable_segments_embeddings=False,
          do_eval=True,
          do_lower_case=False,
          do_train=True,
          doc_stride=128,
          dont_output_nbest=False,
          eval_all_checkpoints=False,
          eval_steps=5000,
          evaluate_during_training=False,
          evaluate_every_epoch=True,
          fp16=False,
          fp16_opt_level='O1',
          gradient_accumulation_steps=1,
          initialize_new_qass=False,
          lang_id=0,
          learning_rate=3e-05,
          local_rank=-1,
          logging_steps=500,
          max_answer_length=10,
          max_grad_norm=1.0,
          max_query_length=64,
          max_seq_length=384,
          max_steps=-1,
          min_steps=200,
          model_name_or_path='../splinter',
          model_type='bert',
          n_best_size=20,
          n_gpu=0,
          nbest_calculation=False,
          no_cuda=False,
          null_score_diff_threshold=0.0,
          num_train_epochs=10.0,
          output_dir='output',
          overwrite_cache=False,
          overwrite_output_dir=True,
          per_gpu_eval_batch_size=16,
          per_gpu_train_batch_size=12,
          predict_file='../mrqa-few-shot/squad/dev_qass.jsonl',
          qass_head=True,
          save_steps=50000,
          seed=42, server_ip='', server_port='', threads=1,
          tokenizer_name='../splinter',
          train_file='../mrqa-few-shot/squad/squad-train-seed-42-num-examples-16_qass.jsonl',
          use_cache=False, verbose_logging=False, version_2_with_negative=False,
          warmup_ratio=0.1, weight_decay=0.0)













Training/evaluation parameters Namespace(adam_epsilon=1e-08, cache_dir='', config_name='', data_dir=None, dataset='squad', dataset_format='mrqa', device='cpu', do_eval=True, do_lower_case=False, do_train=True, doc_stride=128, dont_output_nbest=False, eval_all_checkpoints=False, eval_steps=5000, evaluate_during_training=False, evaluate_every_epoch=True, fp16=False, fp16_opt_level='O1', gradient_accumulation_steps=1, initialize_new_qass=False, lang_id=0, learning_rate=3e-05, local_rank=-1, logging_steps=500, max_answer_length=10, max_grad_norm=1.0, max_query_length=64, max_seq_length=384, max_steps=-1, min_steps=200, model_name_or_path='../splinter', model_type='bert', n_best_size=20, n_gpu=1, nbest_calculation=False, no_cuda=False, null_score_diff_threshold=0.0, num_train_epochs=10.0, output_dir='output', overwrite_cache=False, overwrite_output_dir=True, per_gpu_eval_batch_size=16, per_gpu_train_batch_size=12, predict_file='../mrqa-few-shot/squad/dev_qass.jsonl', qass_head=True, save_steps=50000, seed=42, server_ip='', server_port='', threads=1, tokenizer_name='../splinter', train_file='../mrqa-few-shot/squad/squad-train-seed-42-num-examples-16_qass.jsonl', use_cache=False, verbose_logging=False, version_2_with_negative=False, warmup_ratio=0.1, weight_decay=0.01)
'''