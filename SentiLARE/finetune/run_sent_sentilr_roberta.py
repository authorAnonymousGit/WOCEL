# Fine-tuning SentiLARE for sentence-level sentiment classification on SST, MR, IMDB, Yelp-2/5.
# The code is modified based on run_glue.py in pytorch-transformers

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import csv

import numpy
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from torch.nn import functional as Func

from pytorch_transformers import (WEIGHTS_NAME, RobertaTokenizer, RobertaConfig, AdamW, WarmupLinearSchedule)

from modeling_sentilr_roberta import RobertaForSequenceClassification, create_prox_mat

from sklearn.metrics import f1_score
from sent_data_utils_sentilr import convert_examples_to_features_roberta

from sent_data_utils_sentilr import sstProcessor, mrProcessor, imdbProcessor, yelp2Processor, yelp5Processor

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

processors = {
    "sst": sstProcessor,
    "imdb": imdbProcessor,
    "mr": mrProcessor,
    "yelp2": yelp2Processor,
    "yelp5": yelp5Processor,
}

# SentiLARE's input embeddings include POS embedding, word-level sentiment polarity embedding,
# and sentence-level sentiment polarity embedding (which is set to be unknown during fine-tuning).
is_pos_embedding = True
is_senti_embedding = True
is_polarity_embedding = True


def set_seed(args):
    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, num_labels):
    # Train the model
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers = 0)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    print(len(train_iterator))
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # RoBERTa doesn't use segment_ids
                      'labels':         batch[-1],
                      'pos_tag_ids':    batch[3],
                      'senti_word_ids': batch[4],
                      'polarity_ids':   batch[5]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # TODO: Update the content within the if, if needed
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, num_labels)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logger.info('learning rate: ' + str(scheduler.get_lr()[0]))
                    logger.info('loss: ' + str((tr_loss - logging_loss)/args.logging_steps))
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def compute_metrics(out_label_ids, preds, num_labels):
    # Calculate accuracy
    acc_result = (preds == out_label_ids).mean()

    # Calculate cem
    dist_dict = {label: 0 for label in range(1, num_labels + 1)}
    for label in out_label_ids:
        dist_dict[label + 1] += 1
    prox_mat = create_prox_mat(dist_dict, inv=False)
    cem_num = sum(prox_mat[preds, out_label_ids])
    cem_den = sum(prox_mat[out_label_ids, out_label_ids])
    cem_result = (cem_num / cem_den).mean()

    # Calculate mae
    mae_result = (np.abs(out_label_ids - preds)).mean()
    return round(float(acc_result), 5), round(float(cem_result), 5), round(float(mae_result), 5)


def evaluate_train(args, model, tokenizer, num_labels, write_to_csv=True, file_name_suffix="", prefix=""):
    # Do validation
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    acc_results = []
    cem_results = []
    mae_results = []

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # Prepare the validation dataset
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers = 0)

        # Eval!
        logger.info("***** Running evaluation on train {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        softmax_values = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # RoBERTa doesn't use segment_ids
                          'pos_tag_ids': batch[3],
                          'senti_word_ids': batch[4],
                          'polarity_ids': batch[5],
                          'labels':         batch[-1]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            logits_shaped = logits.view(-1, num_labels)
            softmax_batch = Func.softmax(logits_shaped, dim=1)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                softmax_values = softmax_batch.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                softmax_values = np.append(softmax_values, softmax_batch.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        acc_result, cem_result, mae_result = compute_metrics(out_label_ids, preds, num_labels)
        acc_results.append(acc_result)
        cem_results.append(cem_result)
        mae_results.append(mae_result)

        # Log the results on the validation set
        output_eval_file = os.path.join(eval_output_dir, "train_results_" + file_name_suffix + ".txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Train results {} *****".format(prefix))
            logger.info("  acc = %s", str(acc_result))
            writer.write("acc = %s\n" % (str(acc_result)))
            logger.info("  cem = %s", str(cem_result))
            writer.write("cem = %s\n" % (str(cem_result)))
            logger.info("  mae = %s", str(mae_result))
            writer.write("mae = %s\n" % (str(mae_result)))

        if write_to_csv:
            probabilities = list(map(lambda y: list(map(lambda x: round(x, 4), y)), softmax_values.tolist()))
            preds_new = [pred + 1 for pred in preds]
            out_label_ids_new = [label + 1 for label in out_label_ids]

            output_eval_file_csv = os.path.join(eval_output_dir, "train_results_" + file_name_suffix + ".csv")
            val_results = [['key_index', 'prediction', 'probability', 'label']] + \
                          list(zip(list(range(len(preds))), preds_new, probabilities, out_label_ids_new))

            with open(output_eval_file_csv, "w+", newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                csv_writer.writerows(val_results)

    return acc_results, cem_results, mae_results


def evaluate(args, model, tokenizer, num_labels, write_to_csv=False, file_name_suffix="", prefix=""):
    # Do validation
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    acc_results = []
    cem_results = []
    mae_results = []

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # Prepare the validation dataset
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers = 0)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        softmax_values = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # RoBERTa doesn't use segment_ids
                          'pos_tag_ids': batch[3],
                          'senti_word_ids': batch[4],
                          'polarity_ids': batch[5],
                          'labels':         batch[-1]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            logits_shaped = logits.view(-1, num_labels)
            softmax_batch = Func.softmax(logits_shaped, dim=1)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                softmax_values = softmax_batch.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                softmax_values = np.append(softmax_values, softmax_batch.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        acc_result, cem_result, mae_result = compute_metrics(out_label_ids, preds, num_labels)
        acc_results.append(acc_result)
        cem_results.append(cem_result)
        mae_results.append(mae_result)

        # Log the results on the validation set
        output_eval_file = os.path.join(eval_output_dir, "eval_results_" + file_name_suffix + ".txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            logger.info("  acc = %s", str(acc_result))
            writer.write("acc = %s\n" % (str(acc_result)))
            logger.info("  cem = %s", str(cem_result))
            writer.write("cem = %s\n" % (str(cem_result)))
            logger.info("  mae = %s", str(mae_result))
            writer.write("mae = %s\n" % (str(mae_result)))

        if write_to_csv:
            probabilities = list(map(lambda y: list(map(lambda x: round(x, 4), y)), softmax_values.tolist()))
            preds_new = [pred+1 for pred in preds]
            out_label_ids_new = [label+1 for label in out_label_ids]

            output_eval_file_csv = os.path.join(eval_output_dir, "val_results_" + file_name_suffix + ".csv")
            val_results = [['key_index', 'prediction', 'probability', 'label']] + \
                           list(zip(list(range(len(preds))), preds_new, probabilities, out_label_ids_new))

            with open(output_eval_file_csv, "w+", newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                csv_writer.writerows(val_results)

    return acc_results, cem_results, mae_results


def test(args, model, tokenizer, num_labels, file_name_suffix="", prefix=""):
    # Do test
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    acc_results = []
    cem_results = []
    mae_results = []

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # Prepare the test set
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=False, test=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

        # Test!
        logger.info("***** Running test {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        softmax_values = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # RoBERTa doesn't use segment_ids
                          'labels':         batch[-1],
                          'pos_tag_ids': batch[3],
                          'senti_word_ids': batch[4],
                          'polarity_ids': batch[5]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            logits_shaped = logits.view(-1, num_labels)
            softmax_batch = Func.softmax(logits_shaped, dim=1)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                softmax_values = softmax_batch.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                softmax_values = np.append(softmax_values, softmax_batch.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        acc_result, cem_result, mae_result = compute_metrics(out_label_ids, preds, num_labels)
        acc_results.append(acc_result)
        cem_results.append(cem_result)
        mae_results.append(mae_result)

        # Log the results on the test set
        output_eval_file = os.path.join(eval_output_dir, "test_results_" + file_name_suffix + ".txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results {} *****".format(prefix))
            logger.info("  acc = %s", str(acc_result))
            writer.write("acc = %s\n" % (str(acc_result)))
            logger.info("  cem = %s", str(cem_result))
            writer.write("cem = %s\n" % (str(cem_result)))
            logger.info("  mae = %s", str(mae_result))
            writer.write("mae = %s\n" % (str(mae_result)))

        probabilities = list(map(lambda y: list(map(lambda x: round(x, 4), y)), softmax_values.tolist()))
        preds_new = [pred + 1 for pred in preds]
        out_label_ids_new = [label + 1 for label in out_label_ids]

        output_eval_file_csv = os.path.join(eval_output_dir, "test_results_" + file_name_suffix + ".csv")
        val_results = [['key_index', 'prediction', 'probability', 'label']] + \
                      list(zip(list(range(len(preds))), preds_new, probabilities, out_label_ids_new))

        with open(output_eval_file_csv, "w+", newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerows(val_results)

    return acc_results, cem_results, mae_results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test = False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Select the data processor according to the task name
    processor = processors[task]()
    output_mode = "classification"
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'test' if test else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_test_examples(args.data_dir) \
            if test else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features_roberta(examples, label_list, args.max_seq_length, tokenizer, output_mode,
             cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
             cls_token=tokenizer.cls_token,
             cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
             sep_token=tokenizer.sep_token,
             sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
             pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
             pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
             pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
             mode = args.task_name)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
    all_senti_ids = torch.tensor([f.senti_ids for f in features], dtype=torch.long)
    all_polarity_ids = torch.tensor([f.polarity_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_pos_ids, all_senti_ids, all_polarity_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # NEW PARAMETERS FOR GoBERT
    parser.add_argument('--loss_type', type=str, default='CrossEntropy', help="Loss Type- CE or OCE")
    parser.add_argument('--alpha', type=float, default=0.6, help="Alpha for OCE loss")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and \
            not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. "
                         "Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    processor = processors[args.task_name]()
    args.output_mode = "classification"
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config, pos_tag_embedding=is_pos_embedding,
                                        senti_embedding=is_senti_embedding,
                                        polarity_embedding=is_polarity_embedding,
                                        loss_type=args.loss_type, alpha=args.alpha)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, num_labels)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Validation
    metrics = ['acc', 'cem', 'mae']
    results_by_metric = {metric: [] for metric in metrics}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir +
                                                                            '/**/' + WEIGHTS_NAME,
                                                                            recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging

        # Do validation on all the checkpoints
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            print(checkpoint)
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, pos_tag_embedding=is_pos_embedding,
                                                senti_embedding=is_senti_embedding,
                                                polarity_embedding=is_polarity_embedding,
                                                loss_type=args.loss_type, alpha=args.alpha)
            model.to(args.device)
            acc_results, cem_results, mae_results = evaluate(args, model, tokenizer, num_labels,
                                                             write_to_csv=False, prefix=global_step)
            results_by_metric['acc'].extend(acc_results)
            results_by_metric['cem'].extend(cem_results)
            results_by_metric['mae'].extend(mae_results)

        # Select the checkpoint with best validation accuracy
        results_summary_rows = [['metric', 'train_acc', 'train_mae', 'train_cem',
                                 'val_acc', 'val_mae', 'val_cem',
                                 'test_acc', 'test_mae', 'test_cem']]
        for metric in metrics:
            best_valid_position = results_by_metric[metric].index(max(results_by_metric[metric]))
            checkpoint_best = checkpoints[best_valid_position]
            global_step = checkpoint_best.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint_best, pos_tag_embedding=is_pos_embedding,
                                                senti_embedding=is_senti_embedding,
                                                polarity_embedding=is_polarity_embedding,
                                                loss_type=args.loss_type, alpha=args.alpha)
            model.to(args.device)
            train_acc, train_cem, train_mae = evaluate_train(args, model, tokenizer, num_labels,
                                                             write_to_csv=True, file_name_suffix=metric, prefix="")
            val_acc, val_cem, val_mae = evaluate(args, model, tokenizer, num_labels, write_to_csv=True,
                                                 file_name_suffix=metric, prefix=global_step)
            test_acc, test_cem, test_mae = test(args, model, tokenizer, num_labels,
                                                file_name_suffix=metric, prefix=global_step)

            results_summary_rows.extend([[metric, train_acc[0], train_mae[0], train_cem[0],
                                          val_acc[0], val_mae[0], val_cem[0],
                                          test_acc[0], test_mae[0], test_cem[0]]])

            print("*" * 20)
            print(f"FINAL RESULTS, SELECT MODEL BY BEST {metric}")
            logger.info(f"Validation Accuracy: {val_acc[0]}, Validation CEM: {val_cem[0]}, Validation MAE: {val_mae[0]}")
            logger.info(f"Test Accuracy: {test_acc[0]}, Test CEM: {test_cem[0]}, Test MAE: {test_mae[0]}")
            print("*"*20)

        with open(args.output_dir + "primary_results_summary.csv", "a+") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerows(results_summary_rows)
        return


if __name__ == "__main__":
    main()
