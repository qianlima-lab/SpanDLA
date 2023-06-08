import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from data_loader import RefDataset, collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import gensim
import utils
from tensorboard_logger import configure, log_value
import os
import sys
from pathlib2 import Path
import accuracy
import numpy as np
from termcolor import colored
from transformers import AutoTokenizer, AutoModel
from config import args as config_args
from sklearn import metrics
import transformers

torch.multiprocessing.set_sharing_strategy('file_system')

preds_stats = utils.predictions_analysis()


def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums


def import_model(model_name, labels):
    module = __import__('models.' + model_name, fromlist=['models'])
    return module.create(labels)


class Accuracies(object):
    def __init__(self):
        self.thresholds = np.arange(0, 1, 0.05)
        self.accuracies = {k: accuracy.Accuracy() for k in self.thresholds}

    def update(self, output_np, targets_np):
        for k, t in enumerate(targets_np):
            for threshold in self.thresholds:
                h = np.append(output_np, [1])
                tt = np.append(t, [1])

                self.accuracies[threshold].update(h, tt)

    def calc_accuracy(self):
        min_pk = np.inf
        min_threshold = None
        min_epoch_windiff = None
        for threshold in self.thresholds:
            epoch_pk, epoch_windiff = self.accuracies[threshold].calc_accuracy()
            if epoch_pk < min_pk:
                min_pk = epoch_pk
                min_threshold = threshold
                min_epoch_windiff = epoch_windiff

        return min_pk, min_epoch_windiff, min_threshold


def train(model, args, epoch, dataset, logger, optimizer):
    model.train()
    total_sc_loss = float(0)
    total_seg_loss = float(0)
    with tqdm(desc='Training', total=len(dataset)) as pbar:
        for i, (data, adj, target, raw_sentences, paths) in enumerate(dataset):
            if True:
                pbar.update()

                combine_target = torch.cat(target, 0).numpy()
                seg_target, class_target = utils.get_targets(combine_target)
                targets = (maybe_cuda(torch.tensor(seg_target).long()), maybe_cuda(torch.tensor(class_target).long()),
                           maybe_cuda(torch.tensor(combine_target).long()))
                output_seg, loss, de_output_seg, att_score, att_target = model(raw_sentences, targets)
                (loss[0] + loss[1]).backward()
                optimizer.step()
                optimizer.zero_grad()
                total_sc_loss += loss[0].item()
                total_seg_loss += loss[1].item()
                # logger.debug('Batch %s - Train error %7.4f', i, loss.data[0])
                pbar.set_description('Training, sc_loss={:.4} seg_loss={:.4}'.format(loss[0].item(), loss[1].item()))
            # except Exception as e:
            # logger.info('Exception "%s" in batch %s', e, i)
            # logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
            # pass

    total_sc_loss = total_sc_loss / len(dataset)
    total_seg_loss = total_seg_loss / len(dataset)
    logger.debug('Training Epoch: {}, sc_loss={:.4} seg_loss={:.4}.'.format(epoch + 1, total_sc_loss, total_seg_loss))
    log_value('Training sc_loss', total_sc_loss, epoch + 1)
    log_value('Training seg_loss', total_seg_loss, epoch + 1)


def validate(model, args, epoch, dataset, logger):
    model.eval()
    all_tp = []
    all_fn = []
    all_acc = []
    com_gold = []
    com_pred = []
    de_pred = []
    label_tp_dict = None
    label_fn_dict = None
    pk = []
    count = 0
    sum = 0
    total_sc_loss = float(0)
    total_seg_loss = float(0)
    with tqdm(desc='Validatinging', total=len(dataset)) as pbar:
        for i, (data, adj, target, raw_sentences, paths) in enumerate(dataset):
            if True:
                pbar.update()
                combine_target = torch.cat(target, 0).numpy()
                seg_target, class_target = utils.get_targets(combine_target)
                targets = (maybe_cuda(torch.tensor(seg_target).long()), maybe_cuda(torch.tensor(class_target).long()),
                           maybe_cuda(torch.tensor(combine_target).long()))
                output_seg, loss, de_output_seg, att_score, att_target = model(raw_sentences, targets)
                preds_stats.add(output_seg, class_target)
                tp, fn = utils.get_seq_eval(utils.transform2seg(output_seg), combine_target)
                pk.append(utils.calculate_seg_eval((utils.transform2seg(output_seg) > 0).astype(np.int), seg_target))
                count += 1 if fn == 0 else 0
                sum += 1
                all_tp.append(tp)
                all_fn.append(fn)
                total_sc_loss += loss[0].item()
                total_seg_loss += loss[1].item()
                all_acc.append(preds_stats.get_accuracy())
                com_gold.extend(class_target.tolist())
                com_pred.extend(output_seg)
                de_pred.extend(de_output_seg)
                pbar.set_description('Validating, sc_loss={:.4} seg_loss={:.4}'.format(loss[0].item(), loss[1].item()))
            # except Exception as e:
            #     # logger.info('Exception "%s" in batch %s', e, i)
            #     logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
            #     pass
        sentence_f1 = metrics.f1_score(com_gold, com_pred, average="micro")
        sentence_f1_per_label = metrics.f1_score(com_gold, com_pred, average=None)
        label_tp_dict, label_fn_dict = utils.get_seq_eval_per_label(utils.transform2seg(com_gold),
                                                                    utils.transform2seg(com_pred), label_tp_dict,
                                                                    label_fn_dict)
        label_span_f1_dict = utils.get_per_label_f1(label_tp_dict, label_fn_dict)
        unit = np.sum(all_tp) / (np.sum(all_tp) + np.sum(all_fn))
        span_f1 = 0. if unit == 0 else 2 * ((unit * unit) / (unit + unit))
        logger.info(
            'Validating Epoch: {} ,accuracy: {:.4},sentence-f1 {:.4},span-f1 {:.4} right:{:.4} pk:{:.4} '.format(
                epoch + 1,
                np.mean(all_acc),
                float(sentence_f1),
                float(span_f1), count / sum,
                np.mean(pk)))
        logger.info('sentence_f1_per_label {}'.format(sentence_f1_per_label))
        logger.info('span_f1_per_label {}'.format(label_span_f1_dict))
        total_sc_loss = total_sc_loss / len(dataset)
        total_seg_loss = total_seg_loss / len(dataset)
        logger.debug(
            'Validating Epoch: {}, sc_loss={:.4} seg_loss={:.4}.'.format(epoch + 1, total_sc_loss, total_seg_loss))
        preds_stats.reset()

        return sentence_f1, span_f1


def test(model, epoch, dataset, logger):
    model.eval()
    model.eval()
    all_tp = []
    label_tp_dict = None
    label_fn_dict = None
    all_fn = []
    all_acc = []
    com_gold = []
    com_pred = []
    de_pred = []
    pk = []
    count = 0
    sum = 0
    total_sc_loss = float(0)
    total_seg_loss = float(0)
    with tqdm(desc='Testinging', total=len(dataset)) as pbar:
        for i, (data, adj, target, raw_sentences, paths) in enumerate(dataset):
            if True:
                pbar.update()
                combine_target = torch.cat(target, 0).numpy()
                seg_target, class_target = utils.get_targets(combine_target)
                targets = (maybe_cuda(torch.tensor(seg_target).long()), maybe_cuda(torch.tensor(class_target).long()),
                           maybe_cuda(torch.tensor(combine_target).long()))
                output_seg, loss, de_output_seg, att_score, att_target = model(raw_sentences, targets)
                preds_stats.add(output_seg, class_target)
                tp, fn = utils.get_seq_eval(utils.transform2seg(output_seg), combine_target)
                pk.append(utils.calculate_seg_eval((utils.transform2seg(output_seg) > 0).astype(np.int), seg_target))
                count += 1 if fn == 0 else 0
                sum += 1
                all_tp.append(tp)
                all_fn.append(fn)
                total_sc_loss += loss[0].item()
                total_seg_loss += loss[1].item()
                all_acc.append(preds_stats.get_accuracy())
                com_gold.extend(class_target.tolist())
                com_pred.extend(output_seg)
                de_pred.extend(de_output_seg)
                pbar.set_description('Testing, sc_loss={:.4} seg_loss={:.4}'.format(loss[0].item(), loss[1].item()))
            # except Exception as e:
            #     # logger.info('Exception "%s" in batch %s', e, i)
            #     logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
            #     pass
        sentence_f1 = metrics.f1_score(com_gold, com_pred, average="micro")
        sentence_f1_per_label = metrics.f1_score(com_gold, com_pred, average=None)
        label_tp_dict, label_fn_dict = utils.get_seq_eval_per_label(utils.transform2seg(com_gold),
                                                                    utils.transform2seg(com_pred), label_tp_dict,
                                                                    label_fn_dict)
        label_span_f1_dict = utils.get_per_label_f1(label_tp_dict, label_fn_dict)
        unit = np.sum(all_tp) / (np.sum(all_tp) + np.sum(all_fn))
        span_f1 = 0. if unit == 0 else 2 * ((unit * unit) / (unit + unit))
        logger.info(
            'Testing Epoch: {} ,accuracy: {:.4},sentence-f1 {:.4},span-f1 {:.4} right:{:.4} pk:{:.4}'.format(
                epoch + 1,
                np.mean(all_acc),
                float(sentence_f1),
                float(span_f1), count / sum,
                np.mean(pk)))
        total_sc_loss = total_sc_loss / len(dataset)
        total_seg_loss = total_seg_loss / len(dataset)
        logger.info('sentence_f1_per_label {}'.format(sentence_f1_per_label))
        logger.info('span_f1_per_label {}'.format(label_span_f1_dict))
        logger.debug(
            'Testing Epoch: {}, sc_loss={:.4} seg_loss={:.4}.'.format(epoch + 1, total_sc_loss, total_seg_loss))
        preds_stats.reset()

        return sentence_f1, span_f1


def main(args):
    sys.path.append(str(Path(__file__).parent))

    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    logger = utils.setup_logger(__name__, os.path.join(args.checkpoint_dir, 'train.log'))
    configure(os.path.join('runs', args.expname))
    data_path = utils.get_data_path_by_config(config_args.language, config_args.dataset)
    train_dataset = RefDataset(data_path + '/train', True)
    dev_dataset = RefDataset(data_path + '/dev')
    test_dataset = RefDataset(data_path + '/test')

    train_dl = DataLoader(train_dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=True,
                          num_workers=args.num_workers)
    dev_dl = DataLoader(dev_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                        num_workers=args.num_workers)
    test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                         num_workers=args.num_workers)
    global window_size_test, window_size_dev
    window_size_test = utils.get_windows_size(test_dl)
    window_size_dev = utils.get_windows_size(dev_dl)
    model, optimizer = import_model(args.model,
                                    utils.get_labels_num_by_config(config_args.language, config_args.dataset,
                                                                   config_args.type) + 1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    # warmup_steps = int(len(train_dl) * args.epochs * 0.1)
    # num_training_steps = len(train_dl) * args.epochs
    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
    #                                                         num_training_steps=num_training_steps)
    if args.load_from:
        with open(args.load_from, 'rb') as f:
            model = torch.load(f)

    model.train()
    model = maybe_cuda(model)
    best_epoch_1 = 0
    best_epoch_2 = 0
    count_span = 0
    count_sentence = 0
    best_sentence_f1 = 0
    best_span_f1 = 0
    best_test_sentence_f1 = 0
    best_test_span_f1 = 0
    for j in list(range(args.epochs)):
        train(model, args, j, train_dl, logger, optimizer)
        scheduler.step()
        val_sentence_f1, val_span_f1 = validate(model, args, j, dev_dl, logger)
        test_sentence_f1, test_span_f1 = test(model, j, test_dl, logger)
        count_span += 1
        count_sentence += 1
        if test_span_f1 > best_test_span_f1:
            best_test_span_f1 = test_span_f1
            with (checkpoint_path / 'best_test_model_span_f1.t7').open('wb') as f:
                torch.save(model, f)
        if test_sentence_f1 > best_test_sentence_f1:
            best_test_sentence_f1 = test_sentence_f1
            with (checkpoint_path / 'best_test_model_sentence_f1.t7').open('wb') as f:
                torch.save(model, f)
        
        if count_span <= 5 and val_span_f1 >= best_span_f1:
            best_span_f1 = max(val_span_f1, best_span_f1)
            with (checkpoint_path / 'best_model_span_f1.t7').open('wb') as f:
                torch.save(model, f)
            best_epoch_2 = j
            count_span = 0
        if count_sentence <= 5 and val_sentence_f1 >= best_sentence_f1:
            best_sentence_f1 = max(val_sentence_f1, best_sentence_f1)
            with (checkpoint_path / 'best_model_sentence_f1.t7').open('wb') as f:
                torch.save(model, f)
            best_epoch_1 = j
            count_sentence = 0
        if count_span > 5 and count_sentence > 5:
            pass
            # break
    with open(checkpoint_path / 'best_model_sentence_f1.t7', 'rb') as f:
        model_1 = torch.load(f)
    test(model_1, best_epoch_1, test_dl, logger)
    with open(checkpoint_path / 'best_model_span_f1.t7', 'rb') as f:
        model_2 = torch.load(f)
    test(model_2, best_epoch_2, test_dl, logger)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--bs', help='Batch size', type=int, default=8)
    parse.add_argument('--test_bs', help='Batch size', type=int, default=1)
    parse.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parse.add_argument('--model', help='Model to run - will import and run')
    parse.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parse.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parse.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parse.add_argument('--num_workers', type=int, default=0)
    parse = parse.parse_args()
    main(parse)
