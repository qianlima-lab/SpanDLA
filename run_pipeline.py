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


def train(model, args, epoch, dataset, logger, optimizer, is_seg=True):
    model.train()
    total_loss = float(0)
    with tqdm(desc='Training', total=len(dataset)) as pbar:
        for i, (data, adj, target, raw_sentences, paths) in enumerate(dataset):
            if True:
                pbar.update()

                combine_target = torch.cat(target, 0).numpy()
                seg_target, class_target = utils.get_targets(combine_target)
                targets = (maybe_cuda(torch.tensor(seg_target).long()), maybe_cuda(torch.tensor(class_target).long()),
                           maybe_cuda(torch.tensor(combine_target).long()))
                output_seg, output_sc, loss_seg, loss_sc = model(raw_sentences, targets)
                if is_seg:
                    loss_seg.backward()
                    loss = loss_seg
                else:
                    loss_sc.backward()
                    loss = loss_sc
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                # logger.debug('Batch %s - Train error %7.4f', i, loss.data[0])
                pbar.set_description('Training, loss={:.4}'.format(loss.item()))
            # except Exception as e:
            # logger.info('Exception "%s" in batch %s', e, i)
            # logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
            # pass

    total_loss = total_loss / len(dataset)
    logger.debug('Training Epoch: {}, Loss: {:.4}.'.format(epoch + 1, total_loss))
    log_value('Training Loss', total_loss, epoch + 1)


def validate(model, args, epoch, dataset, logger):
    model.eval()
    all_tp = []
    all_fn = []
    all_seg_tp = []
    all_seg_fn = []
    all_acc = []
    com_gold = []
    com_pred = []
    with tqdm(desc='Validatinging', total=len(dataset)) as pbar:
        for i, (data, adj, target, raw_sentences, paths) in enumerate(dataset):
            if True:
                pbar.update()
                combine_target = torch.cat(target, 0).numpy()
                seg_target, class_target = utils.get_targets(combine_target)
                targets = (maybe_cuda(torch.tensor(seg_target).long()), maybe_cuda(torch.tensor(class_target).long()),
                           maybe_cuda(torch.tensor(combine_target).long()))
                output_seg, output_sc, loss_seg, loss_sc = model(raw_sentences, targets)
                preds_stats.add(output_sc, class_target)
                tp, fn = utils.get_seq_eval(utils.transform2seg(output_sc), combine_target)
                seg_tp, seg_fn = utils.get_seq_eval(output_seg, seg_target)
                all_tp.append(tp)
                all_fn.append(fn)
                all_seg_tp.append(seg_tp)
                all_seg_fn.append(seg_fn)
                all_acc.append(preds_stats.get_accuracy())
                com_gold.extend(class_target.tolist())
                com_pred.extend(output_sc)
                pbar.set_description('Validatinging, loss={:.4}'.format(loss_sc.item()))
            # except Exception as e:
            #     # logger.info('Exception "%s" in batch %s', e, i)
            #     logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
            #     pass
        sentence_f1 = metrics.f1_score(com_gold, com_pred, average="micro")
        unit = np.sum(all_tp) / (np.sum(all_tp) + np.sum(all_fn))
        span_f1 = 0. if unit == 0 else 2 * ((unit * unit) / (unit + unit))
        seg_unit = np.sum(all_seg_tp) / (np.sum(all_seg_tp) + np.sum(all_seg_fn))
        span_seg_f1 = 0. if seg_unit == 0 else 2 * ((seg_unit * seg_unit) / (seg_unit + seg_unit))
        logger.info(
            'Validating Epoch: {} ,accuracy: {:.4},sentence-f1 {:.4},span-f1 {:.4},seg-span-f1{:.4} '.format(
                epoch + 1,
                np.mean(all_acc),
                float(sentence_f1),
                float(span_f1),
                float(span_seg_f1)))

        preds_stats.reset()

        return sentence_f1, span_f1, span_seg_f1


def test(model, epoch, dataset, logger):
    model.eval()
    all_tp = []
    all_fn = []
    all_seg_tp = []
    all_seg_fn = []
    all_acc = []
    com_gold = []
    com_pred = []
    with tqdm(desc='Testing', total=len(dataset)) as pbar:
        for i, (data, adj, target, raw_sentences, paths) in enumerate(dataset):
            if True:
                pbar.update()
                combine_target = torch.cat(target, 0).numpy()
                seg_target, class_target = utils.get_targets(combine_target)
                targets = (maybe_cuda(torch.tensor(seg_target).long()), maybe_cuda(torch.tensor(class_target).long()),
                           maybe_cuda(torch.tensor(combine_target).long()))
                output_seg, output_sc, loss_seg, loss_sc = model(raw_sentences, targets)
                preds_stats.add(output_sc, class_target)
                tp, fn = utils.get_seq_eval(utils.transform2seg(output_sc), combine_target)
                seg_tp, seg_fn = utils.get_seq_eval(output_seg, seg_target)
                all_tp.append(tp)
                all_fn.append(fn)
                all_seg_tp.append(seg_tp)
                all_seg_fn.append(seg_fn)
                all_acc.append(preds_stats.get_accuracy())
                com_gold.extend(class_target.tolist())
                com_pred.extend(output_sc)
                pbar.set_description('Testing, loss={:.4}'.format(loss_sc.item()))
            # except Exception as e:
            #     # logger.info('Exception "%s" in batch %s', e, i)
            #     logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
            #     pass
        sentence_f1 = metrics.f1_score(com_gold, com_pred, average="micro")
        unit = np.sum(all_tp) / (np.sum(all_tp) + np.sum(all_fn))
        span_f1 = 0. if unit == 0 else 2 * ((unit * unit) / (unit + unit))
        seg_unit = np.sum(all_seg_tp) / (np.sum(all_seg_tp) + np.sum(all_seg_fn))
        span_seg_f1 = 0. if seg_unit == 0 else 2 * ((seg_unit * seg_unit) / (seg_unit + seg_unit))
        logger.info(
            'Test Epoch: {} ,accuracy: {:.4},sentence-f1 {:.4},span-f1 {:.4},seg-span-f1{:.4} '.format(
                epoch + 1,
                np.mean(all_acc),
                float(sentence_f1),
                float(span_f1),
                float(span_seg_f1)))

        preds_stats.reset()

        return sentence_f1, span_f1, span_seg_f1


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
    global window_size
    window_size = utils.get_windows_size(test_dl)
    model, seg_optimizer, sc_optimizer = import_model(args.model,
                                                      utils.get_labels_num_by_config(config_args.language,
                                                                                     config_args.dataset,
                                                                                     config_args.type) + 1)
    seg_scheduler = torch.optim.lr_scheduler.StepLR(seg_optimizer, step_size=1, gamma=0.9)
    sc_scheduler = torch.optim.lr_scheduler.StepLR(sc_optimizer, step_size=1, gamma=0.9)
    if args.load_from:
        with open(args.load_from, 'rb') as f:
            model = torch.load(f)
    model.train()
    model = maybe_cuda(model)
    best_model = model
    optimizer = seg_optimizer
    scheduler = seg_scheduler
    best_sentence_f1 = 0
    best_span_f1 = 0
    best_span_seg_f1 = 0
    best_test_span_seg_f1 = 0
    count = 0
    flag = True
    for j in list(range(args.epochs)):
        train(model, args, j, train_dl, logger, optimizer, flag)
        scheduler.step()
        val_sentence_f1, val_span_f1, val_span_seg_f1 = validate(model, args, j, dev_dl, logger)
        if flag:
            if val_span_seg_f1 > best_span_seg_f1:
                best_span_seg_f1 = val_span_seg_f1
                count = 0
                test_sentence_f1, test_span_f1, test_span_seg_f1 = test(model, j, test_dl, logger)
                if test_span_seg_f1 > best_test_span_seg_f1:
                    best_model = model
                    best_test_span_seg_f1 = test_span_seg_f1
            else:
                count += 1
            if count >= 5:
                flag = False
                model = best_model
                optimizer = sc_optimizer
                scheduler = sc_scheduler
        if not flag and (val_sentence_f1 > best_sentence_f1 or val_span_f1 > best_span_f1):
            best_sentence_f1 = max(val_sentence_f1, best_sentence_f1)
            best_span_f1 = max(val_span_f1, best_span_f1)
            test(model, j, test_dl, logger)
            with (checkpoint_path / 'model{:03d}.t7'.format(j)).open('wb') as f:
                torch.save(model, f)


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
