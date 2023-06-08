# -*- coding:UTF-8 -*-
from __future__ import division

from visualization import plt_heatmap
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from data_loader import RefDataset, collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import utils
import os
import sys
from pathlib2 import Path
import accuracy
from config import args as config_args
from timeit import default_timer as timer
from utils import get_voca_dict_from_config

logger = utils.setup_logger(__name__, 'test_accuracy.log')


def main(args):
    start = timer()

    sys.path.append(str(Path(__file__).parent))

    preds_stats = utils.predictions_analysis()

    dataset_folders = [utils.get_data_path_by_config(config_args.language, config_args.dataset) + '/test']
    with open(args.model, 'rb') as f:
        model = torch.load(f)

    model = maybe_cuda(model)
    model.eval()

    for dataset_path in dataset_folders:
        dataset = RefDataset(dataset_path)
        dl = DataLoader(dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=False)
        windows_size = utils.get_windows_size(dl)
        all_acc = []
        all_tp = []
        all_fn = []
        com_gold = []
        com_pred = []
        label_tp_dict = None
        label_fn_dict = None
        length_tp_dict = None
        length_fn_dict = None
        label_tp_error_dict = None
        label_fn_error_dict = None
        de_pred = []
        pk = []
        count, sum = 0, 0
        total_sc_loss = float(0)
        total_seg_loss = float(0)
        with tqdm(desc='Testing', total=len(dl)) as pbar:
            for i, (data, adj, target, raw_sentences, paths) in enumerate(dl):
                if True:
                    pbar.update()
                    combine_target = torch.cat(target, 0).numpy()
                    seg_target, class_target = utils.get_targets(combine_target)
                    targets = (
                        maybe_cuda(torch.tensor(seg_target).long()), maybe_cuda(torch.tensor(class_target).long()),
                        maybe_cuda(torch.tensor(combine_target).long()))
                    output_seg, loss, de_output_seg, att_score, att_target = model(raw_sentences, targets)
                    plt_heatmap(torch.softmax(att_score[0], dim=1).cpu().detach().numpy(),
                                './' + args.prefix + '-attention/' + str(i) + '_score.png')
                    plt_heatmap(att_target[0].cpu().detach().numpy(), './' + args.prefix + '-attention/' + str(i) + '_target.png')

                    preds_stats.add(output_seg, class_target)
                    tp, fn = utils.get_seq_eval(utils.transform2seg(output_seg), combine_target)
                    if fn == 0:
                        add = '_true'
                    else:
                        add = ''
                    write_seg(raw_sentences, combine_target, './' + args.prefix + '-seg/' + str(i) + '_gold.txt')
                    write_seg(raw_sentences, utils.transform2seg(output_seg),
                              './' + args.prefix + '-seg/' + str(i) + add + '_pred.txt')
                    pk.append(
                        utils.calculate_seg_eval((utils.transform2seg(output_seg) > 0).astype(np.int), seg_target))
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
                                                                        utils.transform2seg(com_pred),
                                                                        label_tp_dict,
                                                                        label_fn_dict)
            length_tp_dict, length_fn_dict = utils.get_seq_eval_per_length(utils.transform2seg(com_gold),
                                                                           utils.transform2seg(com_pred),
                                                                           length_tp_dict,
                                                                           length_fn_dict)
            label_span_f1_dict = utils.get_per_label_f1(label_tp_dict, label_fn_dict)
            length_span_f1_dict = utils.get_per_length_f1(length_tp_dict, length_fn_dict)
            unit = np.sum(all_tp) / (np.sum(all_tp) + np.sum(all_fn))
            span_f1 = 0. if unit == 0 else 2 * ((unit * unit) / (unit + unit))
            logger.info(
                'Testing: accuracy: {:.4},sentence-f1 {:.4},span-f1 {:.4} right:{:.4} pk:{:.4}'.format(
                    np.mean(all_acc),
                    float(sentence_f1),
                    float(span_f1), count / sum,
                    np.mean(pk)))
            total_sc_loss = total_sc_loss / len(dataset)
            total_seg_loss = total_seg_loss / len(dataset)
            logger.info('sentence_f1_per_label {}'.format(sentence_f1_per_label))
            logger.info('span_f1_per_label {}'.format(label_span_f1_dict))
            logger.info('span_f1_per_length {}'.format(length_span_f1_dict))
            logger.debug(
                'Testing Epoch: {}, sc_loss={:.4} seg_loss={:.4}.'.format(i + 1, total_sc_loss, total_seg_loss))
            preds_stats.reset()


def write_seg(sentences, seg, path):
    seg = seg.tolist()
    file = Path(path).open('w', encoding='utf-8')
    res = ''
    idx_to_tag = {idx: tag for tag, idx in
                  get_voca_dict_from_config().items()}
    for sen, s in zip(sentences[0], seg):
        res += sen
        if s != 0:
            res += '\n==========,' + str(idx_to_tag[s])
        res += '\n'
    file.write(res)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--bs', help='Batch size', type=int, default=8)
    parser.add_argument('--model', help='Model to run - will import and run', required=True)
    parser.add_argument('--prefix', type=str)
    main(parser.parse_args())
