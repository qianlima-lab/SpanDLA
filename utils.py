import logging
import random
import sys
from shutil import copy
from decimal import Decimal
import numpy as np
from pathlib2 import Path
from nltk.metrics.segmentation import pk
from segeval.window.windowdiff import window_diff
from sklearn import metrics

from config import args, config_dict


def maybe_cuda(x):
    if args.cuda:
        return x.cuda()
    return x


def setup_logger(logger_name, filename, delete_old=False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    stderr_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(filename, mode='w') if delete_old else logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    stderr_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stderr_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)
    return logger


def unsort(sort_order):
    result = [-1] * len(sort_order)

    for i, index in enumerate(sort_order):
        result[index] = i

    return result


class f1(object):

    def __init__(self, ner_size):
        self.ner_size = ner_size
        self.tp = np.array([0] * (ner_size + 1))
        self.fp = np.array([0] * (ner_size + 1))
        self.fn = np.array([0] * (ner_size + 1))

    def add(self, preds, targets, length):
        tp = self.tp
        fp = self.fp
        fn = self.fn
        ner_size = self.ner_size

        prediction = np.argmax(preds, 2)

        for i in range(len(targets)):
            for j in range(length[i]):
                if targets[i, j] == prediction[i, j]:
                    tp[targets[i, j]] += 1
                else:
                    fp[targets[i, j]] += 1
                    fn[prediction[i, j]] += 1

        unnamed_entity = ner_size - 1
        for i in range(ner_size):
            if i != unnamed_entity:
                tp[ner_size] += tp[i]
                fp[ner_size] += fp[i]
                fn[ner_size] += fn[i]

    def score(self):
        tp = self.tp
        fp = self.fp
        fn = self.fn
        ner_size = self.ner_size

        precision = []
        recall = []
        fscore = []
        for i in range(ner_size + 1):
            precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
            recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
            fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
        print(fscore)

        return fscore[ner_size]


class predictions_analysis(object):

    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def add(self, predicions, targets):
        self.tp += ((predicions == targets) & (1 == predicions)).sum()
        self.tn += ((predicions == targets) & (0 == predicions)).sum()
        self.fp += ((predicions != targets) & (1 == predicions)).sum()
        self.fn += ((predicions != targets) & (0 == predicions)).sum()

    def calc_recall(self):
        if self.tp == 0 and self.fn == 0:
            return -1

        return np.true_divide(self.tp, self.tp + self.fn)

    def calc_precision(self):
        if self.tp == 0 and self.fp == 0:
            return -1

        return np.true_divide(self.tp, self.tp + self.fp)

    def get_f1(self):
        if (self.tp + self.fp == 0):
            return 0.0
        if (self.tp + self.fn == 0):
            return 0.0
        precision = self.calc_precision()
        recall = self.calc_recall()
        if (not ((precision + recall) == 0)):
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return f1

    def get_accuracy(self):

        total = self.tp + self.tn + self.fp + self.fn
        if (total == 0):
            return 0.0
        else:
            return np.true_divide(self.tp + self.tn, total)

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fn = 0
        self.fp = 0


def get_random_files(count, input_folder, output_folder, specific_section=True):
    files = Path(input_folder).glob('*/*/*/*') if specific_section else Path(input_folder).glob('*/*/*/*/*')
    file_paths = []
    for f in files:
        file_paths.append(f)

    random_paths = random.sample(file_paths, count)

    for random_path in random_paths:
        output_path = Path(output_folder).joinpath(random_path.name)
        copy(str(random_path), str(output_path))


def get_label_encoding(sections_info):
    encoding = []
    voca_dict = {}
    if args.type == 0:
        encoding = np.ones(len(sections_info), dtype=np.int).tolist()
    if args.type == 1:
        labels = [info[1].strip() for info in sections_info]
        voca_dict = get_voca_dict_from_config()
        for label in labels:
            encoding.append(voca_dict[label.lower()])
    if args.type == 2:
        labels = [info[2].strip() for info in sections_info]
        # TODO
    return encoding


def get_voca_dict_from_config():
    return config_dict['label_' + args.language + '_' + args.dataset + '_voca']


def get_data_path_by_config(language, dataset):
    datapath = ''
    if language == 'en':
        if dataset == 'disease':
            datapath = config_dict['disease_en_path']
        elif dataset == 'city':
            datapath = config_dict['city_en_path']
        elif dataset == 'nicta':
            datapath = config_dict['nicta_en_path']
        elif dataset == 'pubmed':
            datapath = config_dict['pubmed_en_path']
        elif dataset == 'csabstruct':
            datapath = config_dict['csabstruct_en_path']
        elif dataset == 'math':
            datapath = config_dict['math_en_path']

    elif language == 'de':
        if dataset == 'disease':
            datapath = config_dict['disease_de_path']
        elif dataset == 'city':
            datapath = config_dict['city_de_path']
    return datapath


def get_labels_num_by_config(language, dataset, type):
    if type == 0:
        return 2
    elif type == 1:
        if language == 'en':
            if dataset == 'disease':
                return len(config_dict['label_en_disease_voca'])
            elif dataset == 'city':
                return len(config_dict['label_en_city_voca'])
            elif dataset == 'nicta':
                return len(config_dict['label_en_nicta_voca'])
            elif dataset == 'pubmed':
                return len(config_dict['label_en_pubmed_voca'])
            elif dataset == 'csabstruct':
                return len(config_dict['label_en_csabstruct_voca'])
            elif dataset == 'math':
                return len(config_dict['label_en_math_voca']) - 1
        elif language == 'de':
            if dataset == 'disease':
                return len(config_dict['label_de_disease_voca'])
            elif dataset == 'city':
                return len(config_dict['label_de_city_voca'])
    elif type == 2:
        # TODO
        return 0

    return 0


def get_word2vec_path(language):
    return config_dict[language + '_word2vec_path']


def transform2seg(output_class):
    pre = 0
    cur = 1
    length = len(output_class)
    transformed_output = np.array([0] * length)
    while cur < length:
        if output_class[pre] != output_class[cur]:
            transformed_output[pre] = output_class[pre]
        pre = cur
        cur += 1
    transformed_output[length - 1] = output_class[length - 1]
    return np.array(transformed_output)


def transform2class(seg):
    pre = 0
    length = len(seg)
    result = [-1] * length
    cur = pre
    while cur < length:
        while cur < length and seg[cur] == 0:
            cur += 1
        if cur == length:
            result[pre:length] = [seg[pre - 1]] * (length - pre)
            return result
        result[pre:cur + 1] = [seg[cur]] * (cur - pre + 1)
        cur += 1
        pre = cur
    return result


def calculate_seg_eval(output, target):
    output = [str(o) for o in output]
    target = [str(t) for t in target]
    # diff, deno = window_diff(h, gold, one_minus=False, return_parts=True)
    return pk(target, output)


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    print(mrec[1:], mrec[:-1])
    print(i)
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calculate_map(gold, predict):
    y_true = np.zeros((len(gold), predict.shape[1]))
    for i, x in enumerate(gold):
        y_true[i][x - 1] = 1
    return metrics.average_precision_score(y_true, predict, average='micro')


def get_targets(combine_target):
    seg_target = (combine_target > 0).astype(np.int)
    class_target = transform2class(combine_target)
    return np.array(seg_target), np.array(class_target)


def get_section_level_evl(gold, predict_score, predict_label):
    predict_score = predict_score[:, 1:]
    index = []
    for i, x in enumerate(predict_label):
        if x > 0:
            index.append(i)
    selected_predict_score = predict_score[index]
    left = 0
    gold = transform2class(gold)
    selected_gold = []
    for right in index:
        max_overlap = np.argmax(np.bincount(gold[left:right + 1]))
        selected_gold.append(max_overlap)
        left = right
    f1 = metrics.f1_score(selected_gold, predict_label[index], average='micro')
    y_true = np.zeros((len(selected_gold), predict_score.shape[1]))
    if len(selected_gold) == 0:
        return 0, 0
    for i, x in enumerate(selected_gold):
        y_true[i][x - 1] = 1
    map = metrics.average_precision_score(y_true, selected_predict_score, average='micro')
    return f1, map


def get_windows_size(dataset):
    sum_seg_len = 0
    count_seg = 0
    for i, (data, adj, target, raw_sentences, paths) in enumerate(dataset):
        sum_seg_len += sum([len(t) for t in target])
        count_seg += sum([(t > 0).numpy().sum() for t in target])
    window_size = max(int(sum_seg_len / count_seg / 2.), 2)
    return window_size


def get_seq_eval(gold, pred):
    tp = 0
    fn = 0
    gold_mass = get_mass(gold)
    pred_mass = get_mass(pred)
    for g in gold_mass:
        if g in pred_mass:
            tp += 1
        else:
            fn += 1
    return tp, fn


def get_seq_eval_per_label(gold, pred, label_tp_dict, label_fn_dict):
    if label_tp_dict is None:
        label_tp_dict = {tag: 0 for tag in
                         get_voca_dict_from_config()}
    if label_fn_dict is None:
        label_fn_dict = {tag: 0 for tag in
                         get_voca_dict_from_config()}
    gold_mass = get_mass(gold)
    pred_mass = get_mass(pred)
    idx_to_tag = {idx: tag for tag, idx in
                  get_voca_dict_from_config().items()}
    for g in gold_mass:
        if g in pred_mass:
            label_tp_dict[idx_to_tag[g[2]]] += 1
        else:
            label_fn_dict[idx_to_tag[g[2]]] += 1
    return label_tp_dict, label_fn_dict


def get_seq_eval_per_length(gold, pred, length_tp_dict, length_fn_dict):
    if length_tp_dict is None:
        length_tp_dict = {tag: 0 for tag in
                          range(1, 8)}
    if length_fn_dict is None:
        length_fn_dict = {l: 0 for l in
                          range(1, 8)}
    gold_mass = get_mass(gold)
    pred_mass = get_mass(pred)
    for g in gold_mass:
        length = (int(g[1]) - int(g[0]) + 1) if (int(g[1]) - int(g[0]) + 1) < 7 else 7
        if g in pred_mass:
            length_tp_dict[length] += 1
        else:
            length_fn_dict[length] += 1
    return length_tp_dict, length_fn_dict


def get_per_label_f1(label_tp_dict, label_fn_dict):
    label_f1_dict = {tag: 0.0 for tag in
                     get_voca_dict_from_config()}
    for tag in label_f1_dict:
        fn = label_fn_dict[tag]
        tp = label_tp_dict[tag]
        if fn + tp == 0:
            continue
        label_f1_dict[tag] = tp / (fn + tp)
    return label_f1_dict


def get_per_length_f1(length_tp_dict, length_fn_dict):
    count = 0
    length_f1_dict = {length: 0.0 for length in
                      range(1, 8)}
    for length in length_f1_dict:
        fn = length_fn_dict[length]
        tp = length_tp_dict[length]
        count += fn + tp
    for length in length_f1_dict:
        fn = length_fn_dict[length]
        tp = length_tp_dict[length]
        if fn + tp == 0:
            continue
        length_f1_dict[length] = (tp / (fn + tp), (tp + fn) / count)
    return length_f1_dict


def get_mass(seg):
    res = []
    cur = 0
    for i, x in enumerate(seg):
        if x > 0:
            res.append((cur, i, x))
            cur = i + 1
    return set(res)


def combinetag2spantag(inputs):
    outputs = []
    for input in inputs:
        output = []
        old_tag = 99
        cur = 0
        for i, x in enumerate(input):
            if x > 0:
                output.append([cur, i, old_tag, x])
                old_tag = x
                cur = i + 1
        outputs.append(output)
    return outputs


def test(gold, pred, label_tp_dict, label_fn_dict):
    if label_tp_dict is None:
        label_tp_dict = {tag: 0 for tag in
                         get_voca_dict_from_config()}
    if label_fn_dict is None:
        label_fn_dict = {tag: 0 for tag in
                         get_voca_dict_from_config()}
    gold_mass = get_mass(gold)
    pred_mass = get_mass(pred)
    sentences_pred = transform2class(pred)
    idx_to_tag = {idx: tag for tag, idx in
                  get_voca_dict_from_config().items()}
    for g in gold_mass:
        if g in pred_mass:
            label_tp_dict[idx_to_tag[g[2]]] += 1
        else:
            if idx_to_tag[g[2]] == 'background':
                l = np.argmax(np.bincount(sentences_pred[g[0]:g[1] + 1]))
                label_fn_dict[idx_to_tag[l]] += 1
    return label_tp_dict, label_fn_dict
