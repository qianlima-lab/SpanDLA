# -*- coding:UTF-8 -*-
from __future__ import print_function
from pathlib2 import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from text_manipulation import word_model, extract_sentence_words
import utils
import math
import re
from sentence_transformers import util
from config import args
from embedding import word2vec, embedder

logger = utils.setup_logger(__name__, 'train.log')


def collate_fn(batch):
    batched_data = []
    batched_raw_sentences = []
    batched_targets = []
    batched_adj = []
    paths = []

    window_size = 1
    before_sentence_count = int(math.ceil(float(window_size - 1) / 2))
    after_sentence_count = window_size - before_sentence_count - 1

    for data, targets, targets_var, path, raw_sentences in batch:
        if (len(data) < 1):
            continue
        try:
            max_index = len(data)
            tensored_data = []
            for curr_sentence_index in list(range(0, len(data))):
                from_index = max([0, curr_sentence_index - before_sentence_count])
                to_index = min([curr_sentence_index + after_sentence_count + 1, max_index])
                sentences_window = [word for sentence in data[from_index:to_index] for word in sentence]
                tensored_data.append(torch.FloatTensor(np.concatenate(sentences_window, 0)))
            tensored_targets = torch.zeros(len(data)).long()
            index = 0
            for target in targets:
                tensored_targets[target] = targets_var[index]
                index = index + 1
            # adj = get_adj_by_cosine(tensored_data) if args.similarity == 0 else get_adj_by_sentence_bert(
            #    raw_sentences)
            adj = []
            batched_adj.append(adj)
            batched_data.append(tensored_data)
            batched_targets.append(tensored_targets)
            batched_raw_sentences.append(raw_sentences)
            paths.append(path)
        except Exception as e:
            logger.info('Exception "%s" in file: "%s"', e, path)
            logger.debug('Exception!', exc_info=True)
            continue
    return batched_data, batched_adj, batched_targets, batched_raw_sentences, paths


def clean_section(section):
    cleaned_section = section.replace("'' ", " ").replace(" 's", "'s").replace("``", "").strip(
        '\n')
    return cleaned_section


def read_wiki_seg_file(path):
    return


def read_wiki_section_file(path, dataset, status):
    with Path(path).open('r', encoding='utf-8') as f:
        raw_text = f.read()
    pattern = re.compile(r'==========.*')
    sections = [clean_section(section) for section in re.split(pattern, raw_text) if
                len(section) > 0 and section != "\n"]
    if dataset == 'pubmed' or dataset == 'nicta' or dataset == 'csabstruct':
        sections_info = [info.split(';') for info in re.findall(pattern, raw_text)]
    elif dataset == 'disease' or dataset == 'city':
        sections_info = [info.split(';') for info in re.findall(pattern, raw_text)[:-1]]
    else:
        pattern = re.compile(r'==========;.*')
        sections = [clean_section(section) for section in re.split(pattern, raw_text) if
                    len(section) > 0 and section != "\n"]
        sections_info = [info.split(';') for info in re.findall(pattern, raw_text)]
        sections = [replace_md_latex(replace_md_image(replace_md_bold(s))) for s in sections]
        if sections_info[-1][-1] == 'reference':
            sections_info = sections_info[:-1]
            sections = sections[:-1]
    targets_label = utils.get_label_encoding(sections_info)
    if args.train and status:
        temp = list(range(0, len(sections)))
        np.random.shuffle(temp)
        temp_targets_label = []
        temp_sections = []
        for i in temp:
            temp_targets_label.append(int(targets_label[i]))
            temp_sections.append(sections[i])
        targets_label = temp_targets_label
        sections = temp_sections

    targets = []
    new_text = []
    lastparagraphsentenceidx = 0
    sentences_length = 0
    raw_sentences = []
    for section in sections:
        sentences = [s.strip() for s in section.split('\n') if len(s.split()) > 0]
        if sentences:
            sentences_count = 0
            # This is the number of sentences in the paragraph and where we need to split.
            for sentence in sentences:
                words = extract_sentence_words(sentence)
                if (len(words) == 0):
                    continue
                temp = [word2vec[w].reshape(1, 300) for w in words]
                # temp.append(word2vec['\n'].reshape(1, 300))
                new_text.append(temp)
                sentences_count += 1
                raw_sentences.append(sentence)
            lastparagraphsentenceidx += sentences_count
            sentences_length += sentences_count
            targets.append(lastparagraphsentenceidx - 1)
    # if args.type == 1:
    #    targets_label, targets = targets_transform(sentences_length, targets_label, targets)
    return new_text, targets, targets_label, path, raw_sentences


def get_adj_by_cosine(x):
    average_word_embedding = []
    for sentence in x:
        average_word_embedding.append(torch.mean(sentence, 0).tolist())
    average_word_embedding = utils.maybe_cuda(torch.tensor(average_word_embedding))
    adj = util.pytorch_cos_sim(average_word_embedding, average_word_embedding)
    return adj


def get_adj_by_sentence_bert(x):
    if embedder is None:
        return np.random.randn(len(x) * len(x))
    all_batch_sentences_embedding = utils.maybe_cuda(embedder.encode(x, convert_to_tensor=True))
    cosine_sim = util.pytorch_cos_sim(all_batch_sentences_embedding, all_batch_sentences_embedding)
    adj = cosine_sim.gt(0.98).type(torch.float)
    return utils.maybe_cuda(adj)


def get_multi_label(headings):
    # TODO
    pattern = re.compile(r'\||\s+')
    heading = [s.lower() for s in re.split(pattern, headings) if len(s) > 0]
    return heading


# tramsform targets from 2-classify to n-classify
def targets_transform(sentence_length, targets_label, targets):
    new_targets_label = [-1] * sentence_length
    new_targets = [num for num in range(0, sentence_length)]
    index = 0
    pre_index = 0
    for cur_index in targets:
        new_targets_label[pre_index:cur_index + 1] = [targets_label[index]] * (cur_index - pre_index + 1)
        index += 1
        pre_index = cur_index + 1
    return new_targets_label, new_targets


def replace_md_latex(sentence):
    pattern = re.compile(r"\$.*?\$")
    return re.sub(pattern, u"公式", sentence)


def replace_md_image(sentence):
    pattern = re.compile(r"!\[image-([1-9][0-9]*)\]\(.*?\)|!\[Image\]\(?.*\)")
    return re.sub(pattern, '', sentence)


def replace_md_bold(sentence):
    pattern = re.compile(r"\*\*.*?\*\*")
    re.search(pattern, sentence)
    return re.sub(pattern, lambda m: m.group(0).strip('**'), sentence)


def get_single_label_encoding(label):
    return


def get_multi_label_encoding(label):
    return


# Returns a list of batch_size that contains a list of sentences, where each word is encoded using word2vec.
class RefDataset(Dataset):
    def __init__(self, root, status=False):
        self.textfiles = list(Path(root).glob('**/*[ref,txt,md]'))
        self.status = status
        if len(self.textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))

    def __getitem__(self, index):
        path = self.textfiles[index]
        result = ()
        if args.dataset == 'city' or args.dataset == 'disease' or args.dataset == 'pubmed' or args.dataset == 'nicta' or args.dataset == 'csabstruct':
            result = read_wiki_section_file(path, args.dataset, self.status)
        elif args.dataset == 'cities' or args.dataset == 'element' or args.dataset == 'wiki':
            result = read_wiki_seg_file(path)
        elif args.dataset == 'math':
            result = read_wiki_section_file(path, args.dataset, self.status)
        return result

    def __len__(self):
        return len(self.textfiles)
