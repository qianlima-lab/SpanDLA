from random import sample
from pathlib2 import Path
import re
import csv


def clean_section(section):
    cleaned_section = section.replace("'' ", " ").replace(" 's", "'s").replace("``", "").strip(
        '\n')
    return cleaned_section


def sample_triple(path):
    raw_text = Path(path).open('r', encoding='utf-8').read()
    pattern = re.compile(r'==========.*')
    sections_info = [info.split(';') for info in re.findall(pattern, raw_text)]
    sections = [clean_section(section) for section in re.split(pattern, raw_text) if
                len(section) > 0 and section != "\n"]
    sentences_info = []
    cur = 0
    for section, info, i in zip(sections, sections_info, range(0, len(sections))):

        sentences = [s.strip() for s in section.split('\n') if len(s.strip()) > 0]
        label = info[1].strip().lower()
        section_id = i
        left_boundary = cur
        right_boundary = cur + len(sentences) - 1
        for s in sentences:
            sentence = {}
            sentence['text'] = s
            sentence['label'] = label
            sentence['section_id'] = section_id
            sentence['left_boundary'] = left_boundary
            sentence['right_boundary'] = right_boundary
            sentences_info.append(sentence)
        cur += len(sentences)
    length = len(sentences_info)
    triples = []
    for i, s in enumerate(sentences_info):
        triple = {}
        left_boundary = s['left_boundary']
        right_boundary = s['right_boundary']
        # if left_boundary == right_boundary:
        #     pos = s['text']
        # else:
        #     pos = sample(sentences_info[left_boundary:i] + sentences_info[i + 1:right_boundary + 1], 1)[0]['text']
        pos = sample(sentences_info[left_boundary:right_boundary + 1], 1)[0]['text']
        if len(sections) == 1:
            neg = pos
        else:
            neg = sample(sentences_info[0:left_boundary] + sentences_info[right_boundary + 1:length], 1)[0]['text']
        # triple['anc'] = s['text']
        # triple['pos'] = pos
        # triple['neg'] = neg
        triples.append((s['text'], pos, neg))
    return triples


path = 'D:/work/text-segmentation/data/pubmed/train'
textfiles = list(Path(path).glob('**/*.txt'))
triples = []
for t in textfiles:
    triple = sample_triple(t)
    triples.extend(triple)
head = ('anc', 'pos', 'neg')
f = Path('D:/work/text-segmentation/data/pubmed/train.csv').open('w', newline='', encoding='utf-8')
writer = csv.writer(f)
writer.writerow(head)
writer.writerows(triples)
f.close()
