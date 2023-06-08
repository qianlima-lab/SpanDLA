from pathlib import Path
import json
import re
from nltk.corpus import wordnet
from transformers import BertTokenizer
import numpy as np
from data_loader import clean_section

text_path = "C:/Users/sxc/Desktop/text-segmentation/wikicities-english/test/wikicities.text"
segmenttitles_path = "C:/Users/sxc/Desktop/text-segmentation/wikicities-english/test/wikicities.segmenttitles"
meta_path = "C:/Users/sxc/Desktop/text-segmentation/wikicities-english/test/wikicities.meta"
separate = '========'
stop_words_en = set(Path('data/stopwords_en.csv').open('r', encoding='utf-8').read().split('\n'))
stop_words_de = set(Path('data/stopwords_de.csv').open('r', encoding='utf-8').read().split('\n'))
dash_pattern = re.compile(r"[\-_\/]+")
punct_pattern = re.compile(r"[^\w\s\-_]+")
space_pattern = re.compile(r"[\s]+")
numeric_pattern = re.compile(r"[\d]+")
skip_space_before = {",", ".", ":", ";", "?", "!", ")", "]", "'m", "'s", "'re", "'ve", "'d", "'ll", "n't"}
skip_space_after = {"(", "[", "", "\n"}
umlaut_replacements = [["Ä", "Ae"], ["Ü", "Ue"], ["Ö", "Oe"], ["ä", "ae"], ["ü", "ue"], ["ö", "oe"], ["ß", "ss"],
                       ["–", "-"]]
abbreviations = {"Adj.", "Adm.", "Adv.", "Asst.", "Bart.", "Bldg.", "Brig.", "Bros.", "Capt.", "Cmdr.", "Col.",
                 "Comdr.", "Con.", "Corp.", "Cpl.", "DR.", "Dr.", "Drs.", "Ens.", "Gen.", "Gov.", "Hon.", "Hr.",
                 "Hosp.", "Insp.", "Lt.", "MM.", "MR.", "MRS.", "MS.", "Maj.", "Messrs.",
                 "Mlle.", "Mme.", "Mr.", "Mrs.", "Ms.", "Msgr.", "Op.", "Ord.", "Pfc.", "Ph.", "Prof.", "Pvt.",
                 "Rep.", "Reps.", "Res.", "Rev.", "Rt.", "Sen.", "Sens.", "Sfc.", "Sgt.",
                 "Sr.", "St.", "Supt.", "Surg", "v.", "vs.", "i.e.", "rev.", "e.g.", "No.", "Nr.", "pp.", "I.", "II.",
                 "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX.", "X.", "XI.", "XII.", "XIII.",
                 "XIV.", "XV.", "XVI.", "XVII.", "XVIII.", "XIX.", "XX.", "i.", "ii.", "iii.", "iv.", "v.", "vi.",
                 "vii.", "viii.", "ix.", "x.", "xi.", "xii.", "xiii.", "xiv.", "xv.",
                 "xvi.", "xvii.", "xviii.", "xix.", "xx.", "Adj.", "Adm.", "Adv.", "Asst.", "Bart.", "Bldg.",
                 "Brig.", "Bros.", "Capt.", "Cmdr.", "Col.", "Comdr.", "Con.", "Corp.",
                 "Cpl.", "DR.", "Dr.", "Ens.", "Gen.", "Gov.", "Hon.", "Hosp.", "Insp.", "Lt.", "MM.", "MR.", "MRS.",
                 "MS.", "Maj.", "Messrs.", "Mlle.", "Mme.", "Mr.", "Mrs.", "Ms.",
                 "Msgr.", "Op.", "Ord.", "Pfc.", "Ph.", "Prof.", "Pvt.", "Rep.", "Reps.", "Res.", "Rev.", "Rt.",
                 "Sen.", "Sens.", "Sfc.", "Sgt.", "Sr.", "St.", "Supt.", "Surg.",
                 "Mio.", "Mrd.", "bzw.", "v.", "vs.", "usw.", "d.h.", "z.B.", "u.a.", "etc.", "Mrd.", "MwSt.",
                 "ggf.", "d.J.", "D.h.", "m.E.", "vgl.", "I.F.", "z.T.", "sogen.", "ff.",
                 "u.E.", "g.U.", "g.g.A.", "c.-à-d.", "Buchst.", "u.s.w.", "sog.", "u.ä.", "Std.", "evtl.", "Zt.",
                 "Chr.", "u.U.", "o.ä.", "Ltd.", "b.A.", "z.Zt.", "spp.", "sen.",
                 "SA.", "k.o.", "jun.", "i.H.v.", "dgl.", "dergl.", "Co.", "zzt.", "usf.", "s.p.a.", "Dkr.", "Corp.",
                 "bzgl.", "BSE.", "No.", "Nos.", "Art.", "Nr.", "pp.", "ca.", "Ca"}


def process(des_path):
    raw_segmenttitles = Path(segmenttitles_path).open('r', encoding='utf-8').read()
    raw_text = Path(text_path).open('r', encoding='utf-8').read()
    raw_meta = Path(meta_path).open('r', encoding='utf-8').read()

    segmenttitles = raw_segmenttitles.split('\n')
    text = raw_text.split('\n')
    meta = [int(s) for s in raw_meta.split('\n')]

    document = []

    left = 0
    right = 0
    for index in meta:
        right = right + index
        print(left, right)
        document.append((text[left:right], segmenttitles[left:right]))
        left = right
    new_documents = []
    for (text, segmenttitles) in document:
        seg_info = [segmenttitle.split(',') for segmenttitle in segmenttitles]
        pre_topic = seg_info[0][2]
        new_document = []
        for index in range(0, len(seg_info)):
            topic = seg_info[index][2]
            if topic != pre_topic:
                new_document.append(separate + ',' + pre_topic)
            new_document.append(text[index])
            pre_topic = topic
        new_document.append(separate + ',' + pre_topic)
        new_documents.append(new_document)
    write_documents(new_documents, des_path)


def write_documents(documents, des_path):
    count = 0
    for document in documents:
        text = ''
        for sentence in document:
            text += sentence + '\n'
        Path(des_path + str(count) + '.ref').open('w', encoding='utf-8').write(text)
        count += 1


def get_label_voc(paths):
    help_dict = {}
    label_dict = {}
    heading_dict = {}
    raw_json = []
    for path in paths:
        fs = Path(path).open('r', encoding='utf-8')
        raw_json.extend(json.load(fs))
    for document in raw_json:
        annotations = document['annotations']
        for annotation in annotations:
            section_heading = replace_abbreviations(annotation['sectionHeading']).lower()
            # cleaned_section_heading = clean_heading(section_heading)
            # cleaned_section_heading = section_heading
            headings = re.split(r'\s+|\|\\|/', section_heading)
            for heading in headings:
                if is_legal_word(heading) and heading.count('.') == 0:
                    cleaned_heading = clean_heading(heading)
                    if len(cleaned_heading):
                        add_to_dict(heading_dict, cleaned_heading)
                        help_dict[cleaned_heading] = heading
            section_label = annotation['sectionLabel']
            add_to_dict(label_dict, section_label)
    heading_set = set()
    s = {}
    for heading in heading_dict:
        if heading_dict[heading] > 20:
            heading_set.add(heading)
            s[heading] = help_dict[heading]
    return set(label_dict), heading_set, s


def add_to_dict(dict, element):
    if element not in dict:
        dict[element] = 0
    dict[element] += 1


def is_legal_word(word):
    return word not in stop_words_de \
           and word not in stop_words_en


def clean_heading(heading):
    cleaned_heading = replace_umlauts(heading)
    cleaned_heading = re.sub(numeric_pattern, '#', cleaned_heading)
    cleaned_heading = re.sub(punct_pattern, '', cleaned_heading)
    cleaned_heading = re.sub(dash_pattern, '', cleaned_heading)
    # cleaned_heading = re.sub(space_pattern, '', cleaned_heading)

    return cleaned_heading


def replace_umlauts(heading):
    for rp in umlaut_replacements:
        heading = heading.replace(rp[0], rp[1])
    return heading


def replace_abbreviations(heading):
    for a in abbreviations:
        heading = heading.replace(a, '')
    return heading


def get_info():
    root = 'D:/work/text-segmentation/data/disease/en'
    textfiles = list(Path(root).glob('**/*.ref'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_tokens_sum_per_doc = 0
    max_tokens_num_per_sentence = 0
    max_sentence_sum_per_doc = 0
    all_tokens_sum_per_doc = []
    all_tokens_num_per_sentence = []
    all_sentence_sum_per_doc = []
    res_path = [''] * 3
    for path in textfiles:
        raw_text = Path(path).open('r', encoding='utf-8').read()
        pattern = re.compile(r'==========.*')
        text = re.sub(pattern, '', raw_text)
        sentences = [s.strip() for s in text.split('\n') if len(s.strip()) > 0]
        tokens_num = 0
        for s in sentences:
            encoding = tokenizer(s, return_tensors='np', padding=False)
            input_ids = encoding['input_ids']
            tokens_num += len(input_ids[0]) - 2
            all_tokens_num_per_sentence.append(len(input_ids[0]) - 2)
            if max_tokens_num_per_sentence < len(input_ids[0]) - 2:
                max_tokens_num_per_sentence = max(max_tokens_num_per_sentence, len(input_ids[0]) - 2)
                res_path[0] = path
        all_sentence_sum_per_doc.append(len(sentences))
        all_tokens_sum_per_doc.append(tokens_num)
        if max_tokens_sum_per_doc < tokens_num:
            res_path[1] = path
            max_tokens_sum_per_doc = max(max_tokens_sum_per_doc, tokens_num)
        if max_sentence_sum_per_doc < len(sentences):
            res_path[2] = path
            max_sentence_sum_per_doc = max(max_sentence_sum_per_doc, len(sentences))
    print((max_tokens_num_per_sentence, np.mean(all_tokens_num_per_sentence))
          , (max_tokens_sum_per_doc, np.mean(all_tokens_sum_per_doc))
          , (max_sentence_sum_per_doc, np.mean(all_sentence_sum_per_doc)), res_path)


def trans_data_format(source_path, output_path):
    source_path = 'D:/work/text-segmentation/data/nicta/train'
    textfiles = list(Path(source_path).glob('**/*.txt'))
    data_dict_list = []
    for path in textfiles:
        doc_dict = {}
        raw_text = Path(path).open('r', encoding='utf-8').read()
        pattern = re.compile(r'==========.*')
        sections_info = [info.split(';') for info in re.findall(pattern, raw_text)]
        sections = [clean_section(section) for section in re.split(pattern, raw_text) if
                    len(section) > 0 and section != "\n"]
        doc_sentences = []
        sentences_labels = []
        for i, section in enumerate(sections):
            sentences = [s.strip() for s in section.split('\n') if len(s.strip()) > 0]
            doc_sentences.extend(sentences)
            sentences_labels.extend([sections_info[i][1].strip()] * len(sentences))
        doc_dict['abstract_id'] = 0
        doc_dict['sentences'] = doc_sentences
        doc_dict['labels'] = sentences_labels
        data_dict_list.append(doc_dict)
    f = Path('D:/work/text-segmentation/data/nicta/train.json').open('w+', encoding='utf-8')
    # f = Path(output_path).open('w+', encoding='utf-8')
    for x in data_dict_list:
        f.write(json.dumps(x) + '\n')
    f.close()


def statistics():
    path = 'D:/work/text-segmentation/data/city/en'
    textfiles = list(Path(path).glob('**/*.txt'))
    seg_len = []
    doc_len = []
    dict = {}
    sum1 = 0
    max_span_len = 0
    span_len = []
    for file in textfiles:
        raw_text = Path(file).open('r', encoding='utf-8').read()
        pattern = re.compile(r'==========.*')
        sections_info = [info.split(';') for info in re.findall(pattern, raw_text)[:-1]]
        sections = [clean_section(section) for section in re.split(pattern, raw_text) if
                    len(section) > 0 and section != "\n"]
        sum1 += (np.array([len(s.split('\n')) for s in sections]) >= 10).sum()
        span_len.extend([len(s.split('\n')) for s in sections])
        for x in sections_info:
            add_to_dict(dict, x[1].lower())
        seg_len.append(len(sections_info))
        doc_len.append(len([s for s in raw_text.split('\n') if len(s.strip()) > 0 and s not in sections_info]))
    seg_sum = sum([dict[x] for x in dict])
    print(max(span_len))
    print(np.mean(seg_len), np.std(seg_len), np.mean(doc_len), np.std(doc_len), [(x, dict[x] / seg_sum) for x in dict])


def CSAbstruct_process(path):
    path = 'D:/work/text-segmentation/sequential_sentence_classification/data/CSAbstruct/test.jsonl'
    raw_text = Path(path).open('r', encoding='utf-8').readlines()
    json_list = []
    for line in raw_text:
        json_list.append(json.loads(line))
    for i, doc in enumerate(json_list):
        f = Path(
            'D:/work/text-segmentation/sequential_sentence_classification/data/CSAbstruct/test/'
            + str(i) + '.txt').open('w', encoding='utf-8')
        sentences = doc['sentences']
        labels = doc['labels']
        text = ''
        cur_label = ''
        for s, l in zip(sentences, labels):
            if l != cur_label:
                text += '==========;' + l + '\n'
                cur_label = l
            text += s + '\n'
        f.write(text)
        f.close()


def dataset():
    path = 'D:/work/text-segmentation/data/disease/en/val'
    textfiles = list(Path(path).glob('**/*.ref'))
    des_path = 'D:/work/text-segmentation/data/disease/en/dev_clean.txt'
    count = 0
    f = Path(des_path).open('w+', encoding='utf-8')
    for file in textfiles:
        raw_text = Path(file).open('r', encoding='utf-8').read()
        pattern = re.compile(r'==========.*')
        sections_info = [info.split(';')[1].split('.')[1].lower() for info in re.findall(pattern, raw_text)[:-1]]
        sections = [clean_section(section) for section in re.split(pattern, raw_text) if
                    len(section) > 0 and section != "\n"]
        text = '###' + str(count) + '\n'
        count += 1
        for l, s in zip(sections_info, sections):
            sentences = [sen.strip() for sen in s.split('\n') if len(sen.split()) > 0]
            for line in sentences:
                text += l + ' ' + line + '\n'
        f.write(text)
        f.write('\n')
    f.close()
