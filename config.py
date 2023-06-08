import argparse
from pathlib2 import Path

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='nicta')
args.add_argument('--language', help='en or de', default='en')
args.add_argument('--type', help='0:no-label 1:single-label 2: multi-label', type=int, default=1)
args.add_argument('--cuda', type=bool, default=True)
args.add_argument('--train', type=bool, help='is shuffle for train', default=False)
args.add_argument('--similarity', type=int, help='0 word2vec 1  sentence bert', default=1)
args.add_argument('--sentence_embedding', type=int, help='0 word2vec to lstm 1  sentence bert', default=1)
args = args.parse_args(args=[])


def voca_list2dict(voca_list):
    voca_dict = {}
    for i, voca in enumerate(voca_list):
        voca_dict[voca.strip()] = i + 1
    return voca_dict


config_dict = dict(word2vec_path='/dev_data/sxc/data/word2vec/GoogleNews-vectors-negative300.bin',
                   en_word2vec_path='/dev_data/sxc/data/word2vec/wiki.en.bin',
                   de_word2vec_path='/dev_data/sxc/data/word2vec/wiki.de.bin',
                   en_disease_word2vec='/dev_data/sxc/data/word2vec/en-disease-word2vec.bin',
                   en_city_word2vec='/dev_data/sxc/data/word2vec/en-city-word2vec.bin',
                   de_disease_word2vec='',
                   de_city_word2vec='',
                   wiki_50_path='/dev_data/sxc/data/wiki-50',
                   elements_path='/dev_data/sxc/data/elements',
                   cities_path='/dev_data/sxc/data/cities',
                   clinical_path='/dev_data/sxc/data/clinical',
                   city_en_path='/dev_data/sxc/data/city/en',
                   city_de_path='/dev_data/sxc/data/city/de',
                   disease_en_path='/dev_data/sxc/data/disease/en',
                   disease_de_path='/dev_data/sxc/data/disease/de',
                   pubmed_en_path='/dev_data/sxc/data/pubmed',
                   csabstruct_en_path='/dev_data/sxc/data/csabstruct',
                   nicta_en_path='/dev_data/sxc/data/nicta',
                   math_en_path='/dev_data/sxc/data/math',
                   label_en_city_voca=voca_list2dict(Path('data/vocab/label_en_city.tsv').open('r').read().split('\n')),
                   label_en_disease_voca=voca_list2dict(
                       Path('data/vocab/label_en_disease.tsv').open('r').read().split('\n')),
                   label_de_city_voca=voca_list2dict(Path('data/vocab/label_de_city.tsv').open('r').read().split('\n')),
                   label_de_disease_voca=voca_list2dict(
                       Path('data/vocab/label_de_disease.tsv').open('r').read().split('\n')),
                   label_en_nicta_voca=voca_list2dict(
                       Path('data/vocab/label_en_nicta.tsv').open('r').read().split('\n')),
                   label_en_pubmed_voca=voca_list2dict(
                       Path('data/vocab/label_en_pubmed.tsv').open('r').read().split('\n')),
                   label_en_csabstruct_voca=voca_list2dict(
                       Path('data/vocab/label_en_csabstruct.tsv').open('r').read().split('\n')),
                   label_en_math_voca=voca_list2dict(
                       Path('data/vocab/label_en_math.tsv').open('r').read().split('\n'))
                   )

print(args)
