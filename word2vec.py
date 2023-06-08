import gensim
from config import args
from pathlib2 import Path
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import re
from nltk.corpus import stopwords
from utils import *

stop_words = set(stopwords.words('english'))

textfiles = list(Path(get_data_path_by_config('en', 'disease')).glob('**/*.ref'))
des_path = get_data_path_by_config('en', 'disease') + '/corpus.txt'
cache = ''
length = len(textfiles)
count = 0
for path in textfiles:
    pattern = re.compile(r'==========.*\n')
    text = Path(path).open('r', encoding='utf-8').read()
    text = re.sub(pattern, '', text)
    cache += text
    count += 1
    if count % 1000 == 0:
        Path(des_path).open('a', encoding='utf-8').write(cache)
        cache = ''
        print(length - count)

sentences = LineSentence(get_data_path_by_config('en', 'disease') + '/corpus.txt')
new_sentences = []
for sentence in sentences:
    new_sentence = []
    for word in sentence:
        new_sentence.append(word if word not in stop_words else 'UNK')
    new_sentences.append(new_sentence)
model = Word2Vec(new_sentences, size=256, window=7)
model.wv.save_word2vec_format('D:/work/text-segmentation/data/word2vec/en-disease-word2vec.bin', binary=True)


