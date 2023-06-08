import nltk.data
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import wiki_utils
import wiki_thresholds
import utils
import jieba
import torch
from nltk.stem import WordNetLemmatizer

sentence_tokenizer = None
words_tokenizer = None
missing_stop_words = set(stopwords.words('english'))
logger = utils.setup_logger(__name__, 'text_manipulation.log', True)


def get_punkt():
    global sentence_tokenizer
    if sentence_tokenizer:
        return sentence_tokenizer

    try:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentence_tokenizer = tokenizer
    return sentence_tokenizer


def get_words_tokenizer():
    global words_tokenizer

    if words_tokenizer:
        return words_tokenizer

    words_tokenizer = RegexpTokenizer(r'\w+')
    return words_tokenizer


def split_sentence_with_list(sentence):
    list_pattern = "\n" + wiki_utils.get_list_token() + "."
    if sentence.endswith(list_pattern):
        # splited_sentence = [str for str in sentence.encode('utf-8').split("\n" + wiki_utils.get_list_token() + ".") if len(str) > 0]
        splited_sentence = [str for str in sentence.split("\n" + wiki_utils.get_list_token() + ".") if
                            len(str) > 0]
        splited_sentence.append(wiki_utils.get_list_token() + ".")
        return splited_sentence
    else:
        return [sentence]


def split_sentece_colon_new_line(sentence):
    splited_sentence = sentence.split(":\n")
    if (len(splited_sentence) == 1):
        return splited_sentence
    new_sentences = []
    # -1 . not to add ":" to last sentence
    for i in range(len(splited_sentence) - 1):
        if (len(splited_sentence[i]) > 0):
            new_sentences.append(splited_sentence[i] + ":")
    if (len(splited_sentence[-1]) > 0):
        new_sentences.append(splited_sentence[-1])
    return new_sentences


def split_long_sentences_with_backslash_n(max_words_in_sentence, sentences, doc_id):
    new_sentences = []
    for sentence in sentences:
        sentence_words = extract_sentence_words(sentence)
        if len(sentence_words) > max_words_in_sentence:
            splitted_sentences = sentence.split('\n')
            if len(splitted_sentences) > 1:
                logger.info(
                    "Sentence with backslash was splitted. Doc Id: " + str(doc_id) + "   Sentence:  " + sentence)
            new_sentences.extend(splitted_sentences)
        else:
            if "\n" in sentence:
                logger.info(
                    "No split for sentence with backslash n. Doc Id: " + str(doc_id) + "   Sentence:  " + sentence)
            new_sentences.append(sentence)
    return new_sentences


def split_sentences(text, doc_id):
    sentences = get_punkt().tokenize(text)
    senteces_list_fix = []
    for sentence in sentences:
        seplited_list_sentence = split_sentence_with_list(sentence)
        senteces_list_fix.extend(seplited_list_sentence)

    sentence_colon_fix = []
    for sentence in senteces_list_fix:
        splitted_colon_sentence = split_sentece_colon_new_line(sentence)
        sentence_colon_fix.extend(splitted_colon_sentence)

    sentences_without_backslash_n = split_long_sentences_with_backslash_n(
        wiki_thresholds.max_words_in_sentence_with_backslash_n, sentence_colon_fix, doc_id)

    ret_sentences = []
    for sentence in sentences_without_backslash_n:
        ret_sentences.append(sentence.replace('\n', ' '))
    return ret_sentences


def extract_sentence_words(sentence, remove_missing_emb_words=True, remove_special_tokens=True):
    if (remove_special_tokens):
        for token in wiki_utils.get_special_tokens():
            # Can't do on sentence words because tokenizer delete '***' of tokens.
            sentence = sentence.replace(token, "")
    tokenizer = get_words_tokenizer()
    sentence_words = tokenizer.tokenize(sentence)
    if remove_missing_emb_words:
        sentence_words = [w for w in sentence_words if w not in missing_stop_words]

    return sentence_words


def extract_cn_sentence_words(sentence):
    return jieba.lcut(sentence)


def word_model(word, model):
    if model is None:
        return np.random.randn(1, 300)
    else:
        if word in model:
            return model.get_word_vector(word).reshape(1, 300)
        else:
            return model.get_word_vector('UNK').reshape(1, 300)


def word_bert_model(sentence, bert_tokenizer, bert_model):
    sentence = '[CLS]' + sentence + '[SEP]'
    tokenized_sentence = bert_tokenizer.tokenize(sentence)
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_sentence)
    if len(indexed_tokens) > 512:
        indexed_tokens = indexed_tokens[0:512]
    segments_ids = [0] * len(indexed_tokens)
    tokens_tensor = utils.maybe_cuda(torch.tensor([indexed_tokens]))
    segments_tensors = utils.maybe_cuda(torch.tensor([segments_ids]))
    word_vectors = bert_model(tokens_tensor, segments_tensors)[0]
    word_vectors = word_vectors.squeeze(0)[1:word_vectors.size(1) - 1, :]
    return word_vectors.cpu().detach().numpy().tolist()


def validate_url(url):
    split1 = url.split('://')
    protocol = split1[0]
    split2 = split1[1].split('/')
    hostmain = split2[0]
    fileinfo = split2[-1].split('.')[-1]
    Protocol = ['http', 'https', 'ftp']
    Fileinfo = ['html', 'csv', 'docx']
    return Protocol.count(protocol) != 0 and Fileinfo.count(fileinfo) != 0
