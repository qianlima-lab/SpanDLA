import gensim
from sentence_transformers import SentenceTransformer
from config import config_dict, args
from utils import get_word2vec_path
import fasttext
embedder = None
word2vec = fasttext.load_model(get_word2vec_path(args.language))