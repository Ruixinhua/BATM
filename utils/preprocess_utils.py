import re
import string

from nltk.corpus import stopwords as stop_words
from torchtext.data import get_tokenizer


def text2index(text, word_dict, method="keep", ignore=True):
    return word2index(word_dict, tokenize(text, method), ignore)


def clean_text(text):
    rule = string.punctuation + "0123456789"
    return re.sub(rf'([^{rule}a-zA-Z ])', r" ", text)


def aggressive_process(text):
    stopwords = set(stop_words.words("english"))
    text = text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    text = text.translate(str.maketrans("0123456789", ' ' * len("0123456789")))
    text = [w for w in text.split() if len(w) > 0 and w not in stopwords]
    return text


def tokenize(text, method="keep_all"):
    tokens = []
    text = clean_text(text)
    rule = string.punctuation + "0123456789"
    tokenizer = get_tokenizer('basic_english')
    if method == "keep_all":
        tokens = tokenizer(re.sub(rf'([{rule}])', r" \1 ", text.lower()))
    elif method == "aggressive":
        tokens = aggressive_process(text)
    elif method == "alphabet_only":
        tokens = tokenizer(re.sub(rf'([{rule}])', r" ", text.lower()))
    return tokens


def word2index(word_dict, sent, ignore=True):
    word_index = []
    for word in sent:
        if ignore:
            index = word_dict[word] if word in word_dict else 0
        else:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
            index = word_dict[word]
        word_index.append(index)
    return word_index
