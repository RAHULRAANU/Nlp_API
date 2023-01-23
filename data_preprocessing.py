import os
import pandas as pd
import numpy as np
import swifter
import re
import string
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow
import nltk
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
# from keras.utils import pad_sequences
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


# Text Preprocessing

class text_preprocess:

    def __init__(self):
        pass

    def convert_to_lower(self, text):
        return text.lower()

    def remove_emojis(self, text):
        text = re.sub(r"(?:\@|https?\://)\S+", r" ", text)  # remove links and mentions
        text = re.sub(r"<.*?>", r" ", text)

        wierd_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emotions
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u'\U00010000-\U0010ffff'
                                   u"\u200d"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\u3030"
                                   u"\ufe0f"
                                   u"\u2069"
                                   u"\u2066"
                                   u"\u200c"
                                   u"\u2068"
                                   u"\u2067"
                                   "]+", flags=re.UNICODE)

        rm_emoji = wierd_pattern.sub(r' ', text)
        return rm_emoji

    def remove_html(self, text):
        html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        rm_html = re.sub(html, r' ', text)
        return rm_html

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        URL = url.sub(r' ', text)
        return URL

    def remove_non_ascii(self, text):
        return re.sub(r'[^\x00-\x7f]', r' ', text)  # or ''.join([x for x in text if x in string.printable])

    def remove_numbers(self, text):
        number_pattern = r'\d+'
        without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
        return without_number

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))  # A/c to need give space ' ' , ' '

    def remove_extra_white_spaces(self, text):
        single_char_pattern = re.compile(r'\s+[a-zA-Z]\s+')
        without_sc = re.sub(single_char_pattern, r" ", text)
        #         without_sc = text.replace(' ', '')
        return without_sc

    def preprocessText(self, text):
        return self.remove_extra_white_spaces(self.remove_non_ascii(self.remove_URL(self.remove_html(
            self.remove_punctuation(self.remove_numbers(self.remove_emojis(self.convert_to_lower(text))))))))

if __name__ == "__main__":
    text_prpocess_obj = text_preprocess()
    # df.text = df.text.swifter.apply(lambda x: text_prpocess_obj.preprocessText(x))




# Tokenization of words

from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

# Word Tokenization .
def tokenize(text):
    my_doc = nlp(text)
    token_list = []
    for token in my_doc:
        token_list.append(token.text)
    return token_list







# Remove Stopwords

from nltk.corpus import stopwords
# nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    lst=[]
    for token in text:
        if token not in stop_words:    #checking whether the word is not
            lst.append(token)
    return " ".join(lst)

# df.text = df.text.apply(lambda x:[word for word in x if word not in stop_words])







# Lemmatization

# For remove Whitespace in text
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

lemmatizer = WordNetLemmatizer()

# words = set(nltk.corpus.words.words())
# words = nltk.word_tokenize(corpus)


# nltk.download('words')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

words = set(nltk.corpus.words.words())
# words = nltk.word_tokenize(corpus)


import nltk

class lemmatization:

    def __init__(self):
        pass

    def lemmatizing_space(self, text):
        return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

if __name__ == '__main__':
    lemmatization_obj = lemmatization()
    # df.text = df.text.swifter.apply(lambda x: lemmatization_obj.lemmatizing_space(x))






# Remove string words length between 2

def removelt2wordslength(text):
    for x in text:
        xx = re.compile(r'''\W*\b\w{1,2}\b''')
        rm_word = re.sub(xx, '', text)
        return rm_word




