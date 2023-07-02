from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from decorator import *

from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
import nltk

# nltk.download('wordnet')
# Lemmatize the documents.
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd

from read_data import ReadData, Format


class Preprocessing:
    def __init__(self, docs):
        self.docs = docs

    def tokenize(self, pattern=r'\w+'):
        tokenizer = RegexpTokenizer(pattern)
        for idx in range(len(self.docs)):
            self.docs[idx] = self.docs[idx].lower()  # Convert to lowercase.
            self.docs[idx] = tokenizer.tokenize(self.docs[idx])  # Split into words.

        # Remove numbers, but not words that contain numbers.
        docs = [[token for token in doc if not token.isnumeric()] for doc in self.docs]

        # Remove words that are only one character.
        docs = [[token for token in doc if len(token) > 1] for doc in docs]
        return docs

    def get_word_net_pos(self,tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    def preprocess_data(self, data, isLem=True):
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        sent = ""
        x = nltk.pos_tag((data).split())
        for word, tag in x:
            lemma = lemmatizer.lemmatize(word, pos=self.get_word_net_pos(tag))
            sent += lemma + " "
            data = sent
        preprocessed_data = []
        docs = [[lemmatizer.lemmatize(token,) for token in doc if token not in stop_words] for doc in data]
        print(docs)
        # for document in data:
        #     # Tokenization
        #     tokens = word_tokenize(document)
        #
        #     # Remove stopwords
        #     filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        #
        #     # Stemming
        #     stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        #
        #     # Lemmatization
        #     lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
        #
        #     preprocessed_data.append(lemmatized_tokens)

        return docs


r = ReadData()
url = "https://www.gutenberg.org/files/71037/71037-0.txt"
data_type = Format.TXT
data = r.read_data_from_url(url, data_type)
p = Preprocessing([data])
a = p.tokenize()
p.preprocess_data(a)
