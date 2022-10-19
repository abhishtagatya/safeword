import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TextClean:

    def __init__(self, text):
        self.text = text
        pass

    def remove_nums(self):
        return re.sub(r'\d+', '', self.text)

    def remove_punctuation(self):
        return self.text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    def remove_whitespace(self):
        return self.text.strip()

    def remove_stopword(self):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(self.text)

        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        return filtered_sentence

    def preprocess(self):
        self.text = self.remove_nums()
        self.text = self.remove_punctuation()
        self.text = self.remove_whitespace()
        self.text = self.remove_stopword()

        return self.text


