# -*- coding: utf-8 -*-
import nltk

import config



if __name__ == '__main__':
    corpora = open(config.corpora_path, 'r')
    words = nltk.word_tokenize(corpora.read())
    text = nltk.Text(words)

    text.concordance('هنر')
