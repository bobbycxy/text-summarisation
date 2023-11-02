## main_function.py
## README: This is the main python script that houses the finalised functions for 
## all 3 summarisation methods. With the Summarizer class, users need to only
## feed in 2 variables to derive their event extraction via text summarisation.
## The 2 variables are 1) the filepath to the article, and 2) the extraction method.
## An example of how to utilise the Summarizer class is found 
## in the "event-extraction-workbook.ipynb" and the "event-extraction-script.py".
## ==================================================================================

## IMPORT LIBRARIES
import nltk
import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
import json
import re

from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

## KEY PARAMETERS
with open('PARAMETERS.json', 'r') as f:
    parameters = json.load(f)

MAXIMUM_SENTENCE_EXTRACTION = parameters['MAXIMUM SENTENCE EXTRACTION']
MAXIMUM_WORD_CONSIDERATION = parameters['MAXIMUM WORD CONSIDERATION']
MINIMUM_CHARACTER_ABSTRACTION = parameters['MINIMUM CHARACTER ABSTRACTION']
MAXIMUM_CHARACTER_ABSTRACTION = parameters['MAXIMUM CHARACTER ABSTRACTION']
FINETUNED_T5_MODEL_FILEPATH = parameters['FINETUNED T5 MODEL FILEPATH']
FINETUNED_T5_TOKENISATION_FILEPATH = parameters['FINETUNED T5 TOKENISATION FILEPATH']

## PREPROCESS
def preprocess(filepath, substring1 = '<content>', substring2 = '</content>'):
    '''
    inputs:
        filepath: file path to article
        substring1: by default, it is "<content>"
        substring2: by default, it is "</content>"
    output:
        res: the top N ranked sentences
    '''
    with open(filepath, encoding='utf-8') as f:
        article = f.read()

    idx1 = article.index(substring1)
    idx2 = article.index(substring2)

    res = article[idx1 + (len(substring1) - 1) + 1:idx2]
    res = res.strip() 

    return res

## TEXT SUMMARISATION WITH EXTRACTION USING WEIGHTED FREQUENCY
def summarise_weight_freq(text, n, max_sentence_length):
    '''
    inputs:
        text: body of words
        n: [int] number of sentences, [float and lesser than 1] percentage of sentences, [None] 15% of the sentences extracted
        max_sentence_length: keep sentence in the text that have sentence lengths equal or lesser to this
    output:
        summary: the top N ranked sentences
    '''

    sentences = sent_tokenize(text) # tokenize text into a list of sentences
    stop_words = set(stopwords.words('english')) 

    # In order to rank sentences by frequency, we need to have the word frequencies.
    words = [word.lower() for word in word_tokenize(text) if word.lower() not in stop_words and word.isalnum()]
    word_freq = Counter(words)

    # calculate the sentence scores via weighted word frequencies
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = [word.lower() for word in word_tokenize(sentence) if word.lower() not in stop_words and word.isalnum()]
        sentence_score = sum([word_freq[word] for word in sentence_words])
        if len(sentence_words) <= max_sentence_length:
            sentence_scores[sentence] = sentence_score/len(sentence_words) # calculates the average of the sum of word frequencies per sentence

    # get the top n sentences
    if n == None:
        n = int(0.15 * len(sentences)) # rounds down to approximately 15% of the original sentence
    elif isinstance(n,float) and n <= 1:
        n = int(n * len(sentences))
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse = True)[:n]
    summary = ' '.join(summary_sentences)

    return summary

## TEXT SUMMARISATION WITH EXTRACTION USING TF-IDF
def summarise_tfidf(text, n):
    '''
    inputs:
        text: body of words
        n: [int] number of sentences, [float and lesser than 1] percentage of sentences, [None] 15% of the sentences extracted
    output:
        summary: the top N ranked sentences
    '''

    sentences = sent_tokenize(text) # tokenize text into a list of sentences

    # prepare a TF-IDF matrix using sklearn library
    vectorizer = TfidfVectorizer(stop_words = 'english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # calculate the cosine similarity of each sentence against the whole text
    sentence_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    # get the top n sentences
    if n == None:
        n = int(0.15 * len(sentences)) # rounds down to approximately 15% of the original sentence
    elif isinstance(n,float) and n <= 1:
        n = int(n * len(sentences))
    summary_sentences = nlargest(n, range(len(sentence_scores)), key = sentence_scores.__getitem__)
    summary = ' '.join([sentences[i] for i in sorted(summary_sentences)])

    return summary

## TEXT SUMMARISATION WITH ABSTRACTION
def summariser_t5(model_filepath, tokenizer_filepath):
        model = TFAutoModelForSeq2SeqLM.from_pretrained(model_filepath)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_filepath)

        summarizer = pipeline("summarization", model = model, tokenizer = tokenizer, framework = 'tf')
        return summarizer

## SUMMARIZER CLASS
class Summarizer:

    def __init__(self, text, method):
        with open(text, encoding='utf-8') as f:
            article = f.read()
        assert re.findall('<([a-z]+)(?![^>]*\/>)[^>]*>', article) == ['doc','title','content'] and re.findall('</([a-z]+)(?![^>]*\/>)[^>]*>', article) == ['title', 'content', 'doc'] , "The article may not be in the right format of <doc><title>...</title><content>...</content></doc>. Please double check the article again."
        assert method in ['finetuned-t5','weighted-frequency','tf-idf'], "The SUMMARISATION_METHOD is not available or spelt incorrectly. Please use one from ['finetuned-t5','weighted-frequency','tf-idf']."
        
        self.text = preprocess(text)
        self.method = method

    def __str__(self):
        return "We are using '{}' to summarise the following text:\n\n\n'{}'".format(self.method, self.text)
    
    def summarise(self):
        if self.method == 'tf-idf':
            return summarise_tfidf(self.text, MAXIMUM_SENTENCE_EXTRACTION)

        elif self.method == 'weighted-frequency':
            return summarise_weight_freq(self.text, MAXIMUM_SENTENCE_EXTRACTION, MAXIMUM_WORD_CONSIDERATION)
        
        elif self.method == 'finetuned-t5':
            summarizer = summariser_t5(FINETUNED_T5_MODEL_FILEPATH, FINETUNED_T5_TOKENISATION_FILEPATH)
            result = summarizer(self.text, min_lenth = MINIMUM_CHARACTER_ABSTRACTION, max_length = MAXIMUM_CHARACTER_ABSTRACTION)
            return result[0]['summary_text']
        