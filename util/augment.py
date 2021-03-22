import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm_notebook as tqdm
import os
import re
import pickle
import random
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# Load translation model
en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe').to('cuda')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe').to('cuda')

# Load an En-De Transformer model trained on WMT'19 data:
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe').to('cuda')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe').to('cuda')

stop_words = set(stopwords.words('english'))


def back_translate_ru(text):
    ru = en2ru.translate(text)
    return ru2en.translate(ru)


def back_translate_de(text):
    de = en2de.translate(text)
    return de2en.translate(de)


def back_translate(texts, labels):
    augmented_texts = []
    augmented_labels = []
    for text, label in zip(texts, labels):
        augmented_texts.append(back_translate_ru(text))
        augmented_labels.append(label)
        augmented_texts.append(back_translate_de(text))
        augmented_labels.append(label)
    return augmented_texts, augmented_labels 


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if '_' in l.name() or '-' in l.name():
                continue
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)

    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)


def synonym_replacement(words, num):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >=1 :
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= num:
            break
            
    sentence = ' '.join(new_words)
    return sentence


def word_replacement(texts, labels):
    replaced_texts = []
    replaced_labels = []
    replace_ratio = 0.3
    
    for sent, label in zip(texts, labels):
        words = sent.split(' ')
        num_words = len(words)
        n_sr = max(1, int(replace_ratio*num_words))
        replaced_sent = synonym_replacement(words, n_sr)
        replaced_texts.append(replaced_sent)
        replaced_labels.append(label)        
    return replaced_texts, replaced_labels
    