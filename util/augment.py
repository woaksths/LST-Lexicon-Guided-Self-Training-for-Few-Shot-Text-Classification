import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm_notebook as tqdm
import os
import re
import pickle

# Load translation model
en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe').to('cuda')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe').to('cuda')

# Load an En-De Transformer model trained on WMT'19 data:
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe').to('cuda')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe').to('cuda')


def back_translate(texts, labels):
    augmented_texts = []
    augmented_labels = []
    for text, label in zip(texts, labels):
        print(text)
        augmented_texts.append(back_translate_ru(en2ru, ru2en, text))
        print(back_translate_ru(en2ru, ru2en, text))
        augmented_labels.append(label)
        print(back_translate_de(en2de, de2en, text))
        augmented_texts.append(back_translate_de(en2de, de2en, text))
        augmented_labels.append(label)
        print()
    return augmented_texts, augmented_labels 


def back_translate_ru(en2ru, ru2en, text):
    ru = en2ru.translate(text)
    return ru2en.translate(ru)


def back_translate_de(en2de, de2en, text):
    de = en2de.translate(text)
    return de2en.translate(de)