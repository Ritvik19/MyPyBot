# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.

import os
import pandas as pd
from tqdm.auto import tqdm
import re
import dill
import time, datetime
from nltk import ngrams
from collections import Counter, defaultdict

def timestamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S:')
    return st
print(timestamp(), 'Libraries Imported')

with open('../input/python_code.txt','r') as f:
    data = f.read()
print(timestamp(), 'File Read', len(data))

class LanguageModel():
    def __init__(self, n, pad_left=True, pad_right=True, left_pad_symbol='', right_pad_symbol=''):
        self.model = defaultdict(lambda: defaultdict(lambda: 0))
        self.n = n
        self.pad_left= pad_left
        self.pad_right= pad_right
        self.left_pad_symbol= left_pad_symbol 
        self.right_pad_symbol= right_pad_symbol
        
        
    def train(self, dataString):
        word_grams = ngrams(dataString.split(), self.n+1,
                            pad_left=self.pad_left, pad_right=self.pad_right, 
                            left_pad_symbol=self.left_pad_symbol, right_pad_symbol=self.right_pad_symbol)
        for w in word_grams:
            self.model[w[:-1]][w[-1]] += 1
        for w1 in self.model:
            total_count = float(sum(self.model[w1].values()))
            for w2 in self.model[w1]:
                self.model[w1][w2] /= total_count
    
    def predict(self, query):
        try:
            query = tuple(query.split()[-3:])
            return dict(self.model[query])
        except Exception as e:
            print(e)
            return ''

mypybot = LanguageModel(n=3)
mypybot.train(data)
print(timestamp(), 'Model Trained')
# dill.dump(mypybot, open('Model.pkl', 'wb'))
# print(timestamp(), 'Model Pickled')
mypybot.predict(('import numpy as'))
