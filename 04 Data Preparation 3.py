import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import re
import pickle
from keras.preprocessing.text import Tokenizer
import time, datetime

def timestamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S:')
    return st
tqdm.pandas()

print(timestamp(), 'Libraries Imported')


tokens = pd.read_csv('E:/Scrapped-Data/blogs/posts/tokens.csv')['tokens'].progress_apply(lambda x:str(x).lower()).values.tolist()
print(timestamp(), 'File Read', len(tokens))

tokenizer = pickle.load(open('E:/Tokenizer.pkl', 'rb'))
print(timestamp(), 'Tokenizer Instantiated')

tokens = tokenizer.texts_to_sequences(tqdm(tokens))
print(timestamp(), 'Indexed', len(tokens))

tokens = [item for sublist in tqdm(tokens) for item in sublist]
print(timestamp(), 'Flattened', len(tokens))

pd.DataFrame({'tokens':tokens}).to_csv('E:/Scrapped-Data/blogs/posts/token_ids.csv', index=False)
print(timestamp(), 'File Saved')
