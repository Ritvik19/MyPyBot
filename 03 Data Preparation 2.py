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

tokenizer = Tokenizer()
print(timestamp(), 'Tokenizer Instantiated')

tokenizer.fit_on_texts(tokens)
print(timestamp(), 'Tokenizer Trained')

pickle.dump(tokenizer, open('E:/Tokenizer.pkl', 'wb'))
print(timestamp(), 'Tokenizer Saved')
