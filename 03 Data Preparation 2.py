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

# train_len = 3 + 1
# sequences = []

# for i in tqdm(range(train_len, len(tokens)+1)):
#     seq = tokens[i-train_len:i]
    
#     sequences.append(seq)

# print('Sequenced', len(sequences))

# pd.DataFrame(sequences).to_csv('E:/Scrapped-Data/blogs/posts/sequences.csv', index=False)
# print('File Saved')
