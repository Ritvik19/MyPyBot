import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import time, datetime
from sys import getsizeof
import pickle
def timestamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S:')
    return st
tqdm.pandas()

print(timestamp(), 'Libraries Imported')

tokens = pd.read_csv('E:/Scrapped-Data/blogs/posts/token_ids.csv')['tokens'].values
print(timestamp(), 'File Read', getsizeof(tokens))

train_len = 3 + 1
sequences = []
for i in tqdm(range(train_len, len(tokens)+1)):
    seq = tokens[i-train_len:i]
    sequences.append(seq)
    
print(timestamp(), 'Sequenced', len(sequences), getsizeof(sequences))

pd.DataFrame(sequences).to_csv('E:/Scrapped-Data/blogs/posts/sequences.csv', index=False)
print(timestamp(), 'File Saved')
