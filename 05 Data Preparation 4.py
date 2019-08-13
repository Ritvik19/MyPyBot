import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import time, datetime

def timestamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S:')
    return st
tqdm.pandas()

print(timestamp(), 'Libraries Imported')

tokens = pd.read_csv('E:/Scrapped-Data/blogs/posts/token_ids.csv')
print(timestamp(), 'File Read')

train_len = 3 + 1
sequences = []

for i in tqdm(range(train_len, len(tokens)+1)):
    seq = tokens[i-train_len:i]
    sequences.append(seq)

print('Sequenced', len(sequences))

pd.DataFrame(sequences).to_csv('E:/Scrapped-Data/blogs/posts/sequences.csv', index=False)
print('File Saved')