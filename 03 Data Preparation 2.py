import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import re
import pickle
from keras.preprocessing.text import Tokenizer
tqdm.pandas()
print('Libraries Imported')


tokens = pd.read_csv('E:/Scrapped-Data/blogs/posts/tokens.csv')['tokens'].progress_apply(lambda x:str(x).lower()).values.tolist()
print('File Read', len(tokens))

tokenizer = Tokenizer()
print('Tokenizer Instantiated')

tokenizer.fit_on_texts(tokens)
print('Tokenizer Trained')

pickle.dump(tokenizer, open('E:/Tokenizer.pkl', 'wb'))
print('Tokenizer Saved')

tokens = tokenizer.texts_to_sequences(tokens)
print('Indexed', len(tokens))

tokens = [item for sublist in tokens for item in tqdm(sublist)]
print('Flattened', len(tokens))


train_len = 3 + 1
sequences = []

for i in tqdm(range(train_len, len(tokens)+1)):
    seq = tokens[i-train_len:i]
    
    sequences.append(seq)

print('Sequenced', len(sequences))

pd.DataFrame(sequences).to_csv('E:/Scrapped-Data/blogs/posts/sequences.csv', index=False)
print('File Saved')