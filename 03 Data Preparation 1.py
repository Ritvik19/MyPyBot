from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm
import pandas as pd
import re
print('Libraries Imported')

def get_tokens(corpus):
    tokens = []
    for t in tqdm(word_tokenize(corpus)):
        if re.match(r'\w+', t):
            tokens.append(t)
    return tokens

with open('E:/Scrapped-Data/blogs/posts/sentences.txt', encoding='utf-8') as f:
    data = f.read()
print('File Read')

tokens = get_tokens(data)
print('Data Tokenized', len(tokens))

pd.DataFrame({'tokens':tokens}).to_csv('E:/Scrapped-Data/blogs/posts/tokens.csv', index=False)
print('File Saved')