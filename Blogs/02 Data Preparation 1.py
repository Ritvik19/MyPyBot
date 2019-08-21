import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm.auto import tqdm
import re
import time, datetime

def get_tokens(corpus):
    tokens = []
    for t in tqdm(word_tokenize(corpus)):
        if re.match(r'\w+', t):
            tokens.append(t)
    return tokens

def timestamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S:')
    return st

print(timestamp(), 'Libraries Imported')



data = pd.read_csv('E:/Scrapped-Data/blogs/posts/posts.csv').sample(frac=0.1)
print(timestamp(), 'File Read')

sentences = []
for d in tqdm(data.values):
    sentences += sent_tokenize(str(d))
print(timestamp(), 'Sentences Tokenized', len(sentences))

sentences = list(set(sentences))
print(timestamp(), 'Duplicates Removed', len(sentences))

sentences = '.\n'.join(sentences)
print(timestamp(), 'Corpus Created')

tokens = get_tokens(sentences)
print(timestamp(), 'Data Tokenized', len(tokens))

pd.DataFrame({'tokens':tokens}).to_csv('E:/Scrapped-Data/blogs/posts/tokens.csv', index=False)
print(timestamp(), 'File Saved')
