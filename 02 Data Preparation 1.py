import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
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

data = pd.read_csv('E:/Scrapped-Data/blogs/posts/posts.csv')
print('File Read')

sentences = []
for d in tqdm(data.values):
    sentences += sent_tokenize(str(d))
print('Sentences Tokenized', len(sentences))

sentences = list(set(sentences))
print('Duplicates Removed', len(sentences))

sentences = '.\n'.join(sentences)
print('Corpus Created')

tokens = get_tokens(sentences)
print('Data Tokenized', len(tokens))

pd.DataFrame({'tokens':tokens}).to_csv('E:/Scrapped-Data/blogs/posts/tokens.csv', index=False)
print('File Saved')
