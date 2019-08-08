import pandas as pd
from tqdm.auto import tqdm
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
tqdm.pandas()
print('Libraries Imported')

data = pd.read_csv('E:/Scrapped-Data/blogs/posts/posts.csv')
print('File Read')

sentences = []
for d in tqdm(data.values):
    sentences += sent_tokenize(str(d))
print('Sentences Tokenized', len(sentences))

sentences = list(set(sentences))
print('Duplicates Removed', len(sentences))

pd.Series(sentences).progress_apply(lambda x: len(x.split())).plot.hist(bins=150)
plt.show()

with open('E:/Scrapped-Data/blogs/posts/sentences.txt', 'w', encoding='utf-8') as f:
    f.write(' '.join(sentences))
print('File Saved')    

