import pandas as pd
from nltk import ngrams
from tqdm.auto import trange
from collections import defaultdict
import json
import os

n = 3
model = defaultdict(lambda: defaultdict(lambda: 0))

for i in trange(38531):
    data = pd.read_csv('../data/all-data.csv', skiprows=i ,nrows=1).values[0][0]
    word_grams = ngrams(str(data).split(), n+1, pad_left=True, pad_right=True, left_pad_symbol='', right_pad_symbol='')
    for w in word_grams:
        model[w[:-1]][w[-1]] += 1

for w1 in model:
    total_count = float(sum(model[w1].values()))
for w2 in model[w1]:
    model[w1][w2] /= total_count
droplist = []
for w1 in model:
    for w2 in model[w1]:
        if model[w1][w2] < 0.1:
            droplist.append((w1, w2))
for w1, w2 in droplist:
    del model[w1][w2]
del droplist

DIR = 'E:/Models/MyPyBot-Probabilistic'
if not os.path.exists(DIR):
    os.mkdir(DIR)

with open(f'{DIR}/model.json', 'w') as f:
    k = model.keys()
    v = model.values()
    k1 = [str(i) for i in k]
    json.dump(json.dumps(dict(zip(*[k1, v]))), f)
