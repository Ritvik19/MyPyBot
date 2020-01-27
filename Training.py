import sys, os
from tqdm.auto import tqdm
import json
from collections import defaultdict

def train(i):
    word_grams = json.load(open(f"../data/datafile-{i}.json"))
    for w in word_grams:
        model[tuple(w[:-1])][w[-1]] += 1

def normalize():
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

def save_model(i):
    with open(f'{DIR}/model-{i}.json', 'w') as f:
        k = model.keys() 
        v = model.values() 
        k1 = [str(i) for i in k]
        json.dump(json.dumps(dict(zip(*[k1,v]))),f)     

model = defaultdict(lambda: defaultdict(lambda: 0))
n = 3
for i in tqdm(range(1, 91532)):
    train(i)
    if i % 10000 == 0:
        save_model(i//10000)
        print()
        print(f"trained on {i} datafiles")
        print(f"size of the model: {sys.getsizeof(model)/(1024*1024):.2f} mega bytes")
    
save_model('_')
normalize()
save_model('')

