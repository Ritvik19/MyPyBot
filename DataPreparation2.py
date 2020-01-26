from tqdm.auto import tqdm
import pandas as pd

data = []

for i in tqdm(range(1, 132616)):
    with open(f"../data/datafile-{i}.txt", 'r') as f:
        data.append(f.read())
    if i % 10000 == 0:
        data= list(set(data))
        print()
        print(len(data))
        pd.DataFrame({'text': data}).to_csv(f'../data/batch-{i//10000}.csv', index=False)
        data = []
    
data= list(set(data))
print(len(data))
pd.DataFrame({'text': data}).to_csv(f'../data/batch-{14}.csv', index=False)
