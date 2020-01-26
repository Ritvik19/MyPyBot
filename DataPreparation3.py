import pandas as pd
from tqdm.auto import tqdm

data= pd.DataFrame()
for i in tqdm(range(1, 15)):
    df = pd.read_csv(f'../data/batch-{i}.csv')
    data = pd.concat([data, df])
    data.drop_duplicates(inplace=True)

print()
print(len(data))
data.to_csv(f'../data/all-data.csv', index=False)