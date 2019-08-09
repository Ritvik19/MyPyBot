import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
tqdm.pandas()

print('Libraries Imported')

files = [f for f in os.listdir('E:/Scrapped-Data/blogs') if os.path.isfile(os.path.join('E:/Scrapped-Data/blogs', f))]
POST_REGEX = re.compile(r'<post>((.|\s)*?)</post>')

posts = []
for f in tqdm(files):
    with open(os.path.join('E:/Scrapped-Data/blogs', f), encoding='utf8', errors='ignore') as _:
        content = _.read()
    data = POST_REGEX.findall(content)
    for d in data:
        posts.append(d[0].strip())
        
print('Posts Extracted', len(posts))

posts = list(set(posts))
print('Duplicates Removed', len(posts))

pd.Series(posts).progress_apply(lambda x: len(x.split())).plot.hist(bins=150)
plt.show()

pd.DataFrame({'Posts': posts}).to_csv('E:/Scrapped-Data/blogs/posts/posts.csv', index=False)
print('File Saved')