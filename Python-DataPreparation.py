import os
import pandas as pd
from tqdm.auto import tqdm
import re
import pickle
from keras.preprocessing.text import Tokenizer
import time, datetime

def timestamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S:')
    return st

print(timestamp(), 'Libraries Imported')

DOCSTRING_REGEX = re.compile(r"\"\"\"(.|\n)*?\"\"\"")
COMMENT_REGEX = re.compile(r"#.*")
EXTRAWHITESPACES_REGEX = re.compile(r"\s+")

PYTHON_DIR = 'D:\\Users\\Ritvik\\Anaconda3\\envs\\datascience\\lib'

with open('E:/Scrapped-Data/MyPyBot/python_code.txt','a', encoding='utf-8') as input_f:
    for path, directories, files in tqdm(os.walk(PYTHON_DIR)):
        for file in files:
            if file.endswith('.py'):
                try:
                    with open(os.path.join(path, file), 'r') as data_f:
                        contents = DOCSTRING_REGEX.sub('', data_f.read())
                        contents = COMMENT_REGEX.sub('', contents)
                    input_f.write(EXTRAWHITESPACES_REGEX.sub('', contents))
                    input_f.write('\n')
                except Exception as e:
                    print(os.path.join(path, file) ,str(e))
                    
print(timestamp(), 'Data Extracted')             

def get_tokens(corpus):
    tokens = []
    for t in tqdm(word_tokenize(corpus)):
        if re.match(r'\w+', t):
            tokens.append(t)
    return tokens

tokenizer = Tokenizer()

with open('E:/Scrapped-Data/MyPyBot/python_code.txt','r', encoding='utf-8') as f:
    data = f.read()

print(timestamp(), 'File Read')



tokenizer.fit_on_texts(data)
print(timestamp(), 'Tokenizer Trained')

data = tokenizer.texts_to_sequences(data)
print(timestamp(), 'Data Tokenized')

data = [item for sublist in tqdm(data) for item in sublist]
print(timestamp(), 'Token List Flattened')

pickle.dump(tokenizer, open('Tokenizer.pkl', 'wb'))
print(timestamp(), 'Tokenizer Pickled')

data = pd.DataFrame({'tokens':data}).to_csv('E:/Scrapped-Data/MyPyBot/pytokens.csv', index=False)
print('Data Saved')