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
NOT_WORDS_REGEX = re.compile(r"\W")                           
EXTRAWHITESPACES_REGEX = re.compile(r"\s+")

PYTHON_DIR = 'D:\\Users\\Ritvik\\Anaconda3\\envs\\datascience\\lib'
count = 0
with open('E:/Scrapped-Data/MyPyBot/python_code.txt','a', encoding='utf-8') as input_f:
    for path, directories, files in tqdm(os.walk(PYTHON_DIR)):
        for file in files:
            if file.endswith('.py'):
                try:
                    with open(os.path.join(path, file), 'r') as data_f:
                        contents = DOCSTRING_REGEX.sub('', data_f.read())
                        contents = COMMENT_REGEX.sub('', contents)
                        contents = NOT_WORDS_REGEX.sub(' ', contents)
                    input_f.write(EXTRAWHITESPACES_REGEX.sub(' ', contents))
                    input_f.write('\n')
                    count += 1
                except Exception as e:
#                     print(os.path.join(path, file) ,str(e))
                    pass
                    
print(timestamp(), 'Data Extracted', count)              

tokenizer = Tokenizer()

with open('E:/Scrapped-Data/MyPyBot/python_code.txt','r', encoding='utf-8') as f:
    data = f.read()
data = data.split()

print(timestamp(), 'File Read', len(data))

tokenizer.fit_on_texts(data)
print(timestamp(), 'Tokenizer Trained')

pickle.dump(tokenizer, open('Tokenizer.pkl', 'wb'))
print(timestamp(), 'Tokenizer Pickled')

tokenizer = pickle.load(open('Tokenizer.pkl', 'rb'))

data = tokenizer.texts_to_sequences(tqdm(data))
print(timestamp(), 'Data Tokenized')

data = [item for sublist in tqdm(data) for item in sublist]
print(timestamp(), 'Token List Flattened', len(data))

print(len(tokenizer.word_counts))

pd.DataFrame({'tokens':data}).to_csv('E:/Scrapped-Data/MyPyBot/py_tokens.csv', index=False)
print('Data Saved')

