import sys, os
from tqdm.auto import tqdm
from nltk import ngrams
import re
import json

DOCSTRING_REGEX = re.compile(r"\"\"\"(.|\n)*?\"\"\"")
COMMENT_REGEX = re.compile(r"#.*")
NOT_WORDS_REGEX = re.compile(r"[^a-zA-Z]")
EXTRAWHITESPACES_REGEX = re.compile(r"\s+")

PYTHON_DIR = ['D:\\Users\\Ritvik\\Anaconda3\\pkgs', 'D:\\Users\\Ritvik\\Anaconda3\\envs\\datascience\\lib', 'D:\\Users\\Ritvik\\Anaconda3\\envs\\nlp_course\\lib',
              'D:\\Users\\Ritvik\\Anaconda3\\envs\\Pyradox\\lib', 'D://Users//Ritvik//Anaconda3//envs\\tensorflow\\lib', 'D:\\Users\\Ritvik\\Anaconda3\\envs\\tfdeeplearning\\lib']


def process(dataString, n, i):
    dataString = DOCSTRING_REGEX.sub(' ', dataString)
    dataString = COMMENT_REGEX.sub(' ', dataString)
    dataString = NOT_WORDS_REGEX.sub(' ', dataString)
    dataString = EXTRAWHITESPACES_REGEX.sub(' ', dataString)
    word_grams = ngrams(dataString.split(), n+1, pad_left=True,
                        pad_right=True, left_pad_symbol='', right_pad_symbol='')
    json.dump(list(word_grams), open(f"../data/datafile-{i}.json", 'w'))


count = 0
bytes_ = 0
for DIR in PYTHON_DIR:
    for path, directories, files in tqdm(os.walk(DIR)):
        for file in files:
            if file.endswith('.py'):
                try:
                    with open(os.path.join(path, file), 'r') as data_f:
                        contents = data_f.read()
                        
                        bytes_ += os.stat(os.path.join(path, file)).st_size
                        count += 1
                        
                        process(contents, 3, count)
                except UnicodeDecodeError:
                    pass
                except Exception as e:
                    print(str(e), os.path.join(path, file))
    print(f"processed {count} files")
    print(f"{bytes_} bytes ({bytes_/(1024*1024)} mega bytes) of data")
                        
