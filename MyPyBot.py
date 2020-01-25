import sys, os
from tqdm.auto import tqdm
import time, datetime
from nltk import ngrams
import json, re
from collections import Counter, defaultdict

DOCSTRING_REGEX = re.compile(r"\"\"\"(.|\n)*?\"\"\"")
COMMENT_REGEX = re.compile(r"#.*")
NOT_WORDS_REGEX = re.compile(r"[^a-zA-Z]")
EXTRAWHITESPACES_REGEX = re.compile(r"\s+")


PYTHON_DIR = ['D:\\Users\\Ritvik\\Anaconda3\\pkgs', 'D:\\Users\\Ritvik\\Anaconda3\\envs\\datascience\\lib', 'D:\\Users\\Ritvik\\Anaconda3\\envs\\nlp_course\\lib',
             'D:\\Users\\Ritvik\\Anaconda3\\envs\\Pyradox\\lib', 'D://Users//Ritvik//Anaconda3//envs\\tensorflow\\lib', 
             'D:\\Users\\Ritvik\\Anaconda3\\envs\\tfdeeplearning\\lib']

def train(dataString):
    word_grams = ngrams(dataString.split(), n+1, pad_left=True, pad_right=True, left_pad_symbol='', right_pad_symbol='')
    for w in word_grams:
        model[w[:-1]][w[-1]] += 1

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

def load_model():
    with open(f'{DIR}/model.json', 'r') as f:
        data = json.load(f)
        dic = json.loads(data)
        k = dic.keys() 
        v = dic.values() 
        k1 = [eval(i) for i in k] 
        return dict(zip(*[k1,v]))

count = 0
bytes_ = 0
i = 0
for DIR in PYTHON_DIR:
    model = defaultdict(lambda: defaultdict(lambda: 0))
    n = 3
    for path, directories, files in tqdm(os.walk(DIR)):
        for file in files:
            if file.endswith('.py'):
                try:
                    with open(os.path.join(path, file), 'r') as data_f:
                        contents = DOCSTRING_REGEX.sub(' ', data_f.read())
                        contents = COMMENT_REGEX.sub(' ', contents)
                        contents = NOT_WORDS_REGEX.sub(' ', contents)
                        contents = EXTRAWHITESPACES_REGEX.sub(' ', contents)
                        
                        train(contents)
                        
                        bytes_ += os.stat(os.path.join(path, file)).st_size
                        count += 1
                except UnicodeDecodeError:
                    pass
                except Exception as e:
                    print(os.path.join(path, file) ,str(e))
    i += 1                    
    save_model(i)
    print(f"trained on {count} files")
    print(f"{bytes_} bytes ({bytes_/(1024*1024)} mega bytes) of data")
    print(f"size of the model: {sys.getsizeof(model)/(1024*1024)} mega bytes")

def predict(query):
    try:
        DOCSTRING_REGEX = re.compile(r"\"\"\"(.|\n)*?\"\"\"")
        COMMENT_REGEX = re.compile(r"#.*")
        NOT_WORDS_REGEX = re.compile(r"[^a-zA-Z]")
        EXTRAWHITESPACES_REGEX = re.compile(r"\s+")
        query = DOCSTRING_REGEX.sub(' ', query)
        query = COMMENT_REGEX.sub(' ', query)
        query = NOT_WORDS_REGEX.sub(' ', query)
        query = EXTRAWHITESPACES_REGEX.sub(' ', query)
        query = tuple((['', '', '']+query.split())[-3:])
        return dict(model[query])
    except Exception as e:
        print(e)
        return ''