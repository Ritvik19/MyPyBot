import sys, json, re
DIR = 'E:/Models/MyPyBot-Probabilistic'

def load_model():
    with open(f'{DIR}/model.json', 'r') as f:
        data = json.load(f)
        dic = json.loads(data)
        k = dic.keys() 
        v = dic.values() 
        k1 = [eval(i) for i in k] 
        return dict(zip(*[k1,v]))

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

model = load_model()
query = ' '.join(sys.argv[1:])
print(query)
print(predict(query))
