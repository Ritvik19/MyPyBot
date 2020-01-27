# MyPyBot

A Probabilistic Word-level Language Model for Python Programming

It predicts the next token based on the previous three tokens

There are two major phases of the approach:
1. Data Preparation : Collecting the python code from the libraries installed on the local system, and cleaning it
2. Data Preprocessing: Creating n-grams
3. Model Building   : Training on the n-grams
___

### About the data:

6,39,553 files of python code, 
<br>
i.e. almost 14.81 GB of python code