# Import packages
from nltk import word_tokenize 
from evaluate import load
import numpy as np

import nltk
nltk.download('punkt')

def get_N_w(origin, trans):
    f_in = open(origin, 'r')
    f_out = open(trans, 'r')
    count = 0
    word = 0
    equal = 0
    while count < 100:
        in_1 = f_in.readline()
        out_1 = f_out.readline()
        if not in_1:
            break
        count = count + 1
        word_1 = word_tokenize(in_1)
        word_2 = word_tokenize(out_1)
        equal = equal  + sum(x == y for x, y in zip(word_1, word_2))
        word = word + len(word_1)
    N_w = equal/word
    
    print(N_w)

def get_bert_score(origin, trans):
    f_in = open(origin, 'r', encoding='utf-8')
    f_out = open(trans, 'r', encoding='utf-8')
    predictions = f_in.readlines()[0:100]  
    references = f_out.readlines()[0:100]
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    print(np.mean(results["precision"]))





path1 = "./checkpoints/yelp/non-pri-t/test.rec"
path2 = "./data/yelp/test.txt"
get_N_w(path1, path2)
get_bert_score(path1, path2)

