import spacy
import numpy as np
import _pickle as pkl
import gzip
import os
import sys
import dependency

print("Loading SpaCy...")
nlp = spacy.load('en_core_web_md')
#nlp = spacy.load('en')
print("SpaCy loaded")

output_file = 'pkl/hackabout17.pkl.gz'

folder = '../HoboNet/files/'
files = [folder+'train.txt', folder+'test.txt']

label_dict = {'Other':0, 
              'Message-Topic(e1,e2)':1, 'Message-Topic(e2,e1)':2, 
              'Product-Producer(e1,e2)':3, 'Product-Producer(e2,e1)':4, 
              'Instrument-Agency(e1,e2)':5, 'Instrument-Agency(e2,e1)':6, 
              'Entity-Destination(e1,e2)':7, 'Entity-Destination(e2,e1)':8,
              'Cause-Effect(e1,e2)':9, 'Cause-Effect(e2,e1)':10,
              'Component-Whole(e1,e2)':11, 'Component-Whole(e2,e1)':12,  
              'Entity-Origin(e1,e2)':13, 'Entity-Origin(e2,e1)':14,
              'Member-Collection(e1,e2)':15, 'Member-Collection(e2,e1)':16,
              'Content-Container(e1,e2)':17, 'Content-Container(e2,e1)':18}

def length(file):
    max_len = 0
    i = 0
    for line in open(file):
        splits = line.strip().split('\t')
        label = splits[0]
        pos1 = int(splits[1])
        pos2 = int(splits[2])
        sentence = splits[3]
        le = dependency.shortestPathLength(sentence, pos1, pos2, nlp)
        if le > max_len:
            max_len = le
        i += 1
    return i, max_len 

def get_labels(label):
    l_f = np.zeros((19))
    l_b = np.zeros((19))
    l_c = np.zeros((10))
    num = label_dict[label]
    l_f[num] = 1
    num1 = 0
    if num == 0:
        l_b[num] = 1
        l_c[num] = 1
        return l_c, l_f, l_b

    if num % 2 == 0:
        num1 = num - 1
    else:
        num1 = num + 1
    l_b[num1] = 1
    l_c[(num+1)/2] = 1
    return l_c, l_f, l_b

def read_file(file, lof, max_l):
    i = 0
    words = np.zeros((lof, max_l, 300))
    deps = np.zeros((lof, max_l-1, 50))
    labels_f = np.zeros((lof, 19))
    labels_b = np.zeros((lof, 19))
    labels_c = np.zeros((lof, 10))
    for line in open(file):
        splits = line.strip().split('\t')
        label = splits[0]
        pos1 = int(splits[1])
        pos2 = int(splits[2])
        sentence = splits[3]
        word_vec, dep_vec = dependency.get_vector_embedding(sentence, pos1, pos2, nlp, size=50)
        if np.unique(word_vec).shape[0] == 1:
            continue
        if dep_vec.shape[0] == 0:
            continue
        
        x = max_l - word_vec.shape[0]
        x1 = int(x/2)
        x2 = x - x1
        word_vec = np.lib.pad(word_vec, ((x1,x2), (0,0)), 'constant', constant_values=(0))        
        dep_vec = np.lib.pad(dep_vec, ((x1,x2), (0,0)), 'constant', constant_values=(0))

        words[i] = word_vec
        deps[i] = dep_vec
        labels_c[i], labels_f[i], labels_b[i] = get_labels(label)
        i += 1 
    print(i)
    return [words, deps, labels_c, labels_f, labels_b]

print("Finding length of maximum sentence")
lof_tr, max_l_tr = length(files[0])
lof_te, max_l_te = length(files[1])
max_l = max(max_l_tr, max_l_te)
print("Generating train data")
train_set = read_file(files[0], lof_tr, max_l)
print("Generating test data")
test_set = read_file(files[1], lof_te, max_l)

data = {"train":train_set, "test":test_set}

print("Saving train and test data")
f = gzip.open("./data.pkl", 'wb')
pkl.dump(data, f)
f.close()

print("Data stored in pkl folder")
