import networkx as nx
import spacy
import numpy as np

def shortestPathLength(sentence, sourceWordIndex, targetWordIndex, nlp):
    words, dep= get_vector_embedding(sentence, sourceWordIndex, targetWordIndex, nlp)
    return words.shape[0]

def get_vector_embedding(sentence, sourceWordIndex, targetWordIndex, nlp, size=1):

    document = nlp(sentence, parse=True)	
    edges = []
    for token in document:
        for child in token.children:
            edges.append((token, child))

    graph = nx.Graph(edges)

    try:
        path = nx.shortest_path(graph, source=document[sourceWordIndex], target=document[targetWordIndex])
    except Exception:
        return np.zeros((1,300)), np.zeros((1,size))

    words = []
    deps = []
    k = 0
    for k in range(len(path) - 1):
        words.append(path[k].vector)
        if path[k].is_ancestor(path[k+1]):
            deps.append(vectorize(path[k+1].dep, size))
        else:
            deps.append(vectorize(path[k].dep, 50))

    words.append(path[k].vector)
    return np.array(words), np.array(deps)

def vectorize(dep, size):
    np.random.seed(dep)
    return np.random.rand(size)
