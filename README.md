This a network for solving SemEval2010 Task 8. 
It uses the word embeddings from SpaCy.

The network is a two way bidirectional LSTM with 1-D convolutional filters follows by a combined 21 direction + 10 undirected label prediction loss.

Two instances of the network are combined -- one for the phrase and one for the reversed phrase.

Acheived an accuracy of 84%, but we can boost the accuracy by using better work embeddings.
