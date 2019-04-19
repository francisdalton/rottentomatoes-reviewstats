# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 08:12:50 2019

@author: S523395
"""

#natural language helper function
import nltk

def cluster_score(cluster):
    sig_words = len(cluster)
    total_words = cluster[-1] - cluster[0] + 1
    return sig_words ** 2 / total_words

def text_freq(term, doc):
    doc = doc.lower().split()
    return doc.count(term.lower())/len(doc)

def inv_doc_freq(term, corpus):
    count = 0
    for doc in corpus:
        if term in doc.lower().split():
            count+=1
            
    if count == 0:
        return 1
    return 1 + log(len(corpus) / count)
    
def tf_idf(term,doc,corpus):
    return text_freq(term,doc)*inv_doc_freq(term,corpus)

def score_sentences(sentences, important_words):
    sentence_scores = []
    for sentence in map(nltk.tokenize.word_tokenize,sentences):
        
        word_idxs = []
        for w in important_words:
            if w in sentence:
                word_idxs.append(sentence.index(w))
        word_idxs.sort()
        if len(word_idxs) > 0:
            word_idxs.sort()
            clusters = []
            cluster = [word_idxs[0]]
            for i in range(1, len(word_idxs)):
                if word_idxs[i] - word_idxs[i-1] < 5:
                    cluster.append(word_idxs[i])
                else:
                    clusters.append(cluster)
                    cluster = [word_idxs[i]]
            clusters.append(cluster)
            
            sentence_scores.append(max(map(cluster_score, clusters)))
        else:
            sentence_scores.append(0)
    return sentence_scores