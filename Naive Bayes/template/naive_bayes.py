# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
load_data calls the provided utility to load in the dataset.
You can modify the default values for stemming and lowercase, to improve performance when
    we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def calcProb(train_set,train_labels,laplace):
    posN = 0
    negN = 0
    posProb = {}
    negProb = {}
    for i in range(len(train_set)):
        if (train_labels[i] == 1):
            for word in train_set[i]:
                if (word in posProb):
                    posProb[word] +=1
                else:
                    posProb[word] = 1
            posN += 1
        else:
            for word in train_set[i]:
                if (word in negProb):
                    negProb[word] +=1
                else:
                    negProb[word] = 1
            negN += 1
    posProb = {key: value / posN for key, value in posProb.items()}
    negProb = {key: value / negN for key, value in negProb.items()}
    return posProb,negProb
    
    


def naiveBayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.5,silently=False):
    print_paramter_vals(laplace,pos_prior)
    posProb, negProb = calcProb(train_set,train_labels,laplace)
    dict_items = posProb.items()

    first_two = list(dict_items)[:10]
    print(first_two)
    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        yhats.append(-1)
    return yhats





def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=1.0,pos_prior=0.5, silently=False):
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        yhats.append(-1)
    return yhats

