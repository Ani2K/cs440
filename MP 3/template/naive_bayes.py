# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=True, silently=False):
    # adjusting values
    # lowercase = True

    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.005, pos_prior=0.8,silently=False):
    # adjusting values
    # pos_prior = 4000 / 5000
    # laplace = 0.005
    
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)


    # map words in pos and neg to freq
    wordmapPos = {}
    wordcountPos = 0
    wordmapNeg = {}
    wordcountNeg = 0
    for i in range(len(train_labels)):
        review = train_set[i]
        for word in review:
            if (train_labels[i] == 1):
                wordcountPos += 1
                if (word not in wordmapPos):
                    wordmapPos[word] = 1
                else:
                    wordmapPos[word] += 1
            else:
                wordcountNeg += 1
                if (word not in wordmapNeg):
                        wordmapNeg[word] = 1
                else:
                    wordmapNeg[word] += 1

    # map words to pos and neg probability, unknown word prob as well
    probmapPos = {}
    numUniquePos = len(wordmapPos)
    probUnknownPos = laplace / (wordcountPos + laplace * (numUniquePos+1))
    for posWord in wordmapPos:
        probmapPos[posWord] = (wordmapPos[posWord] + laplace) / (wordcountPos + laplace * (numUniquePos+1))

    probmapNeg = {}
    numUniqueNeg = len(wordmapNeg)
    probUnknownNeg = laplace / (wordcountNeg + laplace * (numUniqueNeg+1))
    for negWord in wordmapNeg:
        probmapNeg[negWord] = (wordmapNeg[negWord] + laplace) / (wordcountNeg + laplace * (numUniqueNeg+1))
    
    # run on dev set
    yhats = []
    for review in dev_set:
        probPos = np.log(pos_prior)
        probNeg = np.log(1 - pos_prior)
        for word in review:
            if (word in probmapPos):
                logprobPosWord = np.log(probmapPos[word])
                probPos += logprobPosWord
            else:
                logprobPosWord = np.log(probUnknownPos)
                probPos += logprobPosWord
            if (word in probmapNeg):
                logprobNegWord = np.log(probmapNeg[word])
                probNeg += logprobNegWord
            else:
                logprobNegWord = np.log(probUnknownNeg)
                probNeg += logprobNegWord
        if (probPos > probNeg):
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats

# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.005, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8, silently=False):
    # adjusting values
    # pos_prior = 4000 / 5000
    # unigram_laplace = 0.005
    # bigram_laplace = 0.005
    # bigram_lambda = 0.5

    # Keep this in the provided template
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    # unigram prob calculations
    uniwordmapPos = {}
    uniwordcountPos = 0
    uniwordmapNeg = {}
    uniwordcountNeg = 0
    for i in range(len(train_labels)):
        review = train_set[i]
        for word in review:
            if (train_labels[i] == 1):
                uniwordcountPos += 1
                if (word not in uniwordmapPos):
                    uniwordmapPos[word] = 1
                else:
                    uniwordmapPos[word] += 1
            else:
                uniwordcountNeg += 1
                if (word not in uniwordmapNeg):
                        uniwordmapNeg[word] = 1
                else:
                    uniwordmapNeg[word] += 1

    uniprobmapPos = {}
    uninumUniquePos = len(uniwordmapPos)
    uniprobUnknownPos = unigram_laplace / (uniwordcountPos + unigram_laplace * (uninumUniquePos+1))
    for posWord in uniwordmapPos:
        uniprobmapPos[posWord] = (uniwordmapPos[posWord] + unigram_laplace) / (uniwordcountPos + unigram_laplace * (uninumUniquePos+1))

    uniprobmapNeg = {}
    uninumUniqueNeg = len(uniwordmapNeg)
    uniprobUnknownNeg = unigram_laplace / (uniwordcountNeg + unigram_laplace * (uninumUniqueNeg+1))
    for negWord in uniwordmapNeg:
        uniprobmapNeg[negWord] = (uniwordmapNeg[negWord] + unigram_laplace) / (uniwordcountNeg + unigram_laplace * (uninumUniqueNeg+1))

    uni_pos = []
    uni_neg = []
    for review in dev_set:
        uniprobPos = np.log(pos_prior)
        uniprobNeg = np.log(1 - pos_prior)
        for word in review:
            if (word in uniprobmapPos):
                logprobPosWord = np.log(uniprobmapPos[word])
                uniprobPos += logprobPosWord
            else:
                logprobPosWord = np.log(uniprobUnknownPos)
                uniprobPos += logprobPosWord
            if (word in uniprobmapNeg):
                logprobNegWord = np.log(uniprobmapNeg[word])
                uniprobNeg += logprobNegWord
            else:
                logprobNegWord = np.log(uniprobUnknownNeg)
                uniprobNeg += logprobNegWord
        uni_pos.append(uniprobPos)
        uni_neg.append(uniprobNeg)


    # bigram prob calculations
    biwordmapPos = {}
    biwordcountPos = 0
    biwordmapNeg = {}
    biwordcountNeg = 0
    for i in range(len(train_labels)):
        review = train_set[i]
        for wordidx in range(len(review) - 1):
            if (train_labels[i] == 1):
                biwordcountPos += 1
                pair = tuple((review[wordidx], review[wordidx + 1]))
                if (pair not in biwordmapPos):
                    biwordmapPos[pair] = 1
                else:
                    biwordmapPos[pair] += 1
            else:
                biwordcountNeg += 1
                pair = tuple((review[wordidx], review[wordidx + 1]))
                if (pair not in biwordmapNeg):
                    biwordmapNeg[pair] = 1
                else:
                    biwordmapNeg[pair] += 1

    biprobmapPos = {}
    binumUniquePos = len(biwordmapPos)
    biprobUnknownPos = bigram_laplace / (biwordcountPos + bigram_laplace * (binumUniquePos+1))
    for posPair in biwordmapPos:
        biprobmapPos[posPair] = (biwordmapPos[posPair] + bigram_laplace) / (biwordcountPos + bigram_laplace * (binumUniquePos+1))

    biprobmapNeg = {}
    binumUniqueNeg = len(biwordmapNeg)
    biprobUnknownNeg = bigram_laplace / (biwordcountNeg + bigram_laplace * (binumUniqueNeg+1))
    for negPair in biwordmapNeg:
        biprobmapNeg[negPair] = (biwordmapNeg[negPair] + bigram_laplace) / (biwordcountNeg + bigram_laplace * (binumUniqueNeg+1))

    yhats = []
    for i in range(len(dev_set)):
        review = dev_set[i]
        biprobPos = np.log(pos_prior)
        biprobNeg = np.log(1 - pos_prior)
        uniprobPos = uni_pos[i]
        uniprobNeg = uni_neg[i]
        for wordidx in range(len(review) - 1):
            pair = tuple((review[wordidx], review[wordidx + 1]))
            if (pair in biprobmapPos):
                logprobPosWord = np.log(biprobmapPos[pair])
                biprobPos += logprobPosWord
            else:
                logprobPosWord = np.log(biprobUnknownPos)
                biprobPos += logprobPosWord
            if (pair in biprobmapNeg):
                logprobNegWord = np.log(biprobmapNeg[pair])
                biprobNeg += logprobNegWord
            else:
                logprobNegWord = np.log(biprobUnknownNeg)
                biprobNeg += logprobNegWord

        totalprobPos = (1 - bigram_lambda) * uniprobPos + bigram_lambda * biprobPos
        totalprobNeg = (1 - bigram_lambda) * uniprobNeg + bigram_lambda * biprobNeg

        if (totalprobPos > totalprobNeg):
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats

