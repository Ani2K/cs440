"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import Counter

def baseline(train, test):
        '''
        input:  training data (list of sentences, with tags on the words)
                test data (list of sentences, no tags on the words)
        output: list of sentences, each sentence is a list of (word,tag) pairs.
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''
        wordtag_dict = {}
        tag_dict = Counter()
        for sentence in train:
                for wordtag in sentence:
                        word = wordtag[0]
                        tag = wordtag[1]
                        if (word not in wordtag_dict):
                                wordtag_dict[word] = Counter()
                        wordtag_dict[word].update({tag:1})
                        tag_dict.update({tag:1})
        defaulttag = tag_dict.most_common(1)[0][0]

        result = []
        for sentence in test:
                result_sentence = []
                for word in sentence:
                        besttag = ''
                        if (word in wordtag_dict):
                                besttag = wordtag_dict[word].most_common(1)[0][0]
                        else:
                                besttag = defaulttag
                        result_sentence.append((word, besttag))
                result.append(result_sentence)
        return result