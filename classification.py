import json, random, os, re, cPickle
import math, collections, itertools, csv
import nltk, nltk.classify.util, nltk.metrics
from multiprocessing import Process, Queue, Pool, cpu_count
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import *
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression,SGDClassifier

pos_data_file = 'datasets/pos-800000.csv.clean_tweet.json'
neg_data_file = 'datasets/neg-800000.csv.clean_tweet.json'
classifier_file = "_classifier.pickle"
features_file = "_feature.pickle"

pos_data = json.load(open(pos_data_file,'r'))
neg_data = json.load(open(neg_data_file,'r'))


def evaluate_features(feature_length, extract_features):

    posFeatures = []
    negFeatures = []

    print "Applying Features..."
    posFeatures = nltk.classify.apply_features(extract_features, pos_data)
    negFeatures = nltk.classify.apply_features(extract_features, neg_data)

    #selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures)*3/4))
    negCutoff = int(math.floor(len(negFeatures)*3/4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]


    
    print "Classifying..."
    #regression classification
    classifier = SklearnClassifier(LogisticRegression())
    classifier.train(trainFeatures)


    # supprot vector classification
    # classifier = SklearnClassifier(LinearSVC())
    # classifier.train(trainFeatures)

    #trains a Naive Bayes Classifier, sci kit classifier
    # classifier = SklearnClassifier(BernoulliNB())
    # classifier.train(trainFeatures)

    # nltk Classifier
    # classifier = NaiveBayesClassifier.train(trainFeatures)  

    cPickle.dump(classifier, open('tests/'+str(feature_length)+classifier_file,'wb'))

    #initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set) 


    print "Training test"
    #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):          
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)
        if( (i+1)%1000 == 0 ):
            print "item %d of %d\n" % ( i+1, len(testFeatures))
    
    features_stats = {
        'training_size' : len(trainFeatures),
        'test_size' : len(testFeatures),
        'accuracy' : nltk.classify.util.accuracy(classifier, testFeatures),
        'pos_precision' : precision(referenceSets['4'], testSets['4']),
        'pos_recall' : recall(referenceSets['4'], testSets['4']),
        'neg_precision' : precision(referenceSets['0'], testSets['0']),
        'neg_recall' : recall(referenceSets['0'], testSets['0'])
    }    

    #prints metrics to show how well the feature selection did
    print 'train on %d instances, test on %d instances' % (features_stats['training_size'], features_stats['test_size'])
    print 'accuracy:', features_stats['accuracy']
    print 'pos precision:', features_stats['pos_precision']
    print 'pos recall:', features_stats['pos_recall']
    print 'neg precision:', features_stats['neg_precision']
    print 'neg recall:', features_stats['neg_recall']

    return features_stats



def create_word_scores():

    #creates lists of all positive and negative words
    posWords = []
    negWords = []

    for tweet,s in pos_data:
        for t in tweet:        
            posWords.append(t)

    for tweet,s in neg_data:
        for t in tweet:        
            negWords.append(t)


    #build frequency distibution of all words and then frequency distributions of words within positive and negative labels
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd[word.lower()] += 1
        cond_word_fd['4'][word.lower()] += 1
    for word in negWords:
        word_fd[word.lower()] += 1
        cond_word_fd['0'][word.lower()] += 1    

    #finds the number of positive and negative words, as well as the total number of words
    pos_word_count = cond_word_fd['4'].N()
    neg_word_count = cond_word_fd['0'].N()
    total_word_count = pos_word_count + neg_word_count


    #builds dictionary of word scores based on chi-squared test
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['4'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['0'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores    


#finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

#creates feature selection mechanism that only uses best words
def best_word_features(tweet):
    return dict([(word, True) for word in tweet if word in best_words])


def tocsvFile(data,filename):
    print "Writing to csv file..."
    keys = data[0].keys()
    with open('results/'+filename+'.csv','wb') as outfile:
        dict_writer = csv.DictWriter(outfile, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)


#tries the best_word_features mechanism with each of the numbers_to_test of features
def run_tests(num):    
    
    feature_set = evaluate_features(num, best_word_features) 
    feature_set['feature_size'] = num   
    print "\n\n"
    return feature_set         

def tests(num_features):
	print 'evaluating best %d word features' % (number_of_features)
	best_words = find_best_words(word_scores, number_of_features)
	print "number of best words " + str(len(best_words))

	cPickle.dump(best_words, open('tests/'+str(number_of_features)+features_file,'wb'))    
	# csvFile.append(run_tests(number_of_features))
	return run_tests(number_of_features)

#finds word scores
word_scores = create_word_scores()    
num_of_words = len(word_scores)
best_words = {}

# numbers of features to select

csvFile = []
tests_to_run = []
for num in range(1, 11):    
    number_of_features = int(math.floor(num_of_words*num/10))
    print number_of_features
    print 'evaluating best %d word features' % (number_of_features)
    best_words = find_best_words(word_scores, number_of_features)
    print "number of best words " + str(len(best_words))

    cPickle.dump(best_words, open('tests/'+str(number_of_features)+features_file,'wb'))    
    csvFile.append(run_tests(number_of_features))


tocsvFile(csvFile,'results')