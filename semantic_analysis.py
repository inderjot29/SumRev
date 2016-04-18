'''
Created on April 17, 2016

@author: inderjot, aprajita
'''

from nltk.corpus import stopwords
import re
from nltk.corpus import wordnet as wn
from itertools import chain
from nltk.tokenize import sent_tokenize, word_tokenize
import sys
import os
import numpy as np
import nltk
from nltk.classify import MaxentClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.metrics import recall,f_measure,precision
from sklearn.svm import SVC,LinearSVC
from sklearn.cross_validation import KFold
import csv

# global variables
trainphraseOpinions={}
testphraseOpinions={}
selectedFeatures={}	


# getting hyponyms of the selected featurewords
def GetHyponyms(index,feature,featureSets):
	features=[]
	features.append(feature)
	ss=wn.synsets(feature)[index]
	hyponyms=list(chain(*[i.lemma_names() for i in ss.hyponyms()]))
	for hyponym in hyponyms:
		features.append(hyponym)
	featureSets[feature]=features
	featureSets.update()
	
#expanding featureset using Wordnet
def GetFeatureSet():
    #following 7 words are taken into account for a feature set.
	featureSet=['staff','service','ambience','food','meal','dish','menu','cost']
	featureSets={}
	for feature in featureSet:
		#extend featureset using direct hyponyms
		#choose appropriate synset based on your knowlegde (kind of hand annotation here)
		if feature=='food':
			GetHyponyms(1,feature,featureSets)
		elif feature == 'service':
			GetHyponyms(0,feature,featureSets)
		elif feature == 'staff':
			GetHyponyms(0,feature,featureSets)
		elif feature == 'meal':
			GetHyponyms(0,feature,featureSets)
		elif feature == 'dish':
			GetHyponyms(1,feature,featureSets)
		elif feature == 'menu':
			GetHyponyms(0,feature,featureSets)
		elif feature == 'cost':
			GetHyponyms(0,feature,featureSets)
			
	return featureSets		
	


	
stoplist = set(stopwords.words('english'))


def RemovePunctAndStopWords(tokens):
	nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
	filtered = [w for w in tokens if nonPunct.match(w)]
	#Remove the stopwords from filtered text
	filtered_words = [word for word in filtered if word.lower() not in stoplist]
	frequent_words=['the','and','of','this','am','etc','also','are','were','was','is']
	filtered_words = [word for word in filtered_words if word.lower() not in frequent_words]
	filtered_words=[word.lower() for word in filtered_words]
	return filtered_words

#returns phrases/aspects which meet the set patterns	
def GetPhrases(taggedTokens):
	phrases=[]
	i=0
	for i in range(0,len(taggedTokens)):
		'''phrases starting with adjective- check for 
		(adjective noun),
		(adjective noun noun),
		(adjective noun verb)'''
		if taggedTokens[i][1]=='JJ':
			if i< len(taggedTokens)-1 and taggedTokens[i+1][1]=='NN':
				if i< len(taggedTokens)-2 and (taggedTokens[i+2][1]=='NN' or taggedTokens[i+2][1]=='VBG' or taggedTokens[i+2][1]=='VBP'):
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]+' '+taggedTokens[i+2][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
				else:
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
		
			'''phrases starting with verb- check for 
			(verb noun)
			(verb adverb)'''
		elif taggedTokens[i][1]=='VBP' or taggedTokens[i][1]=='VBG':
			if  i< len(taggedTokens)-1 and taggedTokens[i+1][1]=='NN':
				phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]
				if phrase is not None and len(phrase)>0:
					phrases.append(phrase)
			elif i< len(taggedTokens)-1 and taggedTokens[i+1][1]=='RB':
				phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]
				if phrase is not None and len(phrase)>0:
					phrases.append(phrase)
			'''phrases starting with adverb- check for 
			(adverb verb),
			(adverb verb noun),
			(adverb adverb adjective),
			(adverb adjective noun),
			(adverb adjective)'''
		elif taggedTokens[i][1]=='RB':
			if i< len(taggedTokens)-1 and (taggedTokens[i+1][1]=='VBP' or taggedTokens[i+1][1]=='VBG') :
				if i< len(taggedTokens)-2 and (taggedTokens[i+2][1]=='NN' or taggedTokens[i+2][1]=='NNS') :
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]+' '+taggedTokens[i+2][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
				else:
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
			elif i< len(taggedTokens)-1 and taggedTokens[i+1][1]=='RB':
				if i< len(taggedTokens)-2 and taggedTokens[i+2][1]=='JJ':
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]+' '+taggedTokens[i+2][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
			elif i< len(taggedTokens)-1 and taggedTokens[i+1][1]=='JJ':
				if i< len(taggedTokens)-2 and taggedTokens[i+2][1]=='NN':
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]+' '+taggedTokens[i+2][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
				else:
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
						
			'''phrases starting with noun- 
			check for 
			(noun verb),
			(noun verb adjective),
			(noun verb adverb),
			(noun adverb adjective),
			(noun adverb verb),
			(noun adjective)'''
		elif taggedTokens[i][1]=='NN' or taggedTokens[i][1]=='NNS':
			if i< len(taggedTokens)-1 and (taggedTokens[i+1][1]=='VBP' or taggedTokens[i+1][1]=='VBG') :
				if i< len(taggedTokens)-2 and taggedTokens[i+2][1]=='JJ':
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]+' '+taggedTokens[i+2][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
				elif i< len(taggedTokens)-2 and taggedTokens[i+2][1]=='RB':
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]+' '+taggedTokens[i+2][0]	
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
				else:
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
			elif i< len(taggedTokens)-1 and taggedTokens[i+1][1]=='RB':
				if i< len(taggedTokens)-2 and taggedTokens[i+2][1]=='JJ':
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]+' '+taggedTokens[i+2][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
				elif i< len(taggedTokens)-2 and (taggedTokens[i+2][1]=='VBG' or taggedTokens[i+2][1]=='VBP') :
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]+' '+taggedTokens[i+2][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
				
			elif i< len(taggedTokens)-1 and taggedTokens[i+1][1]=='JJ':
				phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]
				if phrase is not None and len(phrase)>0:
					phrases.append(phrase)
	return phrases
	
#returns dictionary of phrases along with the opinions based on lexical corpus used
def GetOpinions(phrases,phraseOpinions,fout):
	positiveWords=[]
	negativeWords=[]
	#phraseOpinions={}
	countGood=0
	countBad=0
	countNeutral=0
	for line in open('positive-words.txt'):
		positiveWords.append(line)
		
	for line in open('negative-words.txt'):
		negativeWords.append(line)
	
	positiveWords=map(lambda s: s.strip(), positiveWords)
	negativeWords=map(lambda s: s.strip(), negativeWords)
	i=1
	
	print "start"
	for phrase in phrases:
		polarity=0
		unfiltered_words=phrase.split(' ')
		words=[]
		for word in unfiltered_words:
			if '+' in word:
				new_w=word.split('+')
				for n in new_w:
					words.append(n)
			else:
				words.append(word)
				
		if '****' in phrase :
			if(i==1):
				print "first ****"
				i+=1
			phraseOpinions[phrase]='****'
			#fout.write(str(countGood) +'\t'+str(countBad) +'\t'+str(countNeutral)+'\n')
			fout.write(str(countGood) +'\t'+str(countBad) +'\n')
			#fout.write(str(countGood) +'\n')
			countBad=countGood=countNeutral=0
			
		else :
			for word in words:
				if 'not_' in word:
					word=word.split('not_')[1]
				if word.encode('utf-8') in positiveWords:
					polarity=polarity+1
				elif word.encode('utf-8') in negativeWords:
					polarity=polarity-1
			if polarity>0:
				if 'not_' in phrase:
					if(i==1):
						print "bad:",phrase
					phraseOpinions[phrase]='bad'
					countBad+=1
				else:
					if(i==1):
						print "good:",phrase
					phraseOpinions[phrase]='good'
					countGood+=1
			elif polarity<0:
				if 'not_' in phrase:
					if(i==1):
						print "good:",phrase
					phraseOpinions[phrase]='good'
					countGood+=1
				else:
					if(i==1):
						print "bad:",phrase
					phraseOpinions[phrase]='bad'
					countBad+=1
			else:
				if(i==1):
						print "neutral:",phrase
				phraseOpinions[phrase]='neutral'
				countNeutral+=1
	print "end"
	return phraseOpinions
		
# checks if the phrases obtained contain any of the feature words from featureSet				
def CheckForFeatureWords(allPhrases,featureSet):
	i=1
	flag=0
	for phrase in allPhrases:
		for feature in featureSet:
			for word in featureSet[feature]:
				if word in phrase:
					selectedFeatures[phrase]=feature
					flag=1
		if '****' in phrase:
			flag=1
			selectedFeatures[phrase]='****'
		if flag==0:
			allPhrases.remove(phrase)
		else:
			flag=0
	return allPhrases
	

def word_features(phrase):
	words={}
	for word in phrase.split():
		words[word]=True
	return words
	
def GetFeatureSetForClassifiers(OpinionsSet):
	feature_set=[]
	for phrase in OpinionsSet:
		if OpinionsSet[phrase]=='good':
			feature_set.append((word_features(phrase),'good'))
		elif OpinionsSet[phrase]=='bad':
			feature_set.append((word_features(phrase),'bad'))
		elif OpinionsSet[phrase]=='neutral':
			feature_set.append((word_features(phrase),'neutral'))
	return feature_set
	
	
def CallingClassifiers(training_set,testing_set):
	#print testing_set
	print("training classifier Original Naive Bayes")
	classifier=nltk.NaiveBayesClassifier.train(training_set)
	print("Training classifier Original Naive Bayes completed!")
	print("Original Naive Bayes classifier accuracy percent:",nltk.classify.accuracy(classifier,testing_set)*100)
	
	print(classifier.show_most_informative_features())
	#print(classifier.classify_many(testing_set))
	
	print("training classifier LinearSVC")
	LinearSVC_Classifier=SklearnClassifier(LinearSVC())
	LinearSVC_Classifier.train(training_set)
	print("Training classifier SVC completed!")
	print("Linear SVC classifier accuracy percent:",nltk.classify.accuracy(LinearSVC_Classifier,testing_set)*100)
	
	print("training classifier MaxEnt")
	algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
	MaxEntclassifier = nltk.MaxentClassifier.train(training_set, algorithm,max_iter=3)
	print("Training classifier MaxEnt completed!")
	MaxEntclassifier.show_most_informative_features(10)
	print("Linear MaxEnt classifier accuracy percent:",nltk.classify.accuracy(MaxEntclassifier,testing_set)*100)

# evaluating results using classifiers
def EvaluateUsingClassifiers(OpinionsSet):
	# #obtaining feature_set
	# feature_set=GetFeatureSetForClassifiers(OpinionsSet)
	# print feature_set.shape
	# folds = KFold(len(feature_set), n_folds=4)
	# for train_index,test_index in folds:
		# CallingClassifiers(feature_set[train_index],feature_set[test_index])
	#obtaining feature_set
	feature_set=GetFeatureSetForClassifiers(OpinionsSet)
	
	fLength=len(feature_set)
	
	print "Classification results when train-N =2000,test-N= 2000 "
	training_set=feature_set[0:2000]
	testing_set=feature_set[2001:4000]
	CallingClassifiers(training_set,testing_set)
	
	print "Classification results when train-N =3000,test-N= 1500"
	training_set=feature_set[0:3000]
	testing_set=feature_set[3001:4500]
	CallingClassifiers(training_set,testing_set)
	
	print "Classification results when train-N =5500,test-N= 1000"
	training_set=feature_set[0:5500]
	testing_set=feature_set[5501:6500]
	CallingClassifiers(training_set,testing_set)
	
	print "Classification results when train-N =4409,test-N= 2200"
	training_set=feature_set[2201:6609]
	testing_set=feature_set[0:2200]
	CallingClassifiers(training_set,testing_set)
	
	
# checks if the sentence parsed contains negative values which can change the sentiment of sentence		
def CheckForNOT(tokens):
	for token in tokens:
		if token in ('not',"n't","w'nt","'nt"):
			#print "inside check not if"
			return True
	return False

#writing the results in a CSV file
def WritingResults(OpinionsSet):
	fout=open('Results.csv','w+')
	fout.write('Feature\t Category \t Polarity \n')
	for feature in selectedFeatures:
		fout.write(feature.encode('utf-8'))
		fout.write('\t')
		fout.write(selectedFeatures[feature])
		fout.write('\t')
		fout.write(OpinionsSet[feature])
		fout.write('\n')
		
	fout.close()
	
# Reading reviews from files
def ReadReviews(path,featureSet,fout):
	phraseOpinions={}
	allPhrases=[]
	newPhrases=[]	
	f = open(path, "r")
	lines = f.read().split("\n")
	print "Processing...."
	#for filename in os.listdir(path):
		#print "Reading-",filename
	for line in lines:
		line=unicode(line,'utf-8')
		newPhrase=''
		sentences=line.split('.')
		for sentence in sentences:
			tokens=word_tokenize(sentence)
			isNot=CheckForNOT(tokens)
			#print isNot
			tokens=RemovePunctAndStopWords(tokens)
			taggedTokens=nltk.pos_tag(tokens)
			phrases=GetPhrases(taggedTokens)
			if phrases is not None and len(phrases)>0:
				for phrase in phrases:
					if isNot is True:
						allPhrases.append('not_'+phrase)
						newPhrase='not_'+phrase+' '
						
					else:
						allPhrases.append(phrase)
						newPhrase=phrase+' '
		newPhrases.append(newPhrase)
		allPhrases.append('****')
	
	#checking if these phrases contain feature-words
	updatedPhrases=CheckForFeatureWords(allPhrases,featureSet)
	
	# finding polarity of phrases obtained
	
	phraseOpinions=GetOpinions(updatedPhrases,phraseOpinions,fout)
	
	
	return phraseOpinions

	
def CalculatingPrecisionRecall():
	referenceList=[]
	testList=[]
	with open('Results.csv') as csvfile:
		reader=csv.reader(csvfile)
		for row in reader:
			referenceList.append(row[0])
	with open('Goldstandard.csv') as csvfile:
		reader=csv.reader(csvfile)
		for row in reader:
			testList.append(row[0])
	rlength=len(referenceList)
	tlength=len(testList)
	
	#dividing code on the basis of number of phrases
	i=1
	for i in range(6):
		n= int(rlength*(float(i)/5))
		if n>0:
		#n=n.split('.')[0]
			print "calculating metrics for N=",n
			PrintMetrics(referenceList[0:n],testList[0:n])
	
	
	
def PrintMetrics(referenceList,testList):			
	print "precision-",precision(set(referenceList),set(testList))
	print "recall-",recall(set(referenceList),set(testList))
	print "fmeasure-",f_measure(set(referenceList),set(testList))

	
	
	
	
	
	
		
		
		
	